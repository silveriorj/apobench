"""APEX-Holdout — APEX with held-out final selection.

Fixes the mechanism identified by two runs this session: `baseline_seed`
(zero search) matched or beat plain APEX on 18/27 BBH tasks (macro mean
+0.4pp for doing nothing), and enlarging the search-visible dev pool to a
50/50 split did not close the gap -- `causal_judgement` under 50/50 scored
*worse* on test (0.6067) than both the unoptimized seed (0.6174) and the
original small-dev APEX (0.6348), despite dev climbing to 0.66.

The root cause isn't dev sample size, it's winner's-curse: APEX runs many
generations x operators x candidates, each round keeping whichever candidate
scored best on dev that round. Argmax over N noisy measurements is biased
upward by however many candidates competed, regardless of how large each
individual measurement's sample is -- FUNNELv2Optimizer measured this
directly (dev 0.927 +/- 0.025 vs test 0.838 +/- 0.062, corr -0.46) and fixed
it with held-out selection: reserve a slice of dev the search never touches,
and use it only to re-rank the top few finalists once search is done. This
class ports that exact mechanism (proven, not new) onto APEX.

Everything the search sees during generations/gates/operators comes from
`self._opt_pool` (70% of dev, `_sample_dev` override below). The remaining
~30% (`self._holdout`, floor MIN_HOLDOUT_N) never influences a single
generation/mutation/gate decision -- it is touched exactly once, at
`_finalize`, to re-rank the top `FINALIZE_TOP_K` contenders by
`AuditHistory` score and report the holdout-best as the run's winner instead
of the (upward-biased) dev-pool argmax.
"""
from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional

from pof.core.types import OptimizationResult, PromptRecord
from pof.optimizers import register_optimizer
from pof.optimizers.apex import APEXOptimizer

logger = logging.getLogger(__name__)

MIN_HOLDOUT_N = 12
FINALIZE_TOP_K = 6


@register_optimizer("apex_holdout")
class APEXHoldoutOptimizer(APEXOptimizer):
    """APEX with a held-out slice reserved for final winner selection.

    Same search (expert init, UCB1 operator bandit, minibatch gate) as
    `APEXOptimizer`; only the dev-sampling source and the finalize step
    differ. See module docstring for the motivating result and mechanism.
    """

    name = "apex_holdout"

    def __init__(self, llm, dataset, evaluator, **kwargs):
        super().__init__(llm=llm, dataset=dataset, evaluator=evaluator, **kwargs)

        full_dev = dataset.get_eval_samples("dev", n=None)
        pool = list(full_dev)
        random.Random(42).shuffle(pool)

        n_dev = len(pool)
        n_opt = max(1, n_dev - max(MIN_HOLDOUT_N, round(0.30 * n_dev)))
        n_opt = min(n_opt, n_dev)
        self._opt_pool: List[Dict[str, str]] = pool[:n_opt]
        self._holdout: List[Dict[str, str]] = pool[n_opt:]
        self._holdout_winner: Optional[PromptRecord] = None

        logger.info(
            f"[{self.name}] dev={n_dev} -> optimization pool={len(self._opt_pool)}, "
            f"held-out selection slice={len(self._holdout)}"
        )

    def _sample_dev(self, n: Optional[int], seed: int = 42) -> List[Dict[str, str]]:
        """Sample only from the search-visible pool -- never the holdout."""
        pool = self._opt_pool
        if n is None or n >= len(pool):
            return list(pool)
        return random.Random(seed).sample(pool, n)

    def _finalize(self) -> None:
        """Re-rank the top contenders on the never-searched holdout slice."""
        if self._finalized:
            return
        self._select_on_holdout()
        self._finalized = True

    def _select_on_holdout(self) -> None:
        if not self._holdout:
            return
        contenders = sorted(
            self.tracker.history.records.values(),
            key=lambda r: r.score, reverse=True,
        )[:FINALIZE_TOP_K]
        contenders = [r for r in contenders if r.text]
        if not contenders:
            return

        for record in contenders:
            res = self.evaluator.evaluate(record.text, self._holdout)
            record.scores["holdout"] = res.score
            record.scores["holdout_n"] = float(len(self._holdout))

        winner = max(contenders, key=lambda r: r.scores.get("holdout", 0.0))
        self._holdout_winner = winner
        opt_best = contenders[0]
        note = (
            f"holdout selection over {len(contenders)} finalists on "
            f"{len(self._holdout)} held-out instances: winner holdout="
            f"{winner.scores.get('holdout', 0.0):.3f} (opt-pool={winner.scores.get('dev', 0.0):.3f}); "
            f"opt-pool argmax would have picked holdout="
            f"{opt_best.scores.get('holdout', 0.0):.3f} (opt-pool={opt_best.scores.get('dev', 0.0):.3f})"
            + ("" if winner is opt_best else " -- DIFFERENT prompt chosen")
        )
        logger.info(f"[{self.name}] {note}")
        self.tracker.add_note(note)

    def optimize(self) -> OptimizationResult:
        """Run the search, then report the held-out winner if one was chosen."""
        result = super().optimize()
        winner = self._holdout_winner
        if winner is not None and winner.text:
            result.best_prompt = winner.text
            result.best_score = winner.scores.get("dev", winner.score)
        return result
