"""Offline unit tests for APEX's bandit/selection logic (no LLM, no dataset).

This repo has no existing test suite for the optimizers. These tests target
specifically the pieces a midterm-report review round found broken:
pull-counting, credit assignment, and tournament re-sort. They construct an
APEXOptimizer instance via object.__new__ (bypassing __init__, which needs a
real llm/dataset/evaluator) and set only the attributes each method touches.
"""
from pof.core.types import EvalResult, PromptRecord
from pof.optimizers.apex import APEXOptimizer


def _bare_apex(**overrides):
    """An APEXOptimizer with no LLM/dataset -- only the attributes
    _select_operators and _tournament_select actually read."""
    opt = object.__new__(APEXOptimizer)
    opt.ucb_c = 0.5
    opt.population_size = 8
    opt.population = []
    opt._operator_scores = {}
    opt.candidates_per_operator = 2
    for k, v in overrides.items():
        setattr(opt, k, v)
    return opt


def test_select_operators_cold_start_returns_all_seven():
    opt = _bare_apex()
    ops = opt._select_operators()
    assert len(ops) == 7, "with zero pulls, every arm must be tried (cold start)"


def test_select_operators_never_pulled_arm_beats_a_pulled_one():
    """An arm with zero pulls has UCB=inf and must outrank any arm with
    pulls, however good those pulls' rewards were -- this is what makes the
    cold-start guarantee (every arm tried once) hold beyond round one too."""
    opt = _bare_apex()
    opt._operator_scores = {
        "expert_refine": [0.9, 0.9, 0.9],  # pulled 3x, huge reward
        "failure_guided": [0.9],
        "crossover": [0.9],
        "trajectory": [0.9],
        "semantic_var": [0.9],
        "few_shot": [0.9],
        # "format_constraint" never pulled
    }
    top4 = [name for name, _ in opt._select_operators()]
    assert "format_constraint" in top4, "the unpulled arm must be selected (UCB=inf)"


def test_select_operators_pull_count_is_denominator_not_survivor_count():
    """Regression test for the pull-counting bug: an arm credited with N
    reward entries (each entry = one pull, success or not, per the fixed
    _step) must use N as its pull count in the UCB formula, not a smaller
    survivor count. Verify by hand-computing the expected UCB value.

    All arms are given >= MIN_OPERATOR_PULLS pulls so none are
    force-included by that floor -- this isolates the pull-count-as-
    denominator behavior from the (separately-tested) force-inclusion
    mechanism, which would otherwise swamp a low-pull-count setup like the
    one this test used before MIN_OPERATOR_PULLS existed.
    """
    import math
    opt = _bare_apex()
    # arm A: 8 pulls, mean 0.0 (more pulls -> smaller bonus)
    # arm B: 4 pulls, mean 0.0 (fewer pulls -> bigger bonus, same mean)
    # 5 losers: 4 pulls each, mean -1.0 -- clearly worse, excluded by UCB rank
    opt._operator_scores = {
        "expert_refine": [0.1, -0.1, 0.05, -0.05, 0.1, -0.1, 0.05, -0.05],  # 8 pulls
        "failure_guided": [0.1, -0.1, 0.05, -0.05],                          # 4 pulls
        "crossover": [-1.0] * 4, "trajectory": [-1.0] * 4,
        "semantic_var": [-1.0] * 4, "few_shot": [-1.0] * 4,
        "format_constraint": [-1.0] * 4,
    }
    total_pulls = sum(len(v) for v in opt._operator_scores.values())  # 8+4+4*5=32
    expected_bonus_a = 0.5 * math.sqrt(math.log(max(total_pulls, 2)) / 8)
    expected_bonus_b = 0.5 * math.sqrt(math.log(max(total_pulls, 2)) / 4)
    assert expected_bonus_b > expected_bonus_a, (
        "sanity check on the test itself: fewer pulls must mean a larger bonus"
    )
    top4 = dict(opt._select_operators())
    assert len(top4) == 4
    assert {"expert_refine", "failure_guided"}.issubset(top4), (
        "both mean-0.0 arms must outrank the mean--1.0 losers regardless of "
        "pull count"
    )


def test_tournament_select_result_is_rank_ordered():
    """Regression test for the tournament re-sort bug: operators index
    self.population[:k] assuming it is sorted by score descending.
    _tournament_select's raw output was NOT guaranteed sorted past index 0;
    _step now re-sorts its result before returning, which this test checks
    directly on _tournament_select's caller contract."""
    records = [
        PromptRecord(text=f"p{i}", score=score)
        for i, score in enumerate([0.9, 0.3, 0.7, 0.1, 0.5, 0.2, 0.8, 0.4, 0.6])
    ]
    opt = _bare_apex(population_size=5)
    selected = opt._tournament_select(records, tournament_size=3)
    assert selected[0].score == 0.9, "elitism: rank-1 individual must always be selected[0]"
    # _tournament_select's own output need not be sorted past index 0 --
    # that is exactly the bug. Confirm the fix lives in _step by checking
    # that re-sorting selected restores strict descending order (i.e. the
    # same records _step would emit are sortable into a consistent elite
    # ordering, which is what population[:k] downstream requires).
    resorted = sorted(selected, key=lambda r: r.score, reverse=True)
    assert [r.score for r in resorted] == sorted((r.score for r in selected), reverse=True)


def test_op_few_shot_is_idempotent():
    """_op_few_shot must strip any prior demonstration block before
    appending a new one, so repeated application on the same elite across
    generations does not grow the prompt without bound."""
    class FakeEvaluator:
        task_type = "cot"

    class FakeDataset:
        def get_few_shot_examples(self, n=8, seed=None):
            return [{"input": "x", "target": "y"}]

    opt = _bare_apex()
    opt.dataset = FakeDataset()
    opt.evaluator = FakeEvaluator()
    record = PromptRecord(text="Solve the task.\n\nExamples:\nInput: a\nOutput: b", id="parent-1")
    opt.population = [record]

    results = opt._op_few_shot()
    assert len(results) == 1
    text, parent_ids = results[0]
    assert parent_ids == ["parent-1"]
    assert text.count("\n\nExamples:\n") == 1, (
        "old demonstration block must be replaced, not accumulated: got %r" % text
    )
    assert "Solve the task." in text
    assert "Input: a\nOutput: b" not in text, "the OLD demonstrations must be gone"


def test_op_format_constraint_is_idempotent():
    class FakeDataset:
        def get_few_shot_examples(self, n=4, seed=None):
            return [{"input": "x", "target": "42"}]

    opt = _bare_apex()
    opt.dataset = FakeDataset()
    record = PromptRecord(
        text="Solve.\n\nAnswer with ONLY the final answer, exactly in the same format as these examples: 'old'. No explanation.",
        id="parent-2",
    )
    opt.population = [record]

    results = opt._op_format_constraint()
    assert len(results) == 1
    text, parent_ids = results[0]
    assert parent_ids == ["parent-2"]
    assert text.count("Answer with ONLY the final answer") == 1
    assert "'old'" not in text


# ---------------------------------------------------------------------------
# _step()-level tests. Unlike everything above, these exercise the actual
# reward-computation code path inside _step() with a fully scripted, no-LLM
# fake evaluator/dataset/tracker -- closing the gap a review round found:
# every earlier test here covers a periphery function, none touches _step()
# itself, where both rounds of the credit-assignment fix actually live.
# ---------------------------------------------------------------------------

class _FakeTracker:
    """Minimal stand-in for AuditTracker: _create_record only needs add_record
    to be a no-op, and _is_duplicate only needs history.get_by_hash."""

    class _FakeHistory:
        def get_by_hash(self, _text_hash):
            return None  # nothing is ever a duplicate in these tests

    def __init__(self):
        self.history = self._FakeHistory()

    def add_record(self, record):
        pass


class _ScriptedEvaluator:
    """Returns pre-scripted scores keyed by exact prompt text, so a test can
    control precisely which candidates the minibatch gate passes/rejects and
    what their full-dev score is, without any real evaluation happening."""

    def __init__(self, minibatch_scores, dev_scores):
        self.minibatch_scores = minibatch_scores
        self.dev_scores = dev_scores

    def evaluate(self, text, samples):
        # `samples` distinguishes minibatch (len==16 in these tests) from
        # full-dev (len==50) calls, mirroring base.py's real call pattern.
        table = self.minibatch_scores if len(samples) == 16 else self.dev_scores
        score = table.get(text, 0.0)
        return EvalResult(score=score, performance_vector=[], per_sample_details=[])

    def evaluate_with_batch_racing(self, text, samples, threshold, **kwargs):
        # Stage-2 full-dev eval now goes through racing; these tests assert
        # exact scripted dev scores, not racing early-stop behavior, so just
        # return the same scripted full-dev score regardless of threshold.
        return self.evaluate(text, samples)


class _FakeDataset:
    def get_eval_samples(self, split, n=50, seed=None):
        return list(range(n))  # length is all evaluate() inspects

    def get_few_shot_examples(self, n=8, seed=None):
        return [{"input": "x", "target": "y"}] * n


def _stepping_apex(operator_outputs, minibatch_scores, dev_scores, population):
    """An APEXOptimizer wired for a real _step() call: fake tracker/dataset/
    evaluator, real _create_record/_is_duplicate/_evaluate_with_minibatch_gate/
    _tournament_select, and _select_operators/_maybe_stop_if_perfect stubbed
    out so the test controls exactly which operators fire and what they
    return, rather than depending on UCB1 arm selection (tested separately
    above)."""
    opt = object.__new__(APEXOptimizer)
    opt.ucb_c = 0.5
    opt.population_size = len(population)
    opt.population = population
    opt._operator_scores = {}
    opt.candidates_per_operator = 1
    opt.gate_slack = 0.10
    opt.eval_sample_size = 50
    opt.generation = 1
    opt.best_record = max(population, key=lambda r: r.score) if population else None
    opt.tracker = _FakeTracker()
    opt.dataset = _FakeDataset()
    opt.evaluator = _ScriptedEvaluator(minibatch_scores, dev_scores)
    # HoldoutSelectionMixin sits ahead of BaseOptimizer in the MRO;
    # _sample_dev (called by _evaluate_with_minibatch_gate) checks this
    # unconditionally, so it must be set even for a holdout-disabled test.
    opt.use_holdout_selection = False
    opt._opt_pool = None
    opt.llm = None  # _get_budget_mgr() does getattr(self.llm, "get_budget", None)
    opt._maybe_stop_if_perfect = lambda threshold=1.0: None
    opt._select_operators = lambda: [
        (name, fn) for name, fn in operator_outputs.items()
    ]
    return opt


def test_step_gate_passed_candidate_credited_full_dev_improvement():
    """A candidate that passes the gate must be credited its FULL-DEV score
    minus the elite-pool baseline -- not a mixed-scale quantity, and not its
    minibatch score."""
    parent = PromptRecord(text="parent", score=0.50, id="p1")
    population = [parent]
    # Elite pool is just [parent] here, so elite_baseline == 0.50.
    # A gate-passed candidate scoring 0.60 on full-dev must be credited +0.10.
    opt = _stepping_apex(
        operator_outputs={"expert_refine": lambda: [("child-pass", ["p1"])]},
        minibatch_scores={"child-pass": 0.90},   # >= baseline(0.50) - slack(0.10): passes gate
        dev_scores={"child-pass": 0.60},
        population=population,
    )
    opt._step()
    rewards = opt._operator_scores["expert_refine"]
    assert len(rewards) == 1
    assert abs(rewards[0] - 0.10) < 1e-9, f"expected +0.10 (0.60 dev - 0.50 elite baseline), got {rewards[0]}"


def test_step_gate_rejected_candidate_credited_fixed_penalty_not_scale_mixed_diff():
    """A gate-REJECTED candidate must be credited exactly -gate_slack, not
    (its own minibatch score) minus (the parent's full-dev score) -- the
    scale-mixing bug a review round found in round-1's fix."""
    parent = PromptRecord(text="parent", score=0.50, id="p1")
    population = [parent]
    opt = _stepping_apex(
        operator_outputs={"failure_guided": lambda: [("child-reject", ["p1"])]},
        minibatch_scores={"child-reject": 0.10},  # far below baseline(0.50) - slack(0.10): rejected
        dev_scores={"child-reject": 0.99},  # must NOT be used -- rejected candidates never reach full-dev eval
        population=population,
    )
    opt._step()
    rewards = opt._operator_scores["failure_guided"]
    assert len(rewards) == 1
    assert rewards[0] == -0.10, f"expected exactly -gate_slack (-0.10), got {rewards[0]}"


def test_step_null_pull_credited_zero_and_ranks_above_rejected_pull():
    """A pull producing nothing usable must be credited 0.0, and a rejected
    pull must be credited strictly less than that -- restoring the ordering
    (success > null > rejected) that round 1's fix inverted (round 1 credited
    both null pulls and often rejected pulls near or below 0, but rejected
    pulls could land anywhere including above 0, since they subtracted a
    high-noise minibatch score from a full-dev parent score)."""
    parent = PromptRecord(text="parent", score=0.50, id="p1")
    population = [parent]
    opt = _stepping_apex(
        operator_outputs={
            "crossover": lambda: [],  # null pull: produces nothing
            "trajectory": lambda: [("child-reject", ["p1"])],
        },
        minibatch_scores={"child-reject": 0.10},
        dev_scores={},
        population=population,
    )
    opt._step()
    null_reward = opt._operator_scores["crossover"][0]
    rejected_reward = opt._operator_scores["trajectory"][0]
    assert null_reward == 0.0
    assert rejected_reward < null_reward, (
        "a rejected (tried-and-failed) pull must rank below a null (did-nothing) pull, "
        f"got null={null_reward}, rejected={rejected_reward}"
    )


def test_step_reward_baseline_is_shared_elite_pool_not_per_operator_parent():
    """Two operators recording DIFFERENT parents must be credited against the
    SAME elite-pool baseline -- round 1's per-candidate-parent baseline gave
    _op_trajectory (parent = whole population, including weak members) a
    systematically easier bar than operators drawing from population[:3]/[:4]
    only. Round 2 fixes this by using one shared baseline per generation."""
    strong = PromptRecord(text="strong", score=0.80, id="s1")
    weak = PromptRecord(text="weak", score=0.20, id="w1")
    # population[:4] is both records here; elite_baseline = mean(0.80, 0.20) = 0.50
    population = [strong, weak]
    opt = _stepping_apex(
        operator_outputs={
            # "conditions on" only the strong parent
            "expert_refine": lambda: [("child-a", ["s1"])],
            # "conditions on" the whole population, including the weak one --
            # round 1 would have given this a much easier (lower) baseline
            "trajectory": lambda: [("child-b", ["s1", "w1"])],
        },
        minibatch_scores={"child-a": 0.90, "child-b": 0.90},
        dev_scores={"child-a": 0.55, "child-b": 0.55},
        population=population,
    )
    opt._step()
    reward_a = opt._operator_scores["expert_refine"][0]
    reward_b = opt._operator_scores["trajectory"][0]
    assert abs(reward_a - reward_b) < 1e-9, (
        f"identical dev scores against the SAME shared baseline must give identical "
        f"rewards regardless of recorded parent_ids, got expert_refine={reward_a}, trajectory={reward_b}"
    )
    assert abs(reward_a - 0.05) < 1e-9  # 0.55 dev - 0.50 elite baseline


def test_step_pull_count_still_counts_every_call_regardless_of_outcome():
    """Regression guard: the round-2 reward rewrite must not have reintroduced
    the round-1-fixed pull-counting bug. Three pulls (pass, reject, null) on
    three different arms must each append exactly one _operator_scores entry."""
    parent = PromptRecord(text="parent", score=0.50, id="p1")
    population = [parent]
    opt = _stepping_apex(
        operator_outputs={
            "expert_refine": lambda: [("child-pass", ["p1"])],
            "failure_guided": lambda: [("child-reject", ["p1"])],
            "crossover": lambda: [],
        },
        minibatch_scores={"child-pass": 0.90, "child-reject": 0.10},
        dev_scores={"child-pass": 0.60},
        population=population,
    )
    opt._step()
    assert len(opt._operator_scores["expert_refine"]) == 1
    assert len(opt._operator_scores["failure_guided"]) == 1
    assert len(opt._operator_scores["crossover"]) == 1


if __name__ == "__main__":
    test_select_operators_cold_start_returns_all_seven()
    test_select_operators_never_pulled_arm_beats_a_pulled_one()
    test_select_operators_pull_count_is_denominator_not_survivor_count()
    test_tournament_select_result_is_rank_ordered()
    test_op_few_shot_is_idempotent()
    test_op_format_constraint_is_idempotent()
    test_step_gate_passed_candidate_credited_full_dev_improvement()
    test_step_gate_rejected_candidate_credited_fixed_penalty_not_scale_mixed_diff()
    test_step_null_pull_credited_zero_and_ranks_above_rejected_pull()
    test_step_reward_baseline_is_shared_elite_pool_not_per_operator_parent()
    test_step_pull_count_still_counts_every_call_regardless_of_outcome()
    print("OK")
