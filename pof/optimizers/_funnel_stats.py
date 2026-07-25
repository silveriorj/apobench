"""Cross-run technique statistics for FUNNEL, read from its own accumulated
audit trails. No separate persistence file, no write-concurrency risk:
each run's `audit_funnel_*.json` is already written safely and
independently by `AuditTracker` at the end of every run; this module only
ever reads what's already on disk, so parallel runs across tasks/seeds
never race on a shared mutable file.

This is what makes FUNNEL "empirically reduce over time" rather than
narrow within a single run: every new run scans every prior run's audit
trail under the same experiment root, and only drops a technique once it
has accumulated enough independent observations to judge it with some
statistical confidence — never off one noisy run.
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Tuple


class TechniqueStats(NamedTuple):
    n: int
    mean: float
    std: float


def _load_json(path):
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
        except (json.JSONDecodeError, OSError):
            return None
    return None


def find_history_root(output_dir: str, method_name: str) -> Optional[Path]:
    """Walk up from `output_dir` to the nearest ancestor directory literally
    named `method_name` — the per-method subdirectory every run lands in
    (e.g. `.../funnel/bbh_causal_judgement/seed_42/`) — so history is
    scoped to this method's own runs within the current experiment root,
    not to unrelated older experiments elsewhere under `outputs/`.
    """
    p = Path(output_dir).resolve()
    for candidate in [p, *p.parents]:
        if candidate.name == method_name:
            return candidate
    return None


def scan_technique_stats(
    output_dir: str,
    method_name: str = "funnel",
    technique_names: Optional[Set[str]] = None,
    task_filter: Optional[str] = None,
) -> Dict[str, TechniqueStats]:
    """Aggregate per-technique dev scores from every prior audit trail found
    under this experiment's FUNNEL history root. Returns {} if no history
    root exists yet (first-ever run of the method in this experiment).

    `task_filter` restricts the scan to prior runs of the SAME task (matched
    against the run's directory path). This turns the warm-started priors into
    per-task operator routing: an operator that works on logic tasks and not on
    semantic ones is not averaged into a single misleading global mean.
    Unfiltered (None) pools every task, which is the v1 behavior.
    """
    root = find_history_root(output_dir, method_name)
    if root is None:
        return {}

    scores: Dict[str, List[float]] = {}
    pattern = str(root / "**" / f"audit_{method_name}_*.json")
    for audit_path in glob.glob(pattern, recursive=True):
        if task_filter and task_filter not in Path(audit_path).as_posix():
            continue
        audit = _load_json(audit_path)
        if not audit:
            continue
        recs = (audit.get("history") or {}).get("records") or {}
        for _rid, r in recs.items():
            op = r.get("operator")
            dev = (r.get("scores") or {}).get("dev")
            if op is None or dev is None:
                continue
            if technique_names is not None and op not in technique_names:
                continue
            scores.setdefault(op, []).append(dev)

    stats: Dict[str, TechniqueStats] = {}
    for op, vals in scores.items():
        n = len(vals)
        mean = sum(vals) / n
        if n > 1:
            var = sum((v - mean) ** 2 for v in vals) / (n - 1)
            std = math.sqrt(var)
        else:
            std = 0.0
        stats[op] = TechniqueStats(n=n, mean=mean, std=std)
    return stats


def prune_techniques(
    stats: Dict[str, TechniqueStats],
    all_techniques: List[str],
    min_n: int = 40,
    z_threshold: float = 1.64,
) -> Tuple[List[str], List[str]]:
    """Permanently drop techniques with enough accumulated cross-run history
    (n >= min_n) whose mean dev score sits statistically below this run's
    pool average — a one-sided z-test at `z_threshold` (default 1.64 ~ 95%
    one-sided confidence). Techniques with too little history to judge are
    always kept: pruning only fires once there is real accumulated
    evidence, which is the entire point of scoping it to cross-run history
    instead of a single run's noise.

    Returns (active_techniques, dropped_techniques), both as name lists
    from `all_techniques`, order preserved.
    """
    judged = {op: s for op, s in stats.items() if op in all_techniques and s.n >= min_n}
    if len(judged) < 2:
        return list(all_techniques), []

    total_n = sum(s.n for s in judged.values())
    pool_mean = sum(s.mean * s.n for s in judged.values()) / total_n

    dropped = []
    for op, s in judged.items():
        se = (s.std / math.sqrt(s.n)) if s.std > 0 else 1e-6
        z = (s.mean - pool_mean) / se
        if z <= -z_threshold:
            dropped.append(op)

    active = [t for t in all_techniques if t not in dropped]
    return active, dropped
