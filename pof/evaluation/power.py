"""How large an effect this evaluation could have detected at all.

Every experiment in this project reports a score delta against a baseline.
None of them has so far reported the smallest delta the evaluation was
capable of resolving — so a null result has been indistinguishable from an
underpowered one. On HumanEval's 115-item test split at p≈0.85 the minimum
detectable effect is roughly 9pp, which by itself accounts for a long run of
"no significant difference" conclusions: the effects being chased were
several times smaller than the smallest measurable one.

Reporting this next to every result makes the distinction explicit. It is
also the honest frame for the write-up: "we could not detect an effect below
9pp" is a claim about the instrument, not about prompt optimization.
"""
from __future__ import annotations

import math
from statistics import NormalDist


def minimum_detectable_effect(
    p: float, n: int, alpha: float = 0.05, power: float = 0.80
) -> float:
    """Smallest true difference from `p` detectable at `n` samples.

    One-sample normal approximation against a *known* baseline proportion,
    which is the situation here: the baseline is measured once on the same
    fixed test split and then reused, so it carries no additional sampling
    variance in the comparison. The two-sample form would be a factor of
    √2 larger and would overstate the bound.

    Returns a proportion (0.093 = 9.3pp). Returns 1.0 for n <= 0.
    """
    if n <= 0:
        return 1.0
    p = min(max(p, 0.0), 1.0)
    # Variance vanishes at the boundaries, which would report an absurdly
    # small detectable effect off a saturated score. Floor it at the
    # variance of a single flipped item.
    var = max(p * (1.0 - p), 1.0 / n)
    z_alpha = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_beta = NormalDist().inv_cdf(power)
    return (z_alpha + z_beta) * math.sqrt(var / n)


def describe_power(p: float, n: int) -> str:
    """One-line summary for the run log."""
    mde = minimum_detectable_effect(p, n)
    return (
        f"n={n}, score={p:.4f} -> minimum detectable effect "
        f"{mde * 100:.1f}pp at 80% power (alpha=0.05); "
        f"~{math.ceil(mde * n)} items"
    )
