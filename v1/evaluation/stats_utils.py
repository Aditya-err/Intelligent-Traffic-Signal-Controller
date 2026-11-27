"""Statistical analysis utilities for Phase 8 paired controller evaluation."""

from __future__ import annotations

import itertools
import math
from typing import Sequence

import numpy as np


# Two-sided 95% Student-t critical values (alpha = 0.05) for common df
_T_CRIT_95 = {
    1: 12.706205,
    2: 4.302653,
    3: 3.182446,
    4: 2.776445,
    5: 2.570582,
    6: 2.446912,
    7: 2.364624,
    8: 2.306004,
    9: 2.262157,
    10: 2.228139,
    14: 2.144787,
    19: 2.093024,
    29: 2.045230,
}


def t_critical_95(df: int) -> float:
    """Return two-sided 95% Student-t critical value for degrees of freedom ``df``."""
    if df in _T_CRIT_95:
        return _T_CRIT_95[df]
    return 1.959964 if df >= 30 else 2.262157


def compute_summary_stats(values: Sequence[float]) -> dict[str, float]:
    """Compute mean, sample std (ddof=1), and 95% confidence interval half-width."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        raise ValueError("Cannot summarize empty sequence.")
    mean = float(np.mean(arr))
    if n == 1:
        return {"mean": mean, "std": 0.0, "ci95_half": 0.0, "ci95_low": mean, "ci95_high": mean}
    std = float(np.std(arr, ddof=1))
    se = std / math.sqrt(n)
    half = t_critical_95(n - 1) * se
    return {
        "mean": mean,
        "std": std,
        "ci95_half": half,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def _student_t_cdf(t_val: float, df: int) -> float:
    """Numerical CDF of Student's t-distribution via composite Simpson integration."""
    if t_val == 0.0:
        return 0.5
    sign = 1.0 if t_val > 0 else -1.0
    x = abs(t_val)
    coef = math.gamma((df + 1) / 2.0) / (
        math.sqrt(df * math.pi) * math.gamma(df / 2.0)
    )

    def pdf(u: float) -> float:
        return coef * ((1.0 + (u * u) / df) ** (-(df + 1) / 2.0))

    steps = 2000
    h = x / steps
    total = pdf(0.0) + pdf(x)
    for i in range(1, steps):
        total += (4.0 if i % 2 == 1 else 2.0) * pdf(i * h)
    integral = total * (h / 3.0)
    return 0.5 + sign * min(integral, 0.5)


def paired_t_test(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    """Two-sided paired t-test on matched-seed observations (a - b)."""
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("Paired t-test requires equal-length sequences with n >= 2.")
    diffs = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1))
    if std_d == 0.0:
        return {
            "mean_diff": mean_d,
            "t_stat": 0.0 if mean_d == 0.0 else math.copysign(float("inf"), mean_d),
            "p_value": 1.0 if mean_d == 0.0 else 0.0,
        }
    t_stat = mean_d / (std_d / math.sqrt(n))
    cdf = _student_t_cdf(abs(t_stat), n - 1)
    p_val = max(0.0, min(1.0, 2.0 * (1.0 - cdf)))
    return {"mean_diff": mean_d, "t_stat": float(t_stat), "p_value": float(p_val)}


def exact_permutation_test(a: Sequence[float], b: Sequence[float]) -> float:
    """Exact two-sided paired sign-flip permutation test (for n <= 16)."""
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("Permutation test requires equal-length sequences with n >= 2.")
    diffs = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = len(diffs)
    if n > 16:
        raise ValueError("Exact enumeration intended for n <= 16.")
    obs_abs = abs(float(np.mean(diffs)))
    count_extreme = 0
    total = 1 << n
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        perm_mean = abs(float(np.mean(diffs * np.asarray(signs, dtype=np.float64))))
        if perm_mean >= obs_abs - 1e-12:
            count_extreme += 1
    return float(count_extreme / total)


def pct_improvement(
    baseline_val: float, candidate_val: float, *, lower_is_better: bool
) -> float:
    """Compute percentage improvement of candidate relative to baseline."""
    denom = abs(baseline_val)
    if denom == 0.0:
        return 0.0
    if lower_is_better:
        return ((baseline_val - candidate_val) / denom) * 100.0
    return ((candidate_val - baseline_val) / denom) * 100.0
