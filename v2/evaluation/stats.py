"""
Summary & Statistical Analysis for Intelligent-Traffic-RL v2.

Computes:
1. Summary table (mean, std, 95% CI, % improvement vs Fixed-Time)
2. Paired t-test & exact paired sign-flip permutation test (2^10 = 1024 flips)
3. Holm-Bonferroni multiple-comparison correction (raw + corrected p-values)
4. Direction column ("Better than <baseline>" / "Worse than <baseline>")
"""

from __future__ import annotations

import csv
import itertools
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

try:
    from scipy import stats as _scipy_stats
except ImportError:  # pragma: no cover
    _scipy_stats = None


# Two-sided 95% t critical values for small df (exact for df=9 -> n=10)
_T_CRIT_95 = {
    2: 4.3027,
    3: 3.1824,
    4: 2.7764,
    5: 2.5706,
    6: 2.4469,
    7: 2.3646,
    8: 2.3060,
    9: 2.2622,
    10: 2.2281,
}


def mean_std_ci95(values: Sequence[float]) -> Tuple[float, float, float]:
    """Return (mean, sample_std, half_width_95_ci)."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    m = sum(values) / n
    if n == 1:
        return m, 0.0, 0.0
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    sd = math.sqrt(var)
    t_crit = (
        float(_scipy_stats.t.ppf(0.975, df=n - 1))
        if _scipy_stats is not None
        else _T_CRIT_95.get(n - 1, 1.96)
    )
    ci = t_crit * sd / math.sqrt(n)
    return m, sd, ci


def paired_t_test(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Two-sided paired t-test on (a - b). Returns (t_stat, p_value)."""
    if len(a) != len(b) or len(a) < 2:
        return 0.0, 1.0
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    if var_d == 0.0:
        if mean_d == 0.0:
            return 0.0, 1.0
        return math.copysign(float("inf"), mean_d), 0.0
    t_stat = mean_d / math.sqrt(var_d / n)
    if _scipy_stats is not None:
        res = _scipy_stats.ttest_rel(a, b)
        return float(res.statistic), float(res.pvalue)
    # Normal fallback if scipy unavailable
    p_val = math.erfc(abs(t_stat) / math.sqrt(2.0))
    return t_stat, min(1.0, p_val)


def exact_sign_flip_permutation_test(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Exact two-sided paired sign-flip permutation test over all 2^n sign assignments.
    For n=10, 2^10 = 1024 exact permutations.
    """
    if len(a) != len(b) or len(a) == 0:
        return 1.0
    diffs = [x - y for x, y in zip(a, b)]
    obs = abs(sum(diffs) / len(diffs))
    n = len(diffs)
    extreme = 0
    total = 1 << n
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        perm_mean = abs(sum(s * d for s, d in zip(signs, diffs)) / n)
        if perm_mean >= obs - 1e-12:
            extreme += 1
    return extreme / total


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """Apply Holm-Bonferroni step-down adjustment to a list of raw p-values."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [1.0] * m
    running_max = 0.0
    for rank, (orig_idx, p_raw) in enumerate(indexed):
        adj = min(1.0, p_raw * (m - rank))
        running_max = max(running_max, adj)
        corrected[orig_idx] = round(running_max, 6)
    return corrected


def build_summary_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute mean, std, 95% CI, and % improvement vs Fixed-Time per controller x demand."""
    demands = ("low", "medium", "high")
    controllers = ("fixed_time", "actuated", "dqn")
    metrics = (
        "avg_waiting_time_s",
        "avg_queue_length",
        "avg_travel_time_s",
        "throughput_vph",
        "j1_avg_wait_s",
        "j2_avg_wait_s",
        "j3_avg_wait_s",
    )

    fixed_wait_by_demand: Dict[str, float] = {}
    for d in demands:
        f_vals = [
            float(r["avg_waiting_time_s"])
            for r in rows
            if r["controller"] == "fixed_time" and r["demand"] == d
        ]
        if f_vals:
            fixed_wait_by_demand[d] = sum(f_vals) / len(f_vals)

    summary: List[Dict[str, Any]] = []
    for c in controllers:
        for d in demands:
            sub = [r for r in rows if r["controller"] == c and r["demand"] == d]
            if not sub:
                continue
            entry: Dict[str, Any] = {
                "controller": c,
                "demand": d,
                "n_seeds": len(sub),
            }
            for m_key in metrics:
                vals = [float(r[m_key]) for r in sub]
                m, sd, ci = mean_std_ci95(vals)
                entry[f"{m_key}_mean"] = round(m, 2)
                entry[f"{m_key}_std"] = round(sd, 2)
                entry[f"{m_key}_ci95"] = round(ci, 2)

            base_w = fixed_wait_by_demand.get(d, 0.0)
            ctrl_w = entry["avg_waiting_time_s_mean"]
            pct_imp = ((base_w - ctrl_w) / base_w * 100.0) if base_w > 0 else 0.0
            entry["wait_improvement_vs_fixed_pct"] = round(pct_imp, 2)
            summary.append(entry)
    return summary


def build_statistical_tests(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run paired t-test + exact sign-flip permutation test on `avg_waiting_time_s`
    for (actuated vs fixed_time), (dqn vs fixed_time), and (dqn vs actuated)
    across all 3 demand levels, with Holm correction and Direction column.
    """
    comparisons = [
        ("actuated", "fixed_time"),
        ("dqn", "fixed_time"),
        ("dqn", "actuated"),
    ]
    demands = ("low", "medium", "high")

    records: List[Dict[str, Any]] = []
    for ctrl, base in comparisons:
        for d in demands:
            c_map = {
                int(r["seed"]): float(r["avg_waiting_time_s"])
                for r in rows
                if r["controller"] == ctrl and r["demand"] == d
            }
            b_map = {
                int(r["seed"]): float(r["avg_waiting_time_s"])
                for r in rows
                if r["controller"] == base and r["demand"] == d
            }
            common_seeds = sorted(set(c_map.keys()) & set(b_map.keys()))
            if not common_seeds:
                continue
            a_vals = [c_map[s] for s in common_seeds]
            b_vals = [b_map[s] for s in common_seeds]

            mean_a = sum(a_vals) / len(a_vals) if a_vals else 0.0
            mean_b = sum(b_vals) / len(b_vals) if b_vals else 0.0
            diff = mean_a - mean_b  # negative means lower waiting time = BETTER

            t_stat, p_t = paired_t_test(a_vals, b_vals)
            p_perm = exact_sign_flip_permutation_test(a_vals, b_vals)

            records.append(
                {
                    "comparison": f"{ctrl} vs {base}",
                    "demand": d,
                    "n_pairs": len(common_seeds),
                    "ctrl_wait_s": round(mean_a, 2),
                    "baseline_wait_s": round(mean_b, 2),
                    "diff_s": round(diff, 2),
                    "t_stat": round(t_stat, 3),
                    "p_ttest_raw": round(p_t, 6),
                    "p_perm_raw": round(p_perm, 6),
                    "direction": "",
                }
            )

    holm_t = holm_bonferroni([r["p_ttest_raw"] for r in records])
    holm_p = holm_bonferroni([r["p_perm_raw"] for r in records])

    for idx, rec in enumerate(records):
        rec["p_ttest_holm"] = holm_t[idx]
        rec["p_perm_holm"] = holm_p[idx]
        is_sig = holm_t[idx] < 0.05 and holm_p[idx] < 0.05
        rec["significant"] = "Yes" if is_sig else "No"
        if not is_sig:
            rec["direction"] = "No significant difference"
        elif rec["diff_s"] < -1e-6:
            base = rec["comparison"].split(" vs ")[1]
            rec["direction"] = f"Better than {base}"
        elif rec["diff_s"] > 1e-6:
            base = rec["comparison"].split(" vs ")[1]
            rec["direction"] = f"Worse than {base}"
        else:
            rec["direction"] = "No significant difference"

    return records


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
