"""Peer-review-grade statistical utilities for evaluation results."""

import numpy as np
from typing import List, Tuple


def compute_result(
    scores: List[float],
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float, float, int, str]:
    """
    Bootstrap confidence interval for a list of binary (or continuous) scores.

    Returns
    -------
    mean, ci_lower, ci_upper, std, n, method
    where method is "bootstrap" or "exact" (used when n < 10).
    """
    if len(scores) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0, "n/a"

    arr = np.array(scores, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0)

    if len(arr) < 10:
        return mean, max(0.0, mean - std), min(1.0, mean + std), std, len(arr), "exact"

    rng = np.random.default_rng(seed)
    boot_means = rng.choice(arr, size=(n_bootstrap, len(arr)), replace=True).mean(axis=1)

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(boot_means, alpha / 2 * 100))
    ci_upper = float(np.percentile(boot_means, (1 - alpha / 2) * 100))

    return mean, ci_lower, ci_upper, std, len(arr), "bootstrap"


def compute_comparison(
    scores_a: List[float],
    scores_b: List[float],
    n_permutations: int = 10_000,
    n_comparisons: int = 1,
    seed: int = 42,
) -> Tuple[float, float, float, Tuple[float, float], bool]:
    """
    Two-sample permutation test with Cohen's d and 95% CI on the difference.

    Parameters
    ----------
    scores_a, scores_b : lists of per-sample scores (0/1 or continuous)
    n_permutations : number of permutation iterations (default 10 000)
    n_comparisons : denominator for Bonferroni correction (default 1 = no correction)
    seed : RNG seed for reproducibility

    Returns
    -------
    delta, p_value, cohens_d, ci_delta, significant
    where delta = mean_a - mean_b (positive = a is higher).
    """
    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)

    mean_a, mean_b = np.mean(a), np.mean(b)
    delta = float(mean_a - mean_b)

    na, nb = len(a), len(b)
    var_a = float(np.var(a, ddof=1)) if na > 1 else 0.0
    var_b = float(np.var(b, ddof=1)) if nb > 1 else 0.0
    pooled_sd = float(np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))) if (na + nb > 2) else 0.0
    cohens_d = delta / pooled_sd if pooled_sd > 0 else 0.0

    # Permutation test
    rng = np.random.default_rng(seed)
    combined = np.concatenate([a, b])
    perm_deltas = np.empty(n_permutations)
    for i in range(n_permutations):
        rng.shuffle(combined)
        perm_deltas[i] = combined[:na].mean() - combined[na:].mean()
    p_value = float(np.mean(np.abs(perm_deltas) >= np.abs(delta)))

    # Bonferroni-corrected significance
    significant = p_value < (0.05 / max(1, n_comparisons))

    # 95% CI on delta via normal approximation (Welch)
    se_delta = np.sqrt(var_a / na + var_b / nb) if (na > 0 and nb > 0) else 0.0
    ci_delta = (delta - 1.96 * se_delta, delta + 1.96 * se_delta)

    return delta, p_value, cohens_d, ci_delta, significant


def format_metric(
    mean: float,
    ci_lower: float,
    ci_upper: float,
    as_pct: bool = True,
    decimals: int = 1,
) -> str:
    """
    Format a metric as "42.3 [38.1, 46.5]" (percentage) or "0.423 [0.381, 0.465]".
    Suitable for LaTeX table cells.
    """
    scale = 100.0 if as_pct else 1.0
    fmt = f"{{:.{decimals}f}}"
    suffix = "\\%" if as_pct else ""
    m = fmt.format(mean * scale) + suffix
    lo = fmt.format(ci_lower * scale)
    hi = fmt.format(ci_upper * scale)
    return f"{m} [{lo}, {hi}]"