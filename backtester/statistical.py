"""Statistical tests for strategy evaluation."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def paired_bootstrap_test(
    returns_a: pd.Series,
    returns_b: pd.Series,
    metric_fn: Callable[[pd.Series], float],
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict:
    """Bootstrap test for difference in metric."""
    rng = np.random.default_rng(seed)
    metric_a = metric_fn(returns_a)
    metric_b = metric_fn(returns_b)
    diff = metric_a - metric_b

    n = min(len(returns_a), len(returns_b))
    a_vals = returns_a.values[:n]
    b_vals = returns_b.values[:n]

    # Bootstrap under the null: resample pairs with replacement,
    # randomly swap assignment to break any real difference
    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        swap = rng.random(n) < 0.5
        a_boot = np.where(swap, b_vals[idx], a_vals[idx])
        b_boot = np.where(swap, a_vals[idx], b_vals[idx])
        boot_diffs[i] = metric_fn(pd.Series(a_boot)) - metric_fn(pd.Series(b_boot))

    # Two-sided p-value
    p_value = float(np.mean(np.abs(boot_diffs) >= abs(diff)))
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    return {
        "metric_a": metric_a,
        "metric_b": metric_b,
        "diff": diff,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    variance_of_sharpes: float,
    n_observations: int,
) -> float:
    """Bailey & Lopez de Prado (2014) deflated Sharpe ratio.

    Returns probability that observed Sharpe exceeds expected maximum
    from n_trials independent strategies.
    """
    try:
        from scipy.stats import norm
    except ImportError:
        raise ImportError(
            "scipy required for deflated_sharpe_ratio. Install with: pip install scipy"
        )

    # Expected max Sharpe from n_trials under null
    e_max_sharpe = np.sqrt(variance_of_sharpes) * (
        (1 - np.euler_gamma) * norm.ppf(1 - 1 / n_trials)
        + np.euler_gamma * norm.ppf(1 - 1 / (n_trials * np.e))
    )

    # DSR: probability that observed > expected max
    se = np.sqrt(variance_of_sharpes / n_observations) if n_observations > 0 else 1e-8
    dsr = float(norm.cdf((sharpe - e_max_sharpe) / max(se, 1e-8)))
    return dsr


def strategy_clustering(
    returns_matrix: pd.DataFrame,
    method: str = "hierarchical",
    n_clusters: int | None = None,
) -> pd.DataFrame:
    """Cluster strategies by return correlation."""
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
    except ImportError:
        raise ImportError("scipy required for strategy_clustering. Install with: pip install scipy")

    corr = returns_matrix.corr()
    dist = np.sqrt(0.5 * (1 - corr.values))
    np.fill_diagonal(dist, 0)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")

    if n_clusters is None:
        n_clusters = max(2, len(returns_matrix.columns) // 2)

    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    return pd.DataFrame(
        {
            "strategy": returns_matrix.columns,
            "cluster": labels,
        }
    )
