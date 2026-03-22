"""Tests for backtester.statistical module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.statistical import (
    deflated_sharpe_ratio,
    paired_bootstrap_test,
    strategy_clustering,
)


def _sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0


class TestPairedBootstrapSignificant:
    def test_paired_bootstrap_significant(self) -> None:
        rng = np.random.default_rng(42)
        # Use large mean difference and enough samples for significance
        a = pd.Series(rng.normal(0.005, 0.01, 1000))
        b = pd.Series(rng.normal(-0.003, 0.01, 1000))
        result = paired_bootstrap_test(a, b, lambda r: float(r.mean()), n_bootstrap=5000)
        assert result["p_value"] < 0.05


class TestPairedBootstrapInsignificant:
    def test_paired_bootstrap_insignificant(self) -> None:
        rng = np.random.default_rng(42)
        a = pd.Series(rng.normal(0.0, 0.01, 200))
        b = pd.Series(rng.normal(0.0, 0.01, 200))
        result = paired_bootstrap_test(a, b, _sharpe, n_bootstrap=5000)
        assert result["p_value"] > 0.1


class TestDeflatedSharpe:
    def test_deflated_sharpe(self) -> None:
        pytest.importorskip("scipy")
        dsr = deflated_sharpe_ratio(
            sharpe=1.5,
            n_trials=100,
            variance_of_sharpes=1.0,
            n_observations=252,
        )
        assert 0 <= dsr <= 1


class TestStrategyClustering:
    def test_strategy_clustering_two_groups(self) -> None:
        pytest.importorskip("scipy")
        rng = np.random.default_rng(42)
        base_a = rng.normal(0, 0.01, 200)
        base_b = rng.normal(0, 0.01, 200)
        returns = pd.DataFrame({
            "s1": base_a + rng.normal(0, 0.001, 200),
            "s2": base_a + rng.normal(0, 0.001, 200),
            "s3": base_b + rng.normal(0, 0.001, 200),
            "s4": base_b + rng.normal(0, 0.001, 200),
        })
        result = strategy_clustering(returns, n_clusters=2)
        assert len(result) == 4
        assert result["cluster"].nunique() == 2
        # s1/s2 should be in same cluster, s3/s4 in same cluster
        assert result.loc[result["strategy"] == "s1", "cluster"].values[0] == \
               result.loc[result["strategy"] == "s2", "cluster"].values[0]
