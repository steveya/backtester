"""Tests for backtester.analytics module."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.analytics import (
    drawdown_table,
    param_sensitivity,
    param_stability,
    performance_table,
    rolling_metrics,
)


def _returns(n: int = 252) -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.0005, 0.01, n), index=pd.bdate_range("2024-01-02", periods=n))


class TestPerformanceTableColumns:
    def test_performance_table_columns(self) -> None:
        r = _returns()
        t = performance_table(r)
        expected = {
            "annualized_return", "annualized_vol", "sharpe", "sortino",
            "calmar", "max_drawdown", "max_drawdown_duration_days",
            "win_rate", "profit_factor", "skew", "kurtosis",
            "best_day", "worst_day",
        }
        assert expected.issubset(set(t.columns))


class TestRollingMetricsShape:
    def test_rolling_metrics_shape(self) -> None:
        r = _returns()
        rm = rolling_metrics(r, window=63)
        assert len(rm) == len(r)
        assert "rolling_sharpe" in rm.columns


class TestDrawdownTableTopN:
    def test_drawdown_table_top_n(self) -> None:
        r = _returns()
        dt = drawdown_table(r, top_n=3)
        assert len(dt) <= 3
        if not dt.empty:
            # Sorted by depth (most negative first)
            assert dt["depth"].iloc[0] <= dt["depth"].iloc[-1]


class TestParamStabilityLowCV:
    def test_param_stability_low_cv(self) -> None:
        # All folds have nearly the same params → low CV
        folds = [{"comp": {"window": 50.0 + i * 0.1}} for i in range(5)]
        ps = param_stability(folds)
        assert not ps.empty
        assert ps["cv"].iloc[0] < 0.1


class TestParamSensitivityShape:
    def test_param_sensitivity_shape(self) -> None:
        def _eval(params):
            w = params["comp"]["window"]
            return -((w - 80) ** 2)

        ps = param_sensitivity(
            _eval,
            {"comp": {"window": 50.0}},
            "window",
            "comp",
            n_points=10,
        )
        assert len(ps) == 10
        assert "value" in ps.columns
        assert "score" in ps.columns
