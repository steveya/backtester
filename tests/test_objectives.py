"""Tests for backtester.objectives module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.objectives import (
    CalmarObjective,
    CompositeObjective,
    MaxDrawdownObjective,
    Objective,
    SharpeObjective,
    SortinoObjective,
    TurnoverObjective,
)


def _pos_returns(n: int = 252) -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.001, 0.01, n))


def _const_returns(n: int = 100) -> pd.Series:
    return pd.Series([0.001] * n)


class TestSharpePositive:
    def test_sharpe_positive(self) -> None:
        r = _pos_returns()
        assert SharpeObjective().compute(r) > 0


class TestSharpeZeroVol:
    def test_sharpe_zero_vol(self) -> None:
        r = _const_returns()
        assert SharpeObjective().compute(r) == 0.0


class TestSortinoIgnoresUpside:
    def test_sortino_ignores_upside(self) -> None:
        # All positive returns → downside std is 0 → sortino = 0
        r = pd.Series([0.01, 0.02, 0.015, 0.005, 0.01])
        assert SortinoObjective().compute(r) == 0.0


class TestMaxDrawdownKnown:
    def test_max_drawdown_known(self) -> None:
        # 10% drawdown: 1.0, 1.1, 0.99 (10% from peak)
        r = pd.Series([0.0, 0.10, -0.10])
        dd = MaxDrawdownObjective().compute(r)
        assert dd < 0  # negative


class TestTurnoverFromWeights:
    def test_turnover_from_weights(self) -> None:
        wh = pd.DataFrame(
            {
                "a": [0.5, 0.3, 0.6],
                "b": [0.5, 0.7, 0.4],
            }
        )
        obj = TurnoverObjective()
        val = obj.compute(pd.Series([0.01, 0.02, 0.01]), weights_history=wh)
        assert val > 0


class TestTurnoverFromTurnoverHistory:
    def test_turnover_from_turnover_history(self) -> None:
        turnover = pd.Series([0.0, 0.2, 0.4])
        obj = TurnoverObjective()
        val = obj.compute(pd.Series([0.01, 0.02, 0.01]), turnover_history=turnover)
        assert val == pytest.approx(0.2)


class TestCalmar:
    def test_calmar(self) -> None:
        r = _pos_returns()
        val = CalmarObjective().compute(r)
        assert np.isfinite(val)


class TestCompositeWeighted:
    def test_composite_weighted(self) -> None:
        r = _pos_returns()
        comp = CompositeObjective(
            objectives={"sharpe": SharpeObjective(), "dd": MaxDrawdownObjective()},
            weights={"sharpe": 0.7, "dd": 0.3},
        )
        val = comp.compute(r)
        assert np.isfinite(val)


class TestObjectiveSatisfiesProtocol:
    def test_objective_satisfies_protocol(self) -> None:
        for cls in (SharpeObjective, SortinoObjective, MaxDrawdownObjective, CalmarObjective):
            assert isinstance(cls(), Objective)
