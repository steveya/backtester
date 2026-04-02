"""Tests for backtester.accounting."""

from __future__ import annotations

import pandas as pd

from backtester.accounting import (
    held_weights_from_rebalances,
    linear_turnover_costs,
    per_asset_pnl,
    scenario_returns,
)


class TestHeldWeightsFromRebalances:
    def test_expands_sparse_weights_onto_execution_index(self) -> None:
        rebalance_weights = pd.DataFrame(
            {
                "a": [0.6, -0.2],
                "b": [0.4, 0.2],
            },
            index=pd.to_datetime(["2024-01-02", "2024-01-09"]),
        )
        execution_dates = pd.Series(
            pd.to_datetime(["2024-01-03", "2024-01-10"]),
            index=rebalance_weights.index,
        )
        execution_index = pd.bdate_range("2024-01-02", periods=10)

        result = held_weights_from_rebalances(
            rebalance_weights,
            execution_dates,
            execution_index,
            apply_next_bar=False,
        )

        assert result.loc["2024-01-03", "a"] == 0.6
        assert result.loc["2024-01-09", "a"] == 0.6
        assert result.loc["2024-01-10", "a"] == -0.2
        assert result.loc["2024-01-02", "a"] == 0.0


class TestPerAssetPnl:
    def test_multiplies_weights_and_returns(self) -> None:
        index = pd.bdate_range("2024-01-02", periods=2)
        weights = pd.DataFrame({"a": [0.5, 0.5], "b": [-0.2, -0.2]}, index=index)
        returns = pd.DataFrame({"a": [0.01, -0.02], "b": [0.03, 0.01]}, index=index)

        pnl = per_asset_pnl(weights, returns)

        assert pnl.loc[index[0], "a"] == 0.005
        assert pnl.loc[index[1], "b"] == -0.002


class TestLinearTurnoverCosts:
    def test_posts_linear_costs_on_execution_dates(self) -> None:
        rebalance_weights = pd.DataFrame(
            {"a": [0.5, 1.0], "b": [0.5, 0.0]},
            index=pd.to_datetime(["2024-01-02", "2024-01-04"]),
        )
        execution_dates = pd.Series(
            pd.to_datetime(["2024-01-03", "2024-01-05"]),
            index=rebalance_weights.index,
        )
        execution_index = pd.bdate_range("2024-01-02", periods=6)

        costs = linear_turnover_costs(
            rebalance_weights,
            execution_dates,
            execution_index,
            half_spread_bps=5.0,
            multiplier=2.0,
            apply_next_bar=False,
        )

        assert costs.loc["2024-01-03"] == 2.0 * 5.0 * 1e-4 * 1.0
        assert costs.loc["2024-01-05"] == 2.0 * 5.0 * 1e-4 * 1.0
        assert costs.sum() == costs.loc["2024-01-03"] + costs.loc["2024-01-05"]


class TestScenarioReturns:
    def test_returns_and_by_asset_outputs_are_labeled_by_scenario(self) -> None:
        rebalance_weights = pd.DataFrame(
            {"a": [0.5], "b": [0.5]},
            index=pd.to_datetime(["2024-01-02"]),
        )
        execution_dates = pd.Series(
            pd.to_datetime(["2024-01-03"]),
            index=rebalance_weights.index,
        )
        execution_index = pd.bdate_range("2024-01-02", periods=3)
        weights = held_weights_from_rebalances(
            rebalance_weights,
            execution_dates,
            execution_index,
            apply_next_bar=False,
        )
        returns = pd.DataFrame(
            {"a": [0.0, 0.01, 0.0], "b": [0.0, -0.01, 0.02]},
            index=execution_index,
        )

        scenarios, by_asset = scenario_returns(
            weights,
            returns,
            rebalance_weights=rebalance_weights,
            execution_dates=execution_dates,
            cost_multipliers=(1.0, 2.0),
            base_half_spread_bps=5.0,
            apply_next_bar=False,
        )

        assert set(scenarios) == {"1.0x", "2.0x"}
        assert set(by_asset) == {"1.0x", "2.0x"}
        assert by_asset["1.0x"].shape == weights.shape
        assert scenarios["2.0x"].sum() < scenarios["1.0x"].sum()
