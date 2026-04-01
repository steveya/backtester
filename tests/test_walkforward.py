"""Tests for execution-clock walk-forward orchestration."""

from __future__ import annotations

import pandas as pd
import pytest

from backtester.cv import WalkForwardCV
from backtester.execution import ExecutionBarFrame, TargetSchedule
from backtester.execution_engine import EventRebalanceEngine
from backtester.objectives import SharpeObjective, TurnoverObjective
from backtester.walkforward import WalkForwardExecutionRunner


def _bars(*, include_last_fold: bool = True) -> ExecutionBarFrame:
    timestamps = pd.DatetimeIndex(
        [
            "2024-01-01T14:00:00Z",
            "2024-01-01T14:05:00Z",
            "2024-01-01T14:10:00Z",
            "2024-01-01T14:15:00Z",
            "2024-01-01T14:20:00Z",
            "2024-01-01T14:25:00Z",
        ]
    )
    if include_last_fold:
        timestamps = timestamps.append(
            pd.DatetimeIndex(
                [
                    "2024-01-01T14:30:00Z",
                    "2024-01-01T14:35:00Z",
                ]
            )
        )
    prices = [100.0, 100.0, 100.0, 100.0, 100.0, 110.0]
    if include_last_fold:
        prices.extend([110.0, 121.0])

    frame = pd.DataFrame(
        {
            "ts": timestamps,
            "asset": ["CL"] * len(timestamps),
            "close": prices,
            "volume": [1000.0] * len(timestamps),
        }
    )
    return ExecutionBarFrame(frame)


def _decision_index() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        [
            "2024-01-01T14:00:00Z",
            "2024-01-01T14:10:00Z",
            "2024-01-01T14:20:00Z",
            "2024-01-01T14:30:00Z",
        ]
    )


def _runner() -> WalkForwardExecutionRunner:
    return WalkForwardExecutionRunner(
        cv=WalkForwardCV(eval_window=1, step=1, min_train=2),
        objectives=[SharpeObjective(), TurnoverObjective()],
        engine=EventRebalanceEngine(),
    )


class TestWalkForwardExecutionRunner:
    def test_fold_context_maps_eval_decisions_to_next_decision_boundary(self) -> None:
        contexts = []

        def target_generator(context):
            contexts.append(context)
            return TargetSchedule(
                pd.DataFrame({"CL": [1.0]}, index=context.eval_decision_times[:1]),
            )

        result = _runner().run(_bars(), _decision_index(), target_generator)

        assert result.n_folds == 2
        first_context = contexts[0]
        assert list(first_context.eval_bars.timestamps) == list(
            pd.DatetimeIndex(
                [
                    "2024-01-01T14:20:00Z",
                    "2024-01-01T14:25:00Z",
                ]
            )
        )
        second_context = contexts[1]
        assert list(second_context.eval_bars.timestamps) == list(
            pd.DatetimeIndex(
                [
                    "2024-01-01T14:30:00Z",
                    "2024-01-01T14:35:00Z",
                ]
            )
        )

    def test_run_aggregates_fold_results_on_multiindex_histories(self) -> None:
        def target_generator(context):
            return TargetSchedule(
                pd.DataFrame({"CL": [1.0]}, index=context.eval_decision_times[:1]),
            )

        result = _runner().run(_bars(), _decision_index(), target_generator)

        assert list(result.per_fold) == ["wf_0", "wf_1"]
        assert result.portfolio_returns.index.nlevels == 2
        assert result.holdings_history.index.nlevels == 2
        assert result.n_eval_bars == 4
        assert result.n_eval_decisions == 2
        assert "sharpe" in result.metrics
        assert "turnover" in result.metrics
        assert result.event_log["fold_id"].tolist() == ["wf_0", "wf_1"]
        assert result.trade_log["fold_id"].tolist() == ["wf_0", "wf_1"]
        assert result.portfolio_returns.loc[
            ("wf_0", pd.Timestamp("2024-01-01T14:25:00Z"))
        ] == pytest.approx(0.1)
        assert result.portfolio_returns.loc[
            ("wf_1", pd.Timestamp("2024-01-01T14:35:00Z"))
        ] == pytest.approx(0.1)

    def test_run_handles_eval_decisions_without_execution_bars(self) -> None:
        def target_generator(context):
            return TargetSchedule(
                pd.DataFrame({"CL": [1.0]}, index=context.eval_decision_times[:1]),
            )

        result = _runner().run(_bars(include_last_fold=False), _decision_index(), target_generator)

        assert result.n_folds == 2
        assert result.per_fold["wf_1"].n_eval_bars == 0
        assert result.per_fold["wf_1"].execution_result.event_log["status"].tolist() == [
            "dropped_no_execution_bar"
        ]
        assert result.event_log["status"].tolist() == ["aligned", "dropped_no_execution_bar"]

    def test_run_preserves_exposure_target_kind_without_cash_history(self) -> None:
        def target_generator(context):
            return TargetSchedule(
                pd.DataFrame({"CL": [1.0]}, index=context.eval_decision_times[:1]),
                target_kind="exposures",
            )

        result = _runner().run(_bars(), _decision_index(), target_generator)

        assert result.target_kind == "exposures"
        assert result.cash_history is None
        assert all(fold.execution_result.cash_history is None for fold in result.per_fold.values())

    def test_run_rejects_targets_outside_eval_decisions(self) -> None:
        def target_generator(context):
            return TargetSchedule(
                pd.DataFrame({"CL": [1.0]}, index=context.train_decision_times[:1]),
            )

        with pytest.raises(ValueError, match="evaluation decision timestamps"):
            _runner().run(_bars(), _decision_index(), target_generator)

    def test_run_returns_empty_result_when_cv_yields_no_folds(self) -> None:
        runner = WalkForwardExecutionRunner(
            cv=WalkForwardCV(eval_window=5, step=1, min_train=10),
            objectives=[SharpeObjective()],
            engine=EventRebalanceEngine(),
        )

        result = runner.run(
            _bars(),
            _decision_index(),
            lambda context: TargetSchedule(pd.DataFrame(columns=["CL"], dtype="float64")),
        )

        assert result.n_folds == 0
        assert result.portfolio_returns.empty
