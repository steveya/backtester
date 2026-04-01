"""Tests for backtester.execution_engine module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.execution import ExecutionBarFrame, TargetSchedule
from backtester.execution_engine import ALIGNED_CLOSE_NEXT_BAR, EventRebalanceEngine


def _bars() -> ExecutionBarFrame:
    frame = pd.DataFrame(
        {
            "ts": [
                pd.Timestamp("2024-03-04 14:30:00+00:00"),
                pd.Timestamp("2024-03-04 14:30:00+00:00"),
                pd.Timestamp("2024-03-04 14:35:00+00:00"),
                pd.Timestamp("2024-03-04 14:35:00+00:00"),
                pd.Timestamp("2024-03-04 14:40:00+00:00"),
                pd.Timestamp("2024-03-04 14:40:00+00:00"),
                pd.Timestamp("2024-03-04 14:45:00+00:00"),
                pd.Timestamp("2024-03-04 14:45:00+00:00"),
            ],
            "asset": ["CL", "BZ", "CL", "BZ", "CL", "BZ", "CL", "BZ"],
            "close": [100.0, 50.0, 101.0, 51.0, 99.0, 52.0, 99.0, 52.0],
        }
    )
    return ExecutionBarFrame(frame)


class TestEventRebalanceEngine:
    def test_aligns_to_next_bar_and_carries_holdings_forward(self) -> None:
        targets = TargetSchedule(
            pd.DataFrame(
                {"CL": [1.0, 0.0], "BZ": [0.0, 1.0]},
                index=pd.DatetimeIndex(
                    [
                        pd.Timestamp("2024-03-04 14:31:00+00:00"),
                        pd.Timestamp("2024-03-04 14:44:00+00:00"),
                    ]
                ),
            )
        )

        result = EventRebalanceEngine().run(_bars(), targets)

        assert result.fill_convention == ALIGNED_CLOSE_NEXT_BAR
        assert result.aligned_targets.index.tolist() == [
            pd.Timestamp("2024-03-04 14:35:00+00:00"),
            pd.Timestamp("2024-03-04 14:45:00+00:00"),
        ]
        assert result.holdings_history.loc[pd.Timestamp("2024-03-04 14:30:00+00:00"), "CL"] == 0.0
        assert result.holdings_history.loc[pd.Timestamp("2024-03-04 14:30:00+00:00"), "BZ"] == 0.0
        assert result.holdings_history.loc[pd.Timestamp("2024-03-04 14:35:00+00:00"), "CL"] == 1.0
        assert result.holdings_history.loc[pd.Timestamp("2024-03-04 14:35:00+00:00"), "BZ"] == 0.0
        assert result.holdings_history.loc[pd.Timestamp("2024-03-04 14:40:00+00:00"), "CL"] == 1.0
        assert result.holdings_history.loc[pd.Timestamp("2024-03-04 14:40:00+00:00"), "BZ"] == 0.0
        assert result.holdings_history.loc[pd.Timestamp("2024-03-04 14:45:00+00:00"), "CL"] == 0.0
        assert result.holdings_history.loc[pd.Timestamp("2024-03-04 14:45:00+00:00"), "BZ"] == 1.0

    def test_realized_returns_follow_previous_holdings(self) -> None:
        targets = TargetSchedule(
            pd.DataFrame(
                {"CL": [1.0], "BZ": [0.0]},
                index=pd.DatetimeIndex([pd.Timestamp("2024-03-04 14:31:00+00:00")]),
            )
        )

        result = EventRebalanceEngine().run(_bars(), targets)

        assert result.portfolio_returns.loc[pd.Timestamp("2024-03-04 14:35:00+00:00")] == 0.0
        np.testing.assert_allclose(
            result.portfolio_returns.loc[pd.Timestamp("2024-03-04 14:40:00+00:00")],
            (99.0 / 101.0) - 1.0,
            atol=1e-12,
        )
        assert result.portfolio_returns.loc[pd.Timestamp("2024-03-04 14:45:00+00:00")] == 0.0

    def test_turnover_and_trade_log_capture_rebalance_deltas(self) -> None:
        targets = TargetSchedule(
            pd.DataFrame(
                {"CL": [1.0, 0.0], "BZ": [0.0, 1.0]},
                index=pd.DatetimeIndex(
                    [
                        pd.Timestamp("2024-03-04 14:31:00+00:00"),
                        pd.Timestamp("2024-03-04 14:44:00+00:00"),
                    ]
                ),
            )
        )

        result = EventRebalanceEngine().run(_bars(), targets)

        assert result.turnover_history.loc[pd.Timestamp("2024-03-04 14:35:00+00:00")] == 1.0
        assert result.turnover_history.loc[pd.Timestamp("2024-03-04 14:45:00+00:00")] == 2.0
        assert set(result.trade_log["asset"]) == {"CL", "BZ"}
        assert len(result.trade_log) == 3

    def test_multiple_decisions_on_same_bar_keep_latest_snapshot(self) -> None:
        targets = TargetSchedule(
            pd.DataFrame(
                {"CL": [1.0, 0.0], "BZ": [0.0, 1.0]},
                index=pd.DatetimeIndex(
                    [
                        pd.Timestamp("2024-03-04 14:31:00+00:00"),
                        pd.Timestamp("2024-03-04 14:34:00+00:00"),
                    ]
                ),
            )
        )

        result = EventRebalanceEngine().run(_bars(), targets)

        assert result.aligned_targets.index.tolist() == [pd.Timestamp("2024-03-04 14:35:00+00:00")]
        assert result.aligned_targets.loc[pd.Timestamp("2024-03-04 14:35:00+00:00"), "CL"] == 0.0
        assert result.aligned_targets.loc[pd.Timestamp("2024-03-04 14:35:00+00:00"), "BZ"] == 1.0
        assert result.event_log["status"].tolist() == ["superseded_same_bar", "aligned"]

    def test_decisions_after_last_bar_are_logged_and_dropped(self) -> None:
        targets = TargetSchedule(
            pd.DataFrame(
                {"CL": [1.0], "BZ": [0.0]},
                index=pd.DatetimeIndex([pd.Timestamp("2024-03-04 15:00:00+00:00")]),
            )
        )

        result = EventRebalanceEngine().run(_bars(), targets)

        assert result.aligned_targets.empty
        assert result.event_log["status"].tolist() == ["dropped_no_execution_bar"]
        assert result.trade_log.empty
        assert (result.holdings_history == 0.0).all().all()

    def test_weight_targets_emit_implicit_cash_history(self) -> None:
        targets = TargetSchedule(
            pd.DataFrame(
                {"CL": [0.25], "BZ": [0.25]},
                index=pd.DatetimeIndex([pd.Timestamp("2024-03-04 14:31:00+00:00")]),
            ),
            target_kind="weights",
        )

        result = EventRebalanceEngine().run(_bars(), targets)

        assert result.cash_history is not None
        assert result.cash_history.loc[pd.Timestamp("2024-03-04 14:35:00+00:00")] == 0.5

    def test_exposure_targets_do_not_infer_cash(self) -> None:
        targets = TargetSchedule(
            pd.DataFrame(
                {"CL": [1.2], "BZ": [-0.4]},
                index=pd.DatetimeIndex([pd.Timestamp("2024-03-04 14:31:00+00:00")]),
            ),
            target_kind="exposures",
        )

        result = EventRebalanceEngine().run(_bars(), targets)

        assert result.cash_history is None
        assert result.target_kind == "exposures"

    def test_rejects_sparse_or_missing_close_matrix(self) -> None:
        bars = ExecutionBarFrame(
            pd.DataFrame(
                {
                    "ts": [
                        pd.Timestamp("2024-03-04 14:30:00+00:00"),
                        pd.Timestamp("2024-03-04 14:35:00+00:00"),
                    ],
                    "asset": ["CL", "BZ"],
                    "close": [100.0, 51.0],
                }
            )
        )
        targets = TargetSchedule(
            pd.DataFrame(
                {"CL": [1.0]},
                index=pd.DatetimeIndex([pd.Timestamp("2024-03-04 14:31:00+00:00")]),
            )
        )

        with pytest.raises(ValueError, match="dense close-price matrix"):
            EventRebalanceEngine().run(bars, targets)
