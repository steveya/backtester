"""Tests for backtester.execution module."""

from __future__ import annotations

import pandas as pd
import pytest

from backtester.execution import ExecutionBarFrame, TargetSchedule


def _bars_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "obs_time": [
                pd.Timestamp("2024-03-01 14:35:00+00:00"),
                pd.Timestamp("2024-03-01 14:30:00+00:00"),
                pd.Timestamp("2024-03-01 14:30:00+00:00"),
                pd.Timestamp("2024-03-01 14:35:00+00:00"),
            ],
            "symbol": ["BZ", "CL", "BZ", "CL"],
            "px_last": [81.4, 79.1, 81.1, 79.4],
            "trade_volume": [200, 180, 205, 175],
            "session": ["rth", "rth", "rth", "rth"],
        }
    )


def _target_long_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decision_time": [
                pd.Timestamp("2024-03-04 14:30:00+00:00"),
                pd.Timestamp("2024-03-04 14:30:00+00:00"),
                pd.Timestamp("2024-03-11 14:30:00+00:00"),
                pd.Timestamp("2024-03-11 14:30:00+00:00"),
            ],
            "symbol": ["CL", "BZ", "CL", "BZ"],
            "target_weight": [0.5, 0.5, 0.4, 0.6],
        }
    )


class TestExecutionBarFrameFromFrame:
    def test_from_frame_normalizes_columns_and_preserves_metadata(self) -> None:
        bars = ExecutionBarFrame.from_frame(
            _bars_frame(),
            timestamp_col="obs_time",
            asset_col="symbol",
            close_col="px_last",
            volume_col="trade_volume",
        )

        assert list(bars.data.columns) == ["ts", "asset", "close", "volume", "session"]
        assert bars.data["asset"].tolist() == ["BZ", "CL", "BZ", "CL"]
        assert bars.timestamps[0] == pd.Timestamp("2024-03-01 14:30:00+00:00")
        assert bars.asset_ids == ["BZ", "CL"]
        assert bars.metadata_columns == ["session"]

    def test_to_matrix_pivots_requested_field(self) -> None:
        bars = ExecutionBarFrame.from_frame(
            _bars_frame(),
            timestamp_col="obs_time",
            asset_col="symbol",
            close_col="px_last",
            volume_col="trade_volume",
        )

        close_matrix = bars.to_matrix("close")
        assert list(close_matrix.columns) == ["BZ", "CL"]
        assert close_matrix.loc[pd.Timestamp("2024-03-01 14:35:00+00:00"), "CL"] == 79.4


class TestExecutionBarFrameValidation:
    def test_rejects_naive_timestamps(self) -> None:
        frame = _bars_frame()
        frame["obs_time"] = frame["obs_time"].dt.tz_localize(None)

        with pytest.raises(ValueError, match="timezone-aware"):
            ExecutionBarFrame.from_frame(
                frame,
                timestamp_col="obs_time",
                asset_col="symbol",
                close_col="px_last",
            )

    def test_rejects_duplicate_execution_keys(self) -> None:
        frame = pd.DataFrame(
            {
                "ts": [
                    pd.Timestamp("2024-03-01 14:30:00+00:00"),
                    pd.Timestamp("2024-03-01 14:30:00+00:00"),
                ],
                "asset": ["CL", "CL"],
                "close": [79.1, 79.2],
            }
        )

        with pytest.raises(ValueError, match="unique on \\('ts', 'asset'\\)"):
            ExecutionBarFrame(frame)

    def test_slice_filters_time_and_asset(self) -> None:
        bars = ExecutionBarFrame.from_frame(
            _bars_frame(),
            timestamp_col="obs_time",
            asset_col="symbol",
            close_col="px_last",
            volume_col="trade_volume",
        )

        sliced = bars.slice(
            start=pd.Timestamp("2024-03-01 14:35:00+00:00"),
            assets=["CL"],
        )
        assert sliced.data["asset"].tolist() == ["CL"]
        assert sliced.timestamps.tolist() == [pd.Timestamp("2024-03-01 14:35:00+00:00")]


class TestTargetScheduleFromFrame:
    def test_from_frame_pivots_to_wide_schedule(self) -> None:
        schedule = TargetSchedule.from_frame(
            _target_long_frame(),
            timestamp_col="decision_time",
            asset_col="symbol",
            target_col="target_weight",
        )

        assert schedule.asset_ids == ["BZ", "CL"]
        assert schedule.decision_times.tolist() == [
            pd.Timestamp("2024-03-04 14:30:00+00:00"),
            pd.Timestamp("2024-03-11 14:30:00+00:00"),
        ]
        assert schedule.data.loc[pd.Timestamp("2024-03-11 14:30:00+00:00"), "BZ"] == 0.6

    def test_rejects_partial_decision_snapshot(self) -> None:
        frame = _target_long_frame().iloc[:-1]

        with pytest.raises(ValueError, match="complete target snapshot"):
            TargetSchedule.from_frame(
                frame,
                timestamp_col="decision_time",
                asset_col="symbol",
                target_col="target_weight",
            )


class TestTargetScheduleValidation:
    def test_rejects_naive_decision_index(self) -> None:
        schedule = pd.DataFrame(
            {"CL": [0.5], "BZ": [0.5]},
            index=pd.DatetimeIndex([pd.Timestamp("2024-03-04 14:30:00")]),
        )

        with pytest.raises(ValueError, match="timezone-aware"):
            TargetSchedule(schedule)

    def test_validate_for_rejects_missing_assets(self) -> None:
        bars = ExecutionBarFrame.from_frame(
            _bars_frame(),
            timestamp_col="obs_time",
            asset_col="symbol",
            close_col="px_last",
        )
        schedule = TargetSchedule(
            pd.DataFrame(
                {"CL": [0.5], "NG": [0.5]},
                index=pd.DatetimeIndex(
                    [pd.Timestamp("2024-03-04 14:30:00+00:00")],
                ),
            )
        )

        with pytest.raises(ValueError, match="assets not present"):
            schedule.validate_for(bars)

    def test_align_to_assets_expands_universe_with_explicit_fill(self) -> None:
        schedule = TargetSchedule.from_frame(
            _target_long_frame(),
            timestamp_col="decision_time",
            asset_col="symbol",
            target_col="target_weight",
            target_kind="weights",
        )

        aligned = schedule.align_to_assets(["CL", "BZ", "HO"], fill_value=0.0)
        assert aligned.asset_ids == ["CL", "BZ", "HO"]
        assert (aligned.data["HO"] == 0.0).all()
        assert aligned.target_kind == "weights"

    def test_rejects_unknown_target_kind(self) -> None:
        with pytest.raises(ValueError, match="target_kind"):
            TargetSchedule(
                pd.DataFrame(
                    {"CL": [0.5]},
                    index=pd.DatetimeIndex(
                        [pd.Timestamp("2024-03-04 14:30:00+00:00")],
                    ),
                ),
                target_kind="not-a-kind",  # type: ignore[arg-type]
            )
