"""Execution-clock event-based rebalance engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .execution import ExecutionBarFrame, TargetSchedule
from .result import ExecutionBacktestResult

ALIGNED_CLOSE_NEXT_BAR = "aligned_close_next_bar"


@dataclass
class EventRebalanceEngine:
    """Simple event-based rebalance simulator on the execution clock.

    Fill convention:

    - a decision timestamp is aligned to the first execution bar at or after the
      decision time
    - the rebalance is filled at that aligned bar's close
    - post-trade holdings become effective for the next bar's realized return
    """

    fill_convention: str = ALIGNED_CLOSE_NEXT_BAR

    def run(
        self,
        bars: ExecutionBarFrame,
        targets: TargetSchedule,
    ) -> ExecutionBacktestResult:
        """Simulate sparse target rebalances on a dense execution clock."""

        targets.validate_for(bars)
        asset_ids = bars.asset_ids
        schedule = targets.align_to_assets(asset_ids, fill_value=0.0)
        close_matrix = bars.to_matrix("close").reindex(columns=asset_ids)
        self._validate_close_matrix(close_matrix)

        aligned_targets, event_log = self._align_targets(
            execution_times=close_matrix.index,
            schedule=schedule,
        )

        holdings_history = pd.DataFrame(0.0, index=close_matrix.index, columns=asset_ids)
        trade_history = pd.DataFrame(0.0, index=close_matrix.index, columns=asset_ids)
        turnover_history = pd.Series(0.0, index=close_matrix.index, name="turnover")

        previous_holdings = pd.Series(0.0, index=asset_ids)
        returns_matrix = close_matrix.pct_change().fillna(0.0)
        portfolio_returns = pd.Series(0.0, index=close_matrix.index, name="portfolio_return")

        for i, ts in enumerate(close_matrix.index):
            if ts in aligned_targets.index:
                target = aligned_targets.loc[ts]
                trade = target - previous_holdings
                current_holdings = target.astype(float)
            else:
                trade = pd.Series(0.0, index=asset_ids)
                current_holdings = previous_holdings

            holdings_history.loc[ts] = current_holdings
            trade_history.loc[ts] = trade
            turnover_history.loc[ts] = float(trade.abs().sum())

            if i > 0:
                portfolio_returns.loc[ts] = float(
                    previous_holdings.dot(returns_matrix.loc[ts].fillna(0.0))
                )

            previous_holdings = current_holdings

        gross_exposure = holdings_history.abs().sum(axis=1).rename("gross_exposure")
        net_exposure = holdings_history.sum(axis=1).rename("net_exposure")
        cash_history = None
        if schedule.target_kind == "weights":
            cash_history = (1.0 - net_exposure).rename("cash_weight")

        event_log = self._attach_event_turnover(
            event_log=event_log,
            turnover_history=turnover_history,
            aligned_targets=aligned_targets,
        )

        trade_log = self._build_trade_log(
            aligned_targets=aligned_targets,
            trade_history=trade_history,
            event_log=event_log,
        )

        return ExecutionBacktestResult(
            aligned_targets=aligned_targets,
            holdings_history=holdings_history,
            trade_history=trade_history,
            turnover_history=turnover_history,
            portfolio_returns=portfolio_returns,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            cash_history=cash_history,
            event_log=event_log,
            trade_log=trade_log,
            fill_convention=self.fill_convention,
            target_kind=schedule.target_kind,
        )

    @staticmethod
    def _validate_close_matrix(close_matrix: pd.DataFrame) -> None:
        if close_matrix.isna().any().any():
            missing = close_matrix.columns[close_matrix.isna().any()].tolist()
            raise ValueError(
                "EventRebalanceEngine requires a dense close-price matrix; "
                f"missing prices were found for assets {missing}."
            )

    @staticmethod
    def _align_targets(
        *,
        execution_times: pd.DatetimeIndex,
        schedule: TargetSchedule,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if schedule.data.empty:
            empty_targets = pd.DataFrame(columns=schedule.asset_ids, dtype="float64")
            empty_targets.index = pd.DatetimeIndex([], tz="UTC", name="execution_ts")
            empty_event_log = pd.DataFrame(
                columns=[
                    "decision_ts",
                    "execution_ts",
                    "status",
                    "gross_target",
                    "net_target",
                ]
            )
            return empty_targets, empty_event_log

        aligned_records: list[dict[str, object]] = []
        resolved_rows: dict[pd.Timestamp, tuple[pd.Timestamp, pd.Series]] = {}

        for decision_ts, target in schedule.data.iterrows():
            idx = execution_times.searchsorted(decision_ts, side="left")
            if idx >= len(execution_times):
                aligned_records.append(
                    {
                        "decision_ts": decision_ts,
                        "execution_ts": pd.NaT,
                        "status": "dropped_no_execution_bar",
                        "gross_target": float(target.abs().sum()),
                        "net_target": float(target.sum()),
                    }
                )
                continue

            execution_ts = execution_times[idx]
            aligned_records.append(
                {
                    "decision_ts": decision_ts,
                    "execution_ts": execution_ts,
                    "status": "aligned",
                    "gross_target": float(target.abs().sum()),
                    "net_target": float(target.sum()),
                }
            )
            resolved_rows[execution_ts] = (decision_ts, target.astype(float))

        event_log = pd.DataFrame(aligned_records).sort_values("decision_ts").reset_index(drop=True)
        if not event_log.empty:
            duplicated_execution_ts = event_log["execution_ts"].notna() & event_log.duplicated(
                subset=["execution_ts"], keep="last"
            )
            event_log.loc[duplicated_execution_ts, "status"] = "superseded_same_bar"

        if not resolved_rows:
            empty_targets = pd.DataFrame(columns=schedule.asset_ids, dtype="float64")
            empty_targets.index = pd.DatetimeIndex([], tz="UTC", name="execution_ts")
            return empty_targets, event_log

        aligned_targets = pd.DataFrame(
            {execution_ts: target for execution_ts, (_, target) in resolved_rows.items()}
        ).T.sort_index()
        aligned_targets.index = pd.DatetimeIndex(
            aligned_targets.index, tz="UTC", name="execution_ts"
        )
        aligned_targets.columns = schedule.data.columns
        return aligned_targets, event_log

    @staticmethod
    def _attach_event_turnover(
        *,
        event_log: pd.DataFrame,
        turnover_history: pd.Series,
        aligned_targets: pd.DataFrame,
    ) -> pd.DataFrame:
        if event_log.empty:
            return event_log

        result = event_log.copy()
        result["turnover"] = 0.0
        aligned_execution_ts = set(aligned_targets.index)
        for idx, row in result.iterrows():
            execution_ts = row["execution_ts"]
            if pd.notna(execution_ts) and execution_ts in aligned_execution_ts:
                result.at[idx, "turnover"] = float(turnover_history.loc[execution_ts])
        return result

    @staticmethod
    def _build_trade_log(
        *,
        aligned_targets: pd.DataFrame,
        trade_history: pd.DataFrame,
        event_log: pd.DataFrame,
    ) -> pd.DataFrame:
        if aligned_targets.empty:
            return pd.DataFrame(
                columns=[
                    "decision_ts",
                    "execution_ts",
                    "asset",
                    "previous_position",
                    "target_position",
                    "trade",
                    "abs_trade",
                ]
            )

        aligned_events = event_log[event_log["status"] == "aligned"].copy()
        if aligned_events.empty:
            return pd.DataFrame(
                columns=[
                    "decision_ts",
                    "execution_ts",
                    "asset",
                    "previous_position",
                    "target_position",
                    "trade",
                    "abs_trade",
                ]
            )

        rows: list[dict[str, object]] = []
        previous_holdings = pd.Series(0.0, index=trade_history.columns)
        aligned_events = aligned_events.sort_values("decision_ts")

        for _, event in aligned_events.iterrows():
            execution_ts = event["execution_ts"]
            decision_ts = event["decision_ts"]
            current_target = aligned_targets.loc[execution_ts]
            current_trade = trade_history.loc[execution_ts]
            for asset in trade_history.columns:
                trade = float(current_trade[asset])
                if np.isclose(trade, 0.0):
                    continue
                rows.append(
                    {
                        "decision_ts": decision_ts,
                        "execution_ts": execution_ts,
                        "asset": asset,
                        "previous_position": float(previous_holdings[asset]),
                        "target_position": float(current_target[asset]),
                        "trade": trade,
                        "abs_trade": abs(trade),
                    }
                )
            previous_holdings = current_target

        return pd.DataFrame(rows)
