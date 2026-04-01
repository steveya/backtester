"""Walk-forward orchestration on a dense execution clock."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from .cv import CVScheme, CVSplit
from .execution import ExecutionBarFrame, TargetSchedule
from .objectives import Objective
from .result import (
    ExecutionBacktestResult,
    ExecutionFoldContext,
    WalkForwardExecutionFoldResult,
    WalkForwardExecutionResult,
)


def _normalize_decision_index(decision_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if isinstance(decision_index, pd.DatetimeIndex):
        ts = decision_index
    else:
        ts = pd.DatetimeIndex(pd.to_datetime(decision_index, errors="raise"))
    if ts.tz is None:
        if len(ts) == 0:
            return ts.tz_localize("UTC")
        raise ValueError("Decision timestamps must be timezone-aware and explicitly in UTC.")
    normalized = ts.tz_convert("UTC").sort_values()
    if normalized.has_duplicates:
        raise ValueError("Decision timestamps must be unique.")
    return normalized


@runtime_checkable
class ExecutionEngine(Protocol):
    """Execution engine protocol consumed by the walk-forward runner."""

    fill_convention: str

    def run(
        self,
        bars: ExecutionBarFrame,
        targets: TargetSchedule,
    ) -> ExecutionBacktestResult: ...


@runtime_checkable
class ExecutionTargetGenerator(Protocol):
    """Fold-aware target generator used by the walk-forward runner."""

    def __call__(self, context: ExecutionFoldContext) -> TargetSchedule: ...


@dataclass
class WalkForwardExecutionRunner:
    """Run fold-based execution backtests on a separate decision clock."""

    cv: CVScheme
    objectives: list[Objective]
    engine: ExecutionEngine

    def run(
        self,
        bars: ExecutionBarFrame,
        decision_index: pd.DatetimeIndex,
        target_generator: ExecutionTargetGenerator,
    ) -> WalkForwardExecutionResult:
        decisions = _normalize_decision_index(decision_index)
        splits = list(self.cv.splits(decisions))
        if not splits:
            return self._empty_result(bars=bars, decisions=decisions)

        per_fold: dict[str, WalkForwardExecutionFoldResult] = {}
        per_fold_rows: list[dict[str, float | int | str]] = []
        target_kinds: set[str] = set()

        for split in splits:
            context = self._build_fold_context(bars=bars, decisions=decisions, split=split)
            targets = target_generator(context)
            if not isinstance(targets, TargetSchedule):
                raise TypeError("Execution target generators must return a TargetSchedule.")
            self._validate_targets_for_split(targets=targets, split=split)

            execution_result = self._run_fold(eval_bars=context.eval_bars, targets=targets)
            metrics = self._compute_metrics(execution_result)

            per_fold[split.fold_id] = WalkForwardExecutionFoldResult(
                split=split,
                targets=targets,
                execution_result=execution_result,
                metrics=metrics,
                n_eval_decisions=len(split.eval_dates),
                n_eval_bars=len(execution_result.portfolio_returns),
            )
            row: dict[str, float | int | str] = {
                "fold_id": split.fold_id,
                "n_eval_decisions": len(split.eval_dates),
                "n_eval_bars": len(execution_result.portfolio_returns),
            }
            row.update(metrics)
            per_fold_rows.append(row)
            target_kinds.add(execution_result.target_kind)

        if len(target_kinds) > 1:
            raise ValueError("Walk-forward execution folds must agree on target_kind.")

        aligned_targets = self._concat_frames(
            per_fold=per_fold,
            attr="aligned_targets",
            columns=bars.asset_ids,
        )
        holdings_history = self._concat_frames(
            per_fold=per_fold,
            attr="holdings_history",
            columns=bars.asset_ids,
        )
        trade_history = self._concat_frames(
            per_fold=per_fold,
            attr="trade_history",
            columns=bars.asset_ids,
        )
        turnover_history = self._concat_series(per_fold=per_fold, attr="turnover_history")
        portfolio_returns = self._concat_series(per_fold=per_fold, attr="portfolio_returns")
        gross_exposure = self._concat_series(per_fold=per_fold, attr="gross_exposure")
        net_exposure = self._concat_series(per_fold=per_fold, attr="net_exposure")
        cash_history = self._concat_optional_series(per_fold=per_fold, attr="cash_history")
        event_log = self._concat_logs(per_fold=per_fold, attr="event_log")
        trade_log = self._concat_logs(per_fold=per_fold, attr="trade_log")

        metrics = self._compute_metrics(
            ExecutionBacktestResult(
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
                fill_convention=self.engine.fill_convention,
                target_kind=next(iter(target_kinds)) if target_kinds else "weights",
            )
        )

        bar_times = bars.timestamps
        return WalkForwardExecutionResult(
            per_fold=per_fold,
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
            metrics=metrics,
            per_fold_metrics=pd.DataFrame(per_fold_rows),
            cv_scheme=self.cv.name,
            fill_convention=self.engine.fill_convention,
            target_kind=next(iter(target_kinds)) if target_kinds else None,
            n_folds=len(per_fold),
            n_eval_decisions=sum(result.n_eval_decisions for result in per_fold.values()),
            n_eval_bars=sum(result.n_eval_bars for result in per_fold.values()),
            decision_range=(decisions[0], decisions[-1])
            if len(decisions) > 0
            else (pd.NaT, pd.NaT),
            execution_range=(bar_times[0], bar_times[-1])
            if len(bar_times) > 0
            else (pd.NaT, pd.NaT),
        )

    def _build_fold_context(
        self,
        *,
        bars: ExecutionBarFrame,
        decisions: pd.DatetimeIndex,
        split: CVSplit,
    ) -> ExecutionFoldContext:
        return ExecutionFoldContext(
            split=split,
            decision_index=decisions,
            train_decision_times=pd.DatetimeIndex(split.train_dates),
            eval_decision_times=pd.DatetimeIndex(split.eval_dates),
            purge_decision_times=pd.DatetimeIndex(split.purge_dates),
            train_bars=_select_bars_for_decisions(
                bars=bars,
                decision_index=decisions,
                selected_decisions=pd.DatetimeIndex(split.train_dates),
            ),
            eval_bars=_select_bars_for_decisions(
                bars=bars,
                decision_index=decisions,
                selected_decisions=pd.DatetimeIndex(split.eval_dates),
            ),
        )

    @staticmethod
    def _validate_targets_for_split(targets: TargetSchedule, split: CVSplit) -> None:
        unexpected = targets.decision_times.difference(split.eval_dates)
        if len(unexpected) > 0:
            raise ValueError(
                "Target schedules must only reference evaluation decision timestamps; "
                f"found {list(unexpected)} outside split {split.fold_id}."
            )

    def _run_fold(
        self,
        *,
        eval_bars: ExecutionBarFrame,
        targets: TargetSchedule,
    ) -> ExecutionBacktestResult:
        if not eval_bars.data.empty:
            return self.engine.run(eval_bars, targets)
        return self._empty_execution_result(targets=targets)

    def _compute_metrics(self, result: ExecutionBacktestResult) -> dict[str, float]:
        context: dict[str, Any] = {
            "weights_history": result.holdings_history,
            "holdings_history": result.holdings_history,
            "trade_history": result.trade_history,
            "turnover_history": result.turnover_history,
            "gross_exposure": result.gross_exposure,
            "net_exposure": result.net_exposure,
            "cash_history": result.cash_history,
            "aligned_targets": result.aligned_targets,
            "event_log": result.event_log,
            "trade_log": result.trade_log,
            "target_kind": result.target_kind,
        }
        return {
            objective.name: objective.compute(result.portfolio_returns, **context)
            for objective in self.objectives
        }

    def _empty_execution_result(self, *, targets: TargetSchedule) -> ExecutionBacktestResult:
        asset_ids = targets.asset_ids
        empty_index = pd.DatetimeIndex([], tz="UTC", name="execution_ts")
        empty_frame = pd.DataFrame(index=empty_index, columns=asset_ids, dtype="float64")
        event_log = pd.DataFrame(
            [
                {
                    "decision_ts": decision_ts,
                    "execution_ts": pd.NaT,
                    "status": "dropped_no_execution_bar",
                    "gross_target": float(target.abs().sum()),
                    "net_target": float(target.sum()),
                    "turnover": 0.0,
                }
                for decision_ts, target in targets.data.iterrows()
            ]
        )
        trade_log = pd.DataFrame(
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
        cash_history = None
        if targets.target_kind == "weights":
            cash_history = pd.Series(dtype="float64", index=empty_index, name="cash_weight")

        return ExecutionBacktestResult(
            aligned_targets=empty_frame.copy(),
            holdings_history=empty_frame.copy(),
            trade_history=empty_frame.copy(),
            turnover_history=pd.Series(dtype="float64", index=empty_index, name="turnover"),
            portfolio_returns=pd.Series(
                dtype="float64",
                index=empty_index,
                name="portfolio_return",
            ),
            gross_exposure=pd.Series(dtype="float64", index=empty_index, name="gross_exposure"),
            net_exposure=pd.Series(dtype="float64", index=empty_index, name="net_exposure"),
            cash_history=cash_history,
            event_log=event_log,
            trade_log=trade_log,
            fill_convention=self.engine.fill_convention,
            target_kind=targets.target_kind,
        )

    @staticmethod
    def _concat_frames(
        *,
        per_fold: dict[str, WalkForwardExecutionFoldResult],
        attr: str,
        columns: list[str],
    ) -> pd.DataFrame:
        frames = {
            fold_id: getattr(result.execution_result, attr).reindex(columns=columns, fill_value=0.0)
            for fold_id, result in per_fold.items()
        }
        if not frames:
            return pd.DataFrame(columns=columns, dtype="float64")
        return pd.concat(frames, names=["fold_id"])

    @staticmethod
    def _concat_series(
        *,
        per_fold: dict[str, WalkForwardExecutionFoldResult],
        attr: str,
    ) -> pd.Series:
        series = {
            fold_id: getattr(result.execution_result, attr) for fold_id, result in per_fold.items()
        }
        if not series:
            return pd.Series(dtype="float64")
        return pd.concat(series, names=["fold_id"])

    @staticmethod
    def _concat_optional_series(
        *,
        per_fold: dict[str, WalkForwardExecutionFoldResult],
        attr: str,
    ) -> pd.Series | None:
        values = {
            fold_id: getattr(result.execution_result, attr)
            for fold_id, result in per_fold.items()
            if getattr(result.execution_result, attr) is not None
        }
        if not values:
            return None
        return pd.concat(values, names=["fold_id"])

    @staticmethod
    def _concat_logs(
        *,
        per_fold: dict[str, WalkForwardExecutionFoldResult],
        attr: str,
    ) -> pd.DataFrame:
        logs: list[pd.DataFrame] = []
        template_columns: list[str] | None = None
        for fold_id, result in per_fold.items():
            log = getattr(result.execution_result, attr).copy()
            template_columns = ["fold_id", *log.columns.tolist()]
            if log.empty:
                continue
            log.insert(0, "fold_id", fold_id)
            logs.append(log)
        if not logs:
            return pd.DataFrame(columns=template_columns)
        return pd.concat(logs, ignore_index=True)

    def _empty_result(
        self,
        *,
        bars: ExecutionBarFrame,
        decisions: pd.DatetimeIndex,
    ) -> WalkForwardExecutionResult:
        empty_index = pd.MultiIndex.from_arrays(
            [[], []],
            names=["fold_id", "execution_ts"],
        )
        empty_frame = pd.DataFrame(index=empty_index, columns=bars.asset_ids, dtype="float64")
        empty_series = pd.Series(dtype="float64", index=empty_index)
        cash_history = pd.Series(dtype="float64", index=empty_index, name="cash_weight")
        bar_times = bars.timestamps

        return WalkForwardExecutionResult(
            per_fold={},
            aligned_targets=empty_frame.copy(),
            holdings_history=empty_frame.copy(),
            trade_history=empty_frame.copy(),
            turnover_history=empty_series.rename("turnover"),
            portfolio_returns=empty_series.rename("portfolio_return"),
            gross_exposure=empty_series.rename("gross_exposure"),
            net_exposure=empty_series.rename("net_exposure"),
            cash_history=cash_history,
            event_log=pd.DataFrame(),
            trade_log=pd.DataFrame(),
            metrics={objective.name: float("nan") for objective in self.objectives},
            per_fold_metrics=pd.DataFrame(),
            cv_scheme=self.cv.name,
            fill_convention=self.engine.fill_convention,
            target_kind=None,
            n_folds=0,
            n_eval_decisions=0,
            n_eval_bars=0,
            decision_range=(decisions[0], decisions[-1])
            if len(decisions) > 0
            else (pd.NaT, pd.NaT),
            execution_range=(bar_times[0], bar_times[-1])
            if len(bar_times) > 0
            else (pd.NaT, pd.NaT),
        )


def _select_bars_for_decisions(
    *,
    bars: ExecutionBarFrame,
    decision_index: pd.DatetimeIndex,
    selected_decisions: pd.DatetimeIndex,
) -> ExecutionBarFrame:
    if len(selected_decisions) == 0 or bars.data.empty:
        return ExecutionBarFrame(bars.data.iloc[0:0].copy())

    positions = decision_index.get_indexer(selected_decisions)
    if (positions < 0).any():
        raise ValueError("Selected decision timestamps must be a subset of the decision index.")

    bar_times = bars.data["ts"]
    mask = pd.Series(False, index=bars.data.index)
    for position in positions:
        start = decision_index[position]
        if position + 1 < len(decision_index):
            end = decision_index[position + 1]
            mask |= (bar_times >= start) & (bar_times < end)
        else:
            mask |= bar_times >= start
    return ExecutionBarFrame(bars.data.loc[mask].reset_index(drop=True))
