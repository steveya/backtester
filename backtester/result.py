"""Result dataclasses for backtesting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .cv import CVSplit
from .execution import ExecutionBarFrame, TargetSchedule


@dataclass(frozen=True)
class EvalFoldResult:
    """Result from evaluating a pipeline on one CV fold."""

    portfolio_returns: pd.Series
    weights_history: pd.DataFrame
    signal_scores: dict[str, pd.Series] | None = None


@dataclass
class BacktestResult:
    """Complete backtest output."""

    weights_history: pd.DataFrame
    returns_history: pd.Series
    signal_history: dict[str, pd.DataFrame] | None

    metrics: dict[str, float]
    per_fold_metrics: pd.DataFrame

    optimized_params: dict[str, dict[str, float]]
    params_per_fold: list[dict[str, dict[str, float]]]

    cv_scheme: str
    optimizer: str
    n_folds: int
    n_eval_dates: int
    date_range: tuple[pd.Timestamp, pd.Timestamp]


@dataclass
class VariantBacktestResult:
    per_variant: dict[str, BacktestResult]
    rankings: pd.DataFrame
    best_variant: str
    statistical_tests: pd.DataFrame | None = None


@dataclass
class ExecutionBacktestResult:
    """Execution-clock simulation output for event-based rebalancing."""

    aligned_targets: pd.DataFrame
    holdings_history: pd.DataFrame
    trade_history: pd.DataFrame
    turnover_history: pd.Series
    portfolio_returns: pd.Series
    gross_exposure: pd.Series
    net_exposure: pd.Series
    cash_history: pd.Series | None
    event_log: pd.DataFrame
    trade_log: pd.DataFrame
    fill_convention: str
    target_kind: str


@dataclass(frozen=True)
class ExecutionFoldContext:
    """Fold-scoped decision and execution data passed to target generators."""

    split: CVSplit
    decision_index: pd.DatetimeIndex
    train_decision_times: pd.DatetimeIndex
    eval_decision_times: pd.DatetimeIndex
    purge_decision_times: pd.DatetimeIndex
    train_bars: ExecutionBarFrame
    eval_bars: ExecutionBarFrame


@dataclass(frozen=True)
class WalkForwardExecutionFoldResult:
    """Per-fold execution backtest output."""

    split: CVSplit
    targets: TargetSchedule
    execution_result: ExecutionBacktestResult
    metrics: dict[str, float]
    n_eval_decisions: int
    n_eval_bars: int


@dataclass
class WalkForwardExecutionResult:
    """Aggregated walk-forward execution backtest output."""

    per_fold: dict[str, WalkForwardExecutionFoldResult]
    aligned_targets: pd.DataFrame
    holdings_history: pd.DataFrame
    trade_history: pd.DataFrame
    turnover_history: pd.Series
    portfolio_returns: pd.Series
    gross_exposure: pd.Series
    net_exposure: pd.Series
    cash_history: pd.Series | None
    event_log: pd.DataFrame
    trade_log: pd.DataFrame
    metrics: dict[str, float]
    per_fold_metrics: pd.DataFrame
    cv_scheme: str
    fill_convention: str
    target_kind: str | None
    n_folds: int
    n_eval_decisions: int
    n_eval_bars: int
    decision_range: tuple[pd.Timestamp, pd.Timestamp]
    execution_range: tuple[pd.Timestamp, pd.Timestamp]
