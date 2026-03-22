"""Result dataclasses for backtesting."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


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
