"""Production backtesting framework for systematic strategies."""

from .cv import (
    CombinatorialPurgedCV,
    CVScheme,
    CVSplit,
    ExpandingCV,
    PurgedKFoldCV,
    WalkForwardCV,
)
from .objectives import (
    CalmarObjective,
    CompositeObjective,
    MaxDrawdownObjective,
    Objective,
    SharpeObjective,
    SortinoObjective,
    TurnoverObjective,
)
from .optimizers import (
    BacktestOptimizer,
    BayesianOptimizer,
    GradientDescentOptimizer,
    GridSearchOptimizer,
    OptimizeResult,
    RandomSearchOptimizer,
)
from .result import BacktestResult, EvalFoldResult, VariantBacktestResult
from .runner import BacktestRunner

__version__ = "0.1.0"

__all__ = [
    # CV
    "CVScheme",
    "CVSplit",
    "WalkForwardCV",
    "PurgedKFoldCV",
    "CombinatorialPurgedCV",
    "ExpandingCV",
    # Objectives
    "Objective",
    "SharpeObjective",
    "SortinoObjective",
    "MaxDrawdownObjective",
    "TurnoverObjective",
    "CalmarObjective",
    "CompositeObjective",
    # Optimizers
    "BacktestOptimizer",
    "RandomSearchOptimizer",
    "GridSearchOptimizer",
    "BayesianOptimizer",
    "GradientDescentOptimizer",
    "OptimizeResult",
    # Runner
    "BacktestRunner",
    # Results
    "BacktestResult",
    "VariantBacktestResult",
    "EvalFoldResult",
]
