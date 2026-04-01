"""Production backtesting framework for systematic strategies."""

from .cv import (
    CombinatorialPurgedCV,
    CVScheme,
    CVSplit,
    ExpandingCV,
    PurgedKFoldCV,
    WalkForwardCV,
)
from .execution import ExecutionBarFrame, TargetKind, TargetSchedule
from .execution_engine import EventRebalanceEngine
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
from .result import (
    BacktestResult,
    EvalFoldResult,
    ExecutionBacktestResult,
    ExecutionFoldContext,
    VariantBacktestResult,
    WalkForwardExecutionFoldResult,
    WalkForwardExecutionResult,
)
from .runner import BacktestRunner
from .walkforward import ExecutionEngine, ExecutionTargetGenerator, WalkForwardExecutionRunner

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
    # Execution data
    "ExecutionBarFrame",
    "TargetSchedule",
    "TargetKind",
    "EventRebalanceEngine",
    "ExecutionEngine",
    "ExecutionTargetGenerator",
    "ExecutionFoldContext",
    "WalkForwardExecutionRunner",
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
    "ExecutionBacktestResult",
    "WalkForwardExecutionFoldResult",
    "WalkForwardExecutionResult",
]
