"""Production backtesting framework for systematic strategies."""

from .accounting import (
    held_weights_from_rebalances,
    linear_turnover_costs,
    per_asset_pnl,
    scenario_returns,
)
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
    NoOpOptimizer,
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
    # Accounting
    "held_weights_from_rebalances",
    "per_asset_pnl",
    "linear_turnover_costs",
    "scenario_returns",
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
    "NoOpOptimizer",
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
