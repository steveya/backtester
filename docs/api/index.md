# API Overview

This page summarizes the public package surface exposed from `backtester`.

## Top-level imports

The package currently exports the following groups from `backtester.__init__`.

### Cross-validation

```python
from backtester import (
    CVScheme,
    CVSplit,
    WalkForwardCV,
    PurgedKFoldCV,
    CombinatorialPurgedCV,
    ExpandingCV,
)
```

Use these to define how train/evaluation folds are generated from a date index.

### Objectives

```python
from backtester import (
    Objective,
    SharpeObjective,
    SortinoObjective,
    MaxDrawdownObjective,
    TurnoverObjective,
    CalmarObjective,
    CompositeObjective,
)
```

Use objectives to score portfolio return series during optimization and final
reporting.

### Optimizers

```python
from backtester import (
    BacktestOptimizer,
    RandomSearchOptimizer,
    GridSearchOptimizer,
    BayesianOptimizer,
    GradientDescentOptimizer,
    OptimizeResult,
)
```

Use optimizer adapters to search parameter sets for a caller-supplied
evaluation callback.

### Runner and result containers

```python
from backtester import (
    BacktestRunner,
    BacktestResult,
    VariantBacktestResult,
    EvalFoldResult,
)
```

`BacktestRunner` is the current orchestration entry point for full backtest and
variant comparison workflows.

### Execution-clock input abstractions

```python
from backtester import (
    ExecutionBarFrame,
    TargetSchedule,
    TargetKind,
)
```

Use these to define provider-agnostic execution-bar inputs and sparse target
schedules for lower-frequency decision logic executed on a higher-frequency
clock.

`ExecutionBarFrame` stores canonical long-form execution bars with required
``ts``, ``asset``, and ``close`` fields plus optional OHLCV or metadata fields.

`TargetSchedule` stores sparse decision timestamps as full target snapshots
across assets. Missing decision timestamps imply carry-forward behavior between
events rather than an implicit flatten.

`TargetKind` makes the target interpretation explicit:

- `"weights"`: target values are portfolio weights and residual weight is
  implicit cash
- `"exposures"`: target values are caller-defined exposures and no implicit cash
  inference should be assumed

### Execution engine and result container

```python
from backtester import (
    EventRebalanceEngine,
    ExecutionBacktestResult,
)
```

`EventRebalanceEngine` consumes `ExecutionBarFrame` and `TargetSchedule` and
produces an `ExecutionBacktestResult`.

The current fill convention is:

- align each decision timestamp to the first execution bar at or after the decision time
- fill the rebalance at that aligned bar close
- apply the new holdings from the next bar onward

`ExecutionBacktestResult` includes:

- `aligned_targets`
- `holdings_history`
- `trade_history`
- `turnover_history`
- `portfolio_returns`
- `gross_exposure`
- `net_exposure`
- `cash_history` for weight-based targets
- `event_log`
- `trade_log`

### Execution walk-forward runner

```python
from backtester import (
    ExecutionEngine,
    ExecutionFoldContext,
    ExecutionTargetGenerator,
    WalkForwardExecutionFoldResult,
    WalkForwardExecutionRunner,
    WalkForwardExecutionResult,
)
```

`WalkForwardExecutionRunner` orchestrates fold generation on a decision index
separate from the dense execution bars.

Its core contract is:

- input execution bars as `ExecutionBarFrame`
- input decision timestamps as an explicit `DatetimeIndex`
- reuse an existing `CVScheme` on that decision index
- call an application-owned `ExecutionTargetGenerator`
- execute each evaluation fold through an `ExecutionEngine`
- aggregate fold outputs into a `WalkForwardExecutionResult`

`ExecutionFoldContext` is the fold-scoped input passed to the target generator.
It includes:

- the underlying `CVSplit`
- `train_decision_times`
- `eval_decision_times`
- `purge_decision_times`
- `train_bars`
- `eval_bars`

The execution-bar slices are mapped from decision timestamps to holding
intervals on the execution clock. For each selected decision timestamp, the
runner includes execution bars from that decision time until the next decision
boundary in the global decision index.

`WalkForwardExecutionFoldResult` stores the split, returned targets, per-fold
execution result, per-fold metrics, and basic fold counts.

`WalkForwardExecutionResult` aggregates:

- `per_fold`
- `aligned_targets`
- `holdings_history`
- `trade_history`
- `turnover_history`
- `portfolio_returns`
- `gross_exposure`
- `net_exposure`
- `cash_history`
- `event_log`
- `trade_log`
- `metrics`
- `per_fold_metrics`

The aggregated history objects use a `fold_id` top-level index so overlapping
evaluation windows do not silently overwrite one another.

## Utility modules

The package also exposes analysis functions through submodules:

- `backtester.analytics`
- `backtester.attribution`
- `backtester.statistical`

These are imported as module-level utilities rather than re-exported from the
top-level package.

## Current API note

This overview reflects the functionality that currently ships in the repository
today. The execution-clock slice now includes provider-agnostic inputs, the
first event-based execution engine, and the first walk-forward execution
runner.
