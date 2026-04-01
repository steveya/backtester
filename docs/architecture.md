# Architecture

## Overview

The current package follows a compose-and-run pattern:

1. A `BacktestRunner` is configured with a cross-validation scheme, one or more
   objectives, and an optimizer.
2. The caller provides a pipeline and an `evaluate_fn` callback.
3. The runner iterates over folds, optimizes parameters, evaluates the fold,
   and aggregates the results.

This architecture keeps the core library small while allowing the calling
application to own strategy logic.

## Core building blocks

### Cross-validation

`backtester.cv` defines the split protocol and built-in schemes:

- `WalkForwardCV`
- `ExpandingCV`
- `PurgedKFoldCV`
- `CombinatorialPurgedCV`

Each scheme implements the `CVScheme` protocol and yields `CVSplit` values.

### Objectives

`backtester.objectives` defines portfolio-scoring functions used during
optimization and reporting. The current package ships:

- `SharpeObjective`
- `SortinoObjective`
- `MaxDrawdownObjective`
- `TurnoverObjective`
- `CalmarObjective`
- `CompositeObjective`

The interface is protocol-based rather than inheritance-based.

### Optimizers

`backtester.optimizers` provides adapter-style search strategies:

- `RandomSearchOptimizer`
- `GridSearchOptimizer`
- `BayesianOptimizer`
- `GradientDescentOptimizer`

Each optimizer implements the `BacktestOptimizer` protocol and returns an
`OptimizeResult`.

### Runner and results

`backtester.runner` contains `BacktestRunner`, which orchestrates fold
execution. `backtester.result` contains the result containers exposed to callers:

- `EvalFoldResult`
- `BacktestResult`
- `VariantBacktestResult`

### Execution-clock foundations

`backtester.execution` defines provider-agnostic simulation inputs for the next
execution-oriented slice of the package:

- `ExecutionBarFrame` for canonical long-form execution bars on the execution clock
- `TargetSchedule` for sparse decision timestamps and target snapshots
- `TargetKind` to distinguish weight-based and exposure-based targets

These abstractions are intentionally separated from application-owned data
loading and signal generation. They normalize timestamps to UTC, enforce asset
coverage rules, and define the data contracts that later execution engines will
consume.

### Event-based execution engine

`backtester.execution_engine` now provides `EventRebalanceEngine`, the first
execution-clock simulator in the package.

The current fill convention is explicit:

- a decision timestamp aligns to the first execution bar at or after the decision time
- the rebalance is filled at the aligned bar close
- the new holdings become effective for the next bar return

The engine produces structured accounting outputs for later extensions:

- aligned target snapshots
- holdings and trades on the execution clock
- turnover and portfolio return paths
- event and trade logs for later cost or execution-algorithm work

### Walk-forward execution orchestration

`backtester.walkforward` provides `WalkForwardExecutionRunner`, which extends
the execution-clock slice from single-simulation primitives to fold-based
orchestration.

The design keeps the decision clock explicit and separate from the execution
clock:

- cross-validation schemes still operate on a `DatetimeIndex`
- that index is interpreted as decision timestamps rather than raw bar timestamps
- execution bars are selected by mapping each decision timestamp to its holding
  interval until the next decision boundary
- the application target generator receives an `ExecutionFoldContext` with
  fold-scoped decision timestamps and execution-bar slices
- the runner calls an execution engine for each evaluation fold and aggregates
  the results into a `WalkForwardExecutionResult`

This separation matters for causal semantics. In data providers such as
Alphaforge, a 5-minute execution timestamp may represent the first time a bar
is available for trading rather than the bar start. The runner therefore
models fold boundaries in terms of decision availability and holding intervals
instead of assuming the execution clock and decision clock are the same object.

### Post-run analysis

The package also includes standalone analysis utilities:

- `backtester.analytics` for performance tables, rolling metrics, drawdown
  summaries, and parameter analysis
- `backtester.attribution` for signal, instrument, and sector decomposition
- `backtester.statistical` for bootstrap tests, deflated Sharpe ratio, and
  clustering

## Design principles

- Protocols over abstract base classes
- Frozen dataclasses for lightweight data containers where practical
- No mutable global state
- Clear separation between orchestration and strategy-specific logic

## Near-term roadmap note

The repository now ships the first execution-clock vertical slice:

- provider-agnostic execution inputs
- a simple event-based execution engine
- walk-forward execution orchestration on a separate decision clock

The next layers are cost modeling, execution algorithms, instrument-specific
sizing, and application-level integrations that sit above the provider-agnostic
core.
