# Backtester

A production-grade backtesting framework for systematic trading strategies, with built-in cross-validation, parameter optimization, performance analytics, and statistical testing.

## Features

- **Walk-forward & purged cross-validation** — avoid lookahead bias with `WalkForwardCV`, `PurgedKFoldCV`, `CombinatorialPurgedCV`, and `ExpandingCV`
- **Execution-clock foundations** — canonical execution-bar and target-schedule abstractions for provider-agnostic simulation inputs
- **Event-based execution engine** — align sparse target events to execution bars and simulate close-to-close rebalances
- **Execution-clock walk-forward runner** — split on a separate decision index and evaluate each fold on execution-bar intervals
- **Multiple optimization backends** — no-op, random search, grid search, Bayesian (optuna), and gradient-descent callback adapter
- **Composable objectives** — Sharpe, Sortino, Calmar, max drawdown, turnover, and weighted composites
- **Performance analytics** — performance tables, rolling metrics, drawdown analysis, parameter stability and sensitivity
- **PnL attribution** — decompose returns by signal, instrument, or sector
- **Statistical rigor** — paired bootstrap tests, deflated Sharpe ratio, strategy clustering

## Installation

```bash
pip install -e .

# With optional dependencies
pip install -e ".[bayesian]"   # Adds optuna for Bayesian optimization
pip install -e ".[docs]"      # Adds MkDocs for package documentation
pip install -e ".[stats]"     # Adds scipy for statistical tests
pip install -e ".[dev]"       # Adds pytest, ruff, and docs tooling for development
```

## Quick Start

```python
import pandas as pd
from backtester import (
    BacktestRunner,
    WalkForwardCV,
    SharpeObjective,
    NoOpOptimizer,
    EvalFoldResult,
)

# Define how to evaluate one fold
def evaluate_fn(pipeline, split):
    # Your strategy logic here
    returns = pd.Series(...)
    weights = pd.DataFrame(...)
    return EvalFoldResult(
        portfolio_returns=returns,
        weights_history=weights,
    )

# Configure and run
runner = BacktestRunner(
    cv=WalkForwardCV(eval_window=63, step=21, min_train=252),
    objectives=[SharpeObjective()],
    optimizer=NoOpOptimizer(),
)

result = runner.run(
    pipeline=my_pipeline,
    evaluate_fn=evaluate_fn,
    dates=pd.bdate_range("2020-01-02", "2024-12-31"),
    n_trials=50,
)

print(result.metrics)           # Aggregated performance
print(result.per_fold_metrics)  # Breakdown by fold
```

## Execution-Clock Walk-Forward

```python
import pandas as pd
from backtester import (
    EventRebalanceEngine,
    ExecutionBarFrame,
    SharpeObjective,
    TargetSchedule,
    WalkForwardCV,
    WalkForwardExecutionRunner,
)

bars = ExecutionBarFrame.from_frame(
    data=execution_df,
    timestamp_col="ts",
    asset_col="asset",
    close_col="close",
    volume_col="volume",
)

decision_index = pd.DatetimeIndex(decision_timestamps_utc)

def generate_targets(context):
    targets = pd.DataFrame(
        strategy_targets_for(context.eval_decision_times),
        index=context.eval_decision_times,
    )
    return TargetSchedule(targets, target_kind="weights")

runner = WalkForwardExecutionRunner(
    cv=WalkForwardCV(eval_window=4, step=4, min_train=12),
    objectives=[SharpeObjective()],
    engine=EventRebalanceEngine(),
)

result = runner.run(bars, decision_index, generate_targets)

print(result.metrics)
print(result.per_fold_metrics)
```

`WalkForwardExecutionRunner` keeps the decision clock separate from the
execution clock. Each selected decision timestamp is mapped to its holding
interval on the execution bars until the next decision boundary, so the final
decision in a fold still realizes post-trade PnL without leaking later
decision timestamps.

Use `NoOpOptimizer` when the strategy refits on each fold but does not need
hyperparameter search. Use `RandomSearchOptimizer`, `GridSearchOptimizer`, or
`BayesianOptimizer` when the fold evaluation genuinely needs parameter tuning.

`EvalFoldResult` can also carry `signal_scores` and arbitrary `artifacts`.
`BacktestRunner` preserves fold artifacts on the final `BacktestResult`, and
passes `weights_history`, `signal_scores`, `artifacts`, and `split` into
objective `compute()` calls for context-aware scoring.

## Fold-Refit Strategies

```python
from backtester import BacktestRunner, EvalFoldResult, NoOpOptimizer, SharpeObjective, WalkForwardCV

def evaluate_fn(pipeline, split):
    fitted_model = fit_model(train_dates=split.train_dates)
    predictions = predict_on_eval(fitted_model, split.eval_dates)
    weights = build_eval_weights(predictions)
    pnl = weights.mul(eval_returns.loc[split.eval_dates], axis=0).sum(axis=1)
    return EvalFoldResult(
        portfolio_returns=pnl,
        weights_history=weights,
        signal_scores={"alpha": predictions},
        artifacts={"model_coefficients": fitted_model.params},
    )

runner = BacktestRunner(
    cv=WalkForwardCV(eval_window=63, step=21, min_train=252),
    objectives=[SharpeObjective()],
    optimizer=NoOpOptimizer(),
)
```

This pattern is the intended adapter surface for application packages that
refit an estimator on each fold and want the shared runner to own only the
generic orchestration and accounting layers.

## Comparing Strategy Variants

```python
from backtester import VariantBacktestResult

variant_result = runner.run_variants(
    [variant_a, variant_b, variant_c],
    evaluate_fn=evaluate_fn,
    dates=dates,
    n_trials=50,
)

print(variant_result.rankings)      # Side-by-side comparison
print(variant_result.best_variant)  # Winner
```

## Analytics

```python
from backtester.analytics import (
    performance_table,
    rolling_metrics,
    drawdown_table,
    param_stability,
    param_sensitivity,
)

# Full performance summary
perf = performance_table(result.returns_history)

# Rolling statistics
rolling = rolling_metrics(result.returns_history, window=63)

# Top drawdowns
dd = drawdown_table(result.returns_history, top_n=5)

# Parameter consistency across folds
stability = param_stability(result.params_per_fold)
```

## Accounting

```python
from backtester import (
    held_weights_from_rebalances,
    linear_turnover_costs,
    per_asset_pnl,
    scenario_returns,
)

weights_history = held_weights_from_rebalances(
    rebalance_weights=weekly_weights,
    execution_dates=execution_dates,
    execution_index=daily_returns.index,
)

pnl_by_asset = per_asset_pnl(weights_history, daily_returns)
costs = linear_turnover_costs(
    weekly_weights,
    execution_dates,
    daily_returns.index,
    half_spread_bps=5.0,
)

scenarios, by_asset = scenario_returns(
    weights_history,
    daily_returns,
    rebalance_weights=weekly_weights,
    execution_dates=execution_dates,
    cost_multipliers=(1.0, 2.0, 3.0),
    base_half_spread_bps=5.0,
)
```

These helpers are provider-agnostic and strategy-agnostic. Application packages
own signal generation and weight construction; the shared backtester owns the
generic conversion from weights and returns into PnL and cost scenarios.

## Attribution

```python
from backtester.attribution import (
    signal_attribution,
    instrument_attribution,
    sector_attribution,
)

# Decompose PnL by source
signal_pnl = signal_attribution(weights, returns, signal_weights, signal_scores)
instrument_pnl = instrument_attribution(weights, returns)
sector_pnl = sector_attribution(weights, returns, sector_map)
```

## Statistical Testing

```python
from backtester.statistical import (
    paired_bootstrap_test,
    deflated_sharpe_ratio,
    strategy_clustering,
)

# Is strategy A significantly better than B?
test = paired_bootstrap_test(returns_a, returns_b, metric_fn=lambda r: r.mean() / r.std())

# Is this Sharpe robust to multiple testing?
dsr = deflated_sharpe_ratio(sharpe=1.5, n_trials=100, variance_of_sharpes=0.3, n_observations=500)

# Group similar strategies
clusters = strategy_clustering(returns_matrix, n_clusters=3)
```

## Cross-Validation Schemes

| Scheme | Use Case |
|--------|----------|
| `WalkForwardCV` | Expanding or rolling window walk-forward |
| `ExpandingCV` | Convenience wrapper for expanding walk-forward |
| `PurgedKFoldCV` | Lopez de Prado purged k-fold with embargo |
| `CombinatorialPurgedCV` | All combinations of test groups for path diversity |

## Documentation

Long-form package documentation lives in `docs/`.

```bash
pip install -e ".[dev]"
mkdocs build --strict
python examples/execution_walkforward_synthetic.py
```

## Development

```bash
# Run tests
python -m pytest tests/ -v

# Lint
ruff check backtester/ tests/

# Build docs
mkdocs build --strict

# Test with coverage
python -m pytest tests/ --cov=backtester --cov-report=term-missing
```

## License

MIT
