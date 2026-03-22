# Backtester

A production-grade backtesting framework for systematic trading strategies, with built-in cross-validation, parameter optimization, performance analytics, and statistical testing.

## Features

- **Walk-forward & purged cross-validation** — avoid lookahead bias with `WalkForwardCV`, `PurgedKFoldCV`, `CombinatorialPurgedCV`, and `ExpandingCV`
- **Multiple optimization backends** — random search, grid search, Bayesian (optuna), and gradient-descent callback adapter
- **Composable objectives** — Sharpe, Sortino, Calmar, max drawdown, turnover, and weighted composites
- **Performance analytics** — performance tables, rolling metrics, drawdown analysis, parameter stability and sensitivity
- **PnL attribution** — decompose returns by signal, instrument, or sector
- **Statistical rigor** — paired bootstrap tests, deflated Sharpe ratio, strategy clustering

## Installation

```bash
pip install -e .

# With optional dependencies
pip install -e ".[bayesian]"   # Adds optuna for Bayesian optimization
pip install -e ".[stats]"     # Adds scipy for statistical tests
pip install -e ".[dev]"       # Adds pytest, ruff for development
```

## Quick Start

```python
import pandas as pd
from backtester import (
    BacktestRunner,
    WalkForwardCV,
    SharpeObjective,
    RandomSearchOptimizer,
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
    optimizer=RandomSearchOptimizer(seed=42),
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

## Development

```bash
# Run tests
pytest tests/ -v

# Lint
ruff check backtester/ tests/

# Test with coverage
pytest tests/ --cov=backtester --cov-report=term-missing
```

## License

MIT
