# Backtester

Production-grade backtesting framework for systematic trading strategies.

## Project Structure

```
backtester/
├── runner.py        # BacktestRunner — main orchestrator
├── result.py        # Immutable result dataclasses (BacktestResult, EvalFoldResult, VariantBacktestResult)
├── cv.py            # Cross-validation schemes (WalkForwardCV, PurgedKFoldCV, CombinatorialPurgedCV, ExpandingCV)
├── objectives.py    # Optimization objectives (Sharpe, Sortino, Calmar, MaxDrawdown, Turnover, Composite)
├── optimizers.py    # Parameter search (RandomSearch, GridSearch, Bayesian, GradientDescent)
├── analytics.py     # Performance analysis pure functions
├── attribution.py   # PnL decomposition (signal, instrument, sector)
├── statistical.py   # Significance testing (bootstrap, deflated Sharpe, clustering)
tests/
├── test_*.py        # One test file per module (~40 tests)
```

## Design Principles

- **Protocol-based polymorphism**: `CVScheme`, `Objective`, `BacktestOptimizer` use `@runtime_checkable` protocols — no ABC inheritance.
- **Immutable results**: All result dataclasses use `frozen=True`.
- **Pure analytics**: `analytics.py` and `attribution.py` are stateless pure functions.
- **Graceful degradation**: `BayesianOptimizer` falls back to `RandomSearchOptimizer` when optuna is not installed.
- **Capability-based design**: Prefer structural subtyping over registries or observable patterns.

## Development

```bash
# Install with dev deps
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check backtester/ tests/

# With optional deps
pip install -e ".[bayesian,stats,dev]"
```

## Dependencies

- **Core**: numpy>=1.23, pandas>=2.0, alphaforge (git dep)
- **Optional**: optuna>=3.0 (bayesian), scipy>=1.11 (stats)
- **Dev**: pytest, pytest-cov, ruff

## Conventions

- All public APIs are exported from `backtester/__init__.py`.
- Test files mirror source files: `backtester/foo.py` → `tests/test_foo.py`.
- Functions in `analytics.py` take `pd.Series` of returns and return `pd.DataFrame`.
- Attribution functions return DataFrames with a `total` column that sums to portfolio PnL.
- CV splits always include `purge_dates` to prevent lookahead bias.
