# Agents

Guidelines for AI agents working on the backtester codebase.

## Architecture

This library follows a **compose-and-run** pattern:

1. `BacktestRunner` composes a `CVScheme`, list of `Objective`s, and a `BacktestOptimizer`.
2. The user provides a `pipeline` (from alphaforge) and an `evaluate_fn` callback.
3. The runner orchestrates: split → optimize → evaluate → aggregate.

All extensibility is through **protocols**, not inheritance. To add a new CV scheme, objective, or optimizer, implement the corresponding protocol — no registration required.

## Key Protocols

| Protocol | Required Members |
|----------|-----------------|
| `CVScheme` | `name: str`, `splits(dates) -> Iterator[CVSplit]` |
| `Objective` | `name: str`, `direction: str`, `compute(returns, **ctx) -> float` |
| `BacktestOptimizer` | `optimize(base_params, evaluate_fn, n_trials) -> OptimizeResult` |

## Rules for Agents

### Code Style
- Use `ruff` for linting and formatting.
- Type annotations on all public functions.
- Protocols over ABCs. Frozen dataclasses for data containers.
- No mutable global state.

### Testing
- Every new module needs a corresponding `tests/test_<module>.py`.
- Tests must not require network access or external data.
- Use synthetic data (e.g., `np.random.default_rng(42)`) for reproducibility.
- Always run `pytest tests/ -v` before committing.

### Adding New Components

**New CV scheme**: Implement `CVScheme` protocol in `cv.py`, export from `__init__.py`, add tests.

**New objective**: Implement `Objective` protocol in `objectives.py`, export from `__init__.py`, add tests. The `direction` field must be `"maximize"` or `"minimize"`.

**New optimizer**: Implement `BacktestOptimizer` protocol in `optimizers.py`, export from `__init__.py`, add tests. If it requires an optional dependency, follow the `BayesianOptimizer` pattern (try-import with fallback).

**New analytics function**: Add as a pure function in `analytics.py`. Input should be `pd.Series` of returns, output should be `pd.DataFrame`.

### What NOT to Do
- Do not add abstract base classes — use protocols.
- Do not add registries or plugin systems — capability-based design only.
- Do not add state to analytics or attribution functions.
- Do not introduce dependencies beyond the ones declared in `pyproject.toml` without discussion.
- Do not add `__pycache__`, `.pyc`, or build artifacts to version control.
