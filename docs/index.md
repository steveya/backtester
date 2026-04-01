# backtester

`backtester` is a Python library for systematic strategy evaluation. The current
package ships a compose-and-run backtesting workflow built around
cross-validation, parameter optimization, objective evaluation, and post-run
analytics.

## Current package surface

The current release includes:

- Cross-validation schemes for walk-forward and purged validation workflows
- Provider-agnostic execution-bar and target-schedule abstractions for execution-clock simulation
- A simple event-based rebalance engine for sparse decision schedules on a dense execution clock
- A walk-forward execution runner that splits on a separate decision index and evaluates on execution-bar intervals
- Objective functions for common portfolio metrics such as Sharpe and drawdown
- Optimizer adapters for random, grid, Bayesian, and callback-driven search
- A generic `BacktestRunner` for fold orchestration and result aggregation
- Analytics, attribution, and statistical utility functions

The public import surface is summarized in the [API overview](api/index.md).

## Documentation map

- [Architecture](architecture.md): current package design and how the modules fit together
- [Integration guide](guides/integration.md): how an application package hands data and targets into the backtester
- [Development](development.md): local setup, tests, linting, and docs build workflow
- [API overview](api/index.md): curated guide to the public package surface

## Installation

Install the base package:

```bash
pip install -e .
```

For contributor workflows, use the development extras:

```bash
pip install -e ".[dev]"
```

## Current status

The documentation in this site describes the functionality currently shipped in
the repository. The execution-clock slice is now part of the public API through
the execution data abstractions, the event-based execution engine, and the
walk-forward execution runner. Cost modeling, execution algorithms, and
provider-specific integrations remain follow-on work.
