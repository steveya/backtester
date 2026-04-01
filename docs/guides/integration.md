# Integration Guide

This guide describes the intended boundary between an application package and
`backtester`.

## Ownership boundary

The application package owns:

- data loading
- signal construction
- release and decision timing assumptions
- instrument-specific sizing logic

`backtester` owns:

- execution-clock data contracts
- cross-validation on the decision index
- fold orchestration
- execution simulation
- fold/result aggregation

That means `backtester` should receive prepared inputs, not provider configs,
artifact roots, or raw ingestion workflows.

## Required inputs

The execution-clock runner needs three inputs:

1. `ExecutionBarFrame`
2. decision index as `DatetimeIndex`
3. fold-aware target generator returning `TargetSchedule`

The minimal integration shape is:

```python
import pandas as pd
from backtester import (
    EventRebalanceEngine,
    ExecutionBarFrame,
    TargetSchedule,
    WalkForwardCV,
    WalkForwardExecutionRunner,
)

bars = ExecutionBarFrame.from_frame(
    data=execution_df,
    timestamp_col="available_at_utc",
    asset_col="series_key",
    close_col="close",
    volume_col="volume",
)

decision_index = pd.DatetimeIndex(signal_release_times_utc)

def generate_targets(context):
    target_matrix = build_targets_for(context)
    return TargetSchedule(target_matrix, target_kind="weights")

runner = WalkForwardExecutionRunner(
    cv=WalkForwardCV(eval_window=4, step=4, min_train=12),
    objectives=[...],
    engine=EventRebalanceEngine(),
)

result = runner.run(bars, decision_index, generate_targets)
```

## Timestamp semantics

`ExecutionBarFrame.ts` should be interpreted as the first eligible execution
timestamp for that row.

If your provider exposes 5-minute bars keyed by an availability timestamp such
as `available_at_utc`, pass that field directly as `ts`. Do not relabel it as a
bar-start time inside the backtester.

The runner uses the decision index to define fold boundaries, then maps each
selected decision timestamp to its holding interval on the execution clock until
the next decision boundary.

## Preserved metadata

`ExecutionBarFrame` preserves non-standard columns from the application-owned
input frame. This is the right place to carry execution metadata such as:

- `active_contract_id`
- `roll_flag`
- venue/session labels
- participation or liquidity fields

The current execution engine only requires `close`, but preserved metadata lets
later cost and execution layers stay causal without changing the input contract.

## Target generation contract

The target generator receives `ExecutionFoldContext`, which exposes:

- `train_decision_times`
- `eval_decision_times`
- `purge_decision_times`
- `train_bars`
- `eval_bars`

The generator must return targets only for `eval_decision_times`.

This keeps signal construction and release logic in the application package
while ensuring the backtester controls fold boundaries and execution intervals.

## Example

The repository includes a runnable synthetic example in
`examples/execution_walkforward_synthetic.py`.

Run it from the repository root:

```bash
python examples/execution_walkforward_synthetic.py
```
