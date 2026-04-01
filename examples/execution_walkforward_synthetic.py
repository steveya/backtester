"""Provider-agnostic execution-clock walk-forward example with synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtester import (  # noqa: E402
    EventRebalanceEngine,
    ExecutionBarFrame,
    SharpeObjective,
    TargetSchedule,
    TurnoverObjective,
    WalkForwardCV,
    WalkForwardExecutionRunner,
)


def build_execution_bars() -> ExecutionBarFrame:
    """Create a dense 5-minute execution clock for two assets."""

    timestamps = pd.date_range("2024-03-04 14:30:00+00:00", periods=36, freq="5min")
    rows: list[dict[str, object]] = []
    for asset, start_price, drift in (
        ("CL", 80.0, 0.18),
        ("BZ", 82.0, -0.10),
    ):
        for idx, ts in enumerate(timestamps):
            rows.append(
                {
                    "ts": ts,
                    "asset": asset,
                    "close": start_price + drift * idx,
                    "volume": 1000.0 + 25.0 * idx,
                    "active_contract_id": f"{asset}M4",
                    "roll_flag": idx == 18,
                }
            )
    return ExecutionBarFrame(pd.DataFrame(rows))


def build_decision_index() -> pd.DatetimeIndex:
    """Create a lower-frequency decision clock separate from the execution bars."""

    return pd.date_range("2024-03-04 14:30:00+00:00", periods=6, freq="30min")


def generate_targets(context) -> TargetSchedule:
    """Build one exposure snapshot for each evaluation decision timestamp."""

    close_matrix = context.train_bars.to_matrix("close")
    trailing_return = close_matrix.iloc[-1] / close_matrix.iloc[0] - 1.0
    winner = str(trailing_return.idxmax())
    loser = str(trailing_return.idxmin())

    targets = pd.DataFrame(0.0, index=context.eval_decision_times, columns=close_matrix.columns)
    targets[winner] = 1.0
    targets[loser] = -1.0
    return TargetSchedule(targets, target_kind="exposures")


def main() -> None:
    bars = build_execution_bars()
    decision_index = build_decision_index()

    runner = WalkForwardExecutionRunner(
        cv=WalkForwardCV(eval_window=1, step=1, min_train=3),
        objectives=[SharpeObjective(), TurnoverObjective()],
        engine=EventRebalanceEngine(),
    )
    result = runner.run(bars, decision_index, generate_targets)

    print("Walk-forward metrics")
    print(result.metrics)
    print()
    print("Per-fold metrics")
    print(result.per_fold_metrics.to_string(index=False))
    print()
    print("Event log")
    print(result.event_log.to_string(index=False))


if __name__ == "__main__":
    main()
