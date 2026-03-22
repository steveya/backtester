"""Cross-validation schemes for backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterator, Protocol, runtime_checkable

import pandas as pd


@dataclass(frozen=True)
class CVSplit:
    fold_id: str
    train_dates: pd.DatetimeIndex
    eval_dates: pd.DatetimeIndex
    purge_dates: pd.DatetimeIndex


@runtime_checkable
class CVScheme(Protocol):
    """Generate train/eval splits from a date index."""

    name: str

    def splits(self, dates: pd.DatetimeIndex) -> Iterator[CVSplit]: ...


@dataclass
class WalkForwardCV:
    """Walk-forward CV: expanding or rolling window."""

    name: str = "walk_forward"
    train_window: int | None = None  # None = expanding
    eval_window: int = 63
    step: int = 21
    min_train: int = 252

    def splits(self, dates: pd.DatetimeIndex) -> Iterator[CVSplit]:
        n = len(dates)
        fold = 0
        i = self.min_train
        while i + self.eval_window <= n:
            if self.train_window is not None:
                train_start = max(0, i - self.train_window)
            else:
                train_start = 0
            train = dates[train_start:i]
            eval_ = dates[i : i + self.eval_window]
            yield CVSplit(
                fold_id=f"wf_{fold}",
                train_dates=pd.DatetimeIndex(train),
                eval_dates=pd.DatetimeIndex(eval_),
                purge_dates=pd.DatetimeIndex([]),
            )
            fold += 1
            i += self.step


@dataclass
class ExpandingCV:
    """Expanding window CV (convenience wrapper)."""

    name: str = "expanding"
    eval_window: int = 63
    step: int = 21
    min_train: int = 252

    def splits(self, dates: pd.DatetimeIndex) -> Iterator[CVSplit]:
        wf = WalkForwardCV(
            train_window=None,
            eval_window=self.eval_window,
            step=self.step,
            min_train=self.min_train,
        )
        yield from wf.splits(dates)


@dataclass
class PurgedKFoldCV:
    """Lopez de Prado purged k-fold CV."""

    name: str = "purged_kfold"
    n_splits: int = 5
    purge: int = 5
    embargo: int = 10

    def splits(self, dates: pd.DatetimeIndex) -> Iterator[CVSplit]:
        n = len(dates)
        fold_size = n // self.n_splits

        for k in range(self.n_splits):
            eval_start = k * fold_size
            eval_end = (k + 1) * fold_size if k < self.n_splits - 1 else n
            eval_dates = dates[eval_start:eval_end]

            # Purge: exclude dates just before eval
            purge_start = max(0, eval_start - self.purge)
            purge_dates = dates[purge_start:eval_start]

            # Embargo: exclude dates just after eval
            embargo_end = min(n, eval_end + self.embargo)
            embargo_dates = dates[eval_end:embargo_end]

            all_purged = purge_dates.append(embargo_dates)

            # Train: everything not in eval, purge, or embargo
            excluded = set(eval_dates) | set(purge_dates) | set(embargo_dates)
            train_dates = pd.DatetimeIndex([d for d in dates if d not in excluded])

            yield CVSplit(
                fold_id=f"pkf_{k}",
                train_dates=train_dates,
                eval_dates=pd.DatetimeIndex(eval_dates),
                purge_dates=pd.DatetimeIndex(all_purged),
            )


@dataclass
class CombinatorialPurgedCV:
    """Combinatorial purged CV: C(n_groups, test_groups) paths."""

    name: str = "combinatorial_purged"
    n_groups: int = 6
    test_groups: int = 2
    purge: int = 5
    embargo: int = 10

    def splits(self, dates: pd.DatetimeIndex) -> Iterator[CVSplit]:
        n = len(dates)
        group_size = n // self.n_groups
        groups: list[pd.DatetimeIndex] = []
        for g in range(self.n_groups):
            start = g * group_size
            end = (g + 1) * group_size if g < self.n_groups - 1 else n
            groups.append(dates[start:end])

        fold = 0
        for test_combo in combinations(range(self.n_groups), self.test_groups):
            eval_set: set = set()
            for g in test_combo:
                eval_set.update(groups[g])

            # Purge/embargo at each group boundary
            purge_set: set = set()
            for g in test_combo:
                g_start = g * group_size
                g_end = (g + 1) * group_size if g < self.n_groups - 1 else n
                p_start = max(0, g_start - self.purge)
                e_end = min(n, g_end + self.embargo)
                for idx in range(p_start, g_start):
                    purge_set.add(dates[idx])
                for idx in range(g_end, e_end):
                    if idx < n:
                        purge_set.add(dates[idx])

            excluded = eval_set | purge_set
            train_dates = pd.DatetimeIndex([d for d in dates if d not in excluded])

            yield CVSplit(
                fold_id=f"cpcv_{fold}",
                train_dates=train_dates,
                eval_dates=pd.DatetimeIndex(sorted(eval_set)),
                purge_dates=pd.DatetimeIndex(sorted(purge_set)),
            )
            fold += 1
