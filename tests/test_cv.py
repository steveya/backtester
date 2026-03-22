"""Tests for backtester.cv module."""

from __future__ import annotations

import pandas as pd

from backtester.cv import (
    CombinatorialPurgedCV,
    CVScheme,
    ExpandingCV,
    PurgedKFoldCV,
    WalkForwardCV,
)

_DATES = pd.bdate_range("2023-01-02", periods=500)


class TestWalkForwardSplits:
    def test_walk_forward_splits(self) -> None:
        cv = WalkForwardCV(eval_window=63, step=21, min_train=252)
        splits = list(cv.splits(_DATES))
        assert len(splits) > 0
        for s in splits:
            # No overlap between train and eval
            assert len(set(s.train_dates) & set(s.eval_dates)) == 0
            assert len(s.eval_dates) == 63


class TestWalkForwardRolling:
    def test_walk_forward_rolling(self) -> None:
        cv = WalkForwardCV(train_window=100, eval_window=63, step=21, min_train=100)
        splits = list(cv.splits(_DATES))
        assert len(splits) > 0
        for s in splits:
            assert len(s.train_dates) <= 100


class TestWalkForwardExpanding:
    def test_walk_forward_expanding(self) -> None:
        cv = WalkForwardCV(train_window=None, eval_window=63, step=21, min_train=252)
        splits = list(cv.splits(_DATES))
        # Train size should grow monotonically
        train_sizes = [len(s.train_dates) for s in splits]
        assert train_sizes == sorted(train_sizes)


class TestPurgedKFoldPurgeGap:
    def test_purged_kfold_purge_gap(self) -> None:
        cv = PurgedKFoldCV(n_splits=5, purge=5, embargo=0)
        splits = list(cv.splits(_DATES))
        assert len(splits) == 5
        # Non-first folds should have purge dates
        mid_splits = [s for s in splits if not s.fold_id.endswith("_0")]
        assert any(len(s.purge_dates) > 0 for s in mid_splits)


class TestPurgedKFoldEmbargo:
    def test_purged_kfold_embargo(self) -> None:
        cv = PurgedKFoldCV(n_splits=5, purge=0, embargo=10)
        splits = list(cv.splits(_DATES))
        # Non-last folds should have embargo dates
        non_last = [s for s in splits if not s.fold_id.endswith(f"_{cv.n_splits - 1}")]
        assert any(len(s.purge_dates) > 0 for s in non_last)


class TestPurgedKFoldNoLeakage:
    def test_purged_kfold_no_leakage(self) -> None:
        cv = PurgedKFoldCV(n_splits=5, purge=5, embargo=10)
        splits = list(cv.splits(_DATES))
        for s in splits:
            train_set = set(s.train_dates)
            eval_set = set(s.eval_dates)
            assert len(train_set & eval_set) == 0


class TestCombinatorialPurgedPathCount:
    def test_combinatorial_purged_path_count(self) -> None:
        cv = CombinatorialPurgedCV(n_groups=6, test_groups=2, purge=5, embargo=10)
        splits = list(cv.splits(_DATES))
        # C(6,2) = 15
        assert len(splits) == 15


class TestCombinatorialPurgedNoLeakage:
    def test_combinatorial_purged_no_leakage(self) -> None:
        cv = CombinatorialPurgedCV(n_groups=6, test_groups=2, purge=5, embargo=10)
        splits = list(cv.splits(_DATES))
        for s in splits:
            assert len(set(s.train_dates) & set(s.eval_dates)) == 0


class TestExpandingCV:
    def test_expanding_cv(self) -> None:
        cv = ExpandingCV(eval_window=63, step=21, min_train=252)
        splits = list(cv.splits(_DATES))
        train_sizes = [len(s.train_dates) for s in splits]
        assert train_sizes == sorted(train_sizes)


class TestCVSatisfiesProtocol:
    def test_cv_satisfies_protocol(self) -> None:
        for cls in (WalkForwardCV, PurgedKFoldCV, CombinatorialPurgedCV, ExpandingCV):
            assert isinstance(cls(), CVScheme)
