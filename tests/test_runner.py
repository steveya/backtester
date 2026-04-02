"""Tests for backtester.runner module."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from alphaforge.pipeline.protocols import PipelineVariant, SimplePipeline

from backtester.cv import CVSplit, PurgedKFoldCV, WalkForwardCV
from backtester.objectives import MaxDrawdownObjective, SharpeObjective
from backtester.optimizers import NoOpOptimizer, RandomSearchOptimizer
from backtester.result import EvalFoldResult
from backtester.runner import BacktestRunner

_DATES = pd.bdate_range("2023-01-02", periods=500)
_RNG = np.random.default_rng(42)
_RETURNS = pd.DataFrame(
    _RNG.normal(0.0001, 0.01, (500, 3)),
    index=_DATES,
    columns=["a", "b", "c"],
)


@dataclass
class _StubSignal:
    name: str = "stub"

    def score(self, df, **params):
        return pd.Series({"a": 0.5, "b": -0.3, "c": 0.1})


@dataclass
class _StubPipeline:
    name: str = "test"

    def run(self, df, **params):
        return pd.Series({"a": 0.5, "b": -0.3, "c": 0.1})

    def get_all_params(self):
        return {"stub": {"window": 50.0}}

    def set_all_params(self, params):
        pass


def _make_eval_fn(returns: pd.DataFrame):
    def evaluate(pipeline, split: CVSplit) -> EvalFoldResult:
        scores = pipeline.run(pd.DataFrame())
        eval_ret = returns.loc[split.eval_dates].reindex(columns=scores.index).fillna(0)
        port_ret = eval_ret.mul(scores, axis=1).sum(axis=1)
        wh = pd.DataFrame(
            [scores.values] * len(split.eval_dates),
            index=split.eval_dates,
            columns=scores.index,
        )
        return EvalFoldResult(portfolio_returns=port_ret, weights_history=wh)

    return evaluate


class TestRunnerWalkForward:
    def test_runner_walk_forward(self) -> None:
        runner = BacktestRunner(
            cv=WalkForwardCV(eval_window=63, step=63, min_train=252),
            objectives=[SharpeObjective()],
            optimizer=RandomSearchOptimizer(seed=42),
        )
        result = runner.run(_StubPipeline(), _make_eval_fn(_RETURNS), _DATES, n_trials=3)
        assert result.n_folds > 0
        assert "sharpe" in result.metrics


class TestRunnerPurgedKFold:
    def test_runner_purged_kfold(self) -> None:
        runner = BacktestRunner(
            cv=PurgedKFoldCV(n_splits=3, purge=5, embargo=5),
            objectives=[SharpeObjective()],
            optimizer=RandomSearchOptimizer(seed=42),
        )
        result = runner.run(_StubPipeline(), _make_eval_fn(_RETURNS), _DATES, n_trials=3)
        assert result.n_folds == 3
        assert len(result.per_fold_metrics) == 3


class TestRunnerMultiObjective:
    def test_runner_multi_objective(self) -> None:
        runner = BacktestRunner(
            cv=WalkForwardCV(eval_window=63, step=63, min_train=252),
            objectives=[SharpeObjective(), MaxDrawdownObjective()],
            optimizer=RandomSearchOptimizer(seed=42),
        )
        result = runner.run(_StubPipeline(), _make_eval_fn(_RETURNS), _DATES, n_trials=3)
        assert "sharpe" in result.metrics
        assert "max_drawdown" in result.metrics


class TestRunnerVariants:
    def test_runner_run_variants(self) -> None:
        v1 = PipelineVariant("v1", SimplePipeline(name="p1", signal=_StubSignal()))
        v2 = PipelineVariant("v2", SimplePipeline(name="p2", signal=_StubSignal()))

        runner = BacktestRunner(
            cv=WalkForwardCV(eval_window=63, step=63, min_train=252),
            objectives=[SharpeObjective()],
            optimizer=RandomSearchOptimizer(seed=42),
        )
        result = runner.run_variants([v1, v2], _make_eval_fn(_RETURNS), _DATES, n_trials=3)
        assert result.best_variant in ("v1", "v2")
        assert len(result.rankings) == 2


class TestRunnerEmptyPipeline:
    def test_runner_empty_pipeline(self) -> None:
        @dataclass
        class _EmptyPipeline:
            name: str = "empty"

            def run(self, df, **p):
                return pd.Series(dtype="float64")

            def get_all_params(self):
                return {}

            def set_all_params(self, p):
                pass

        def empty_eval(pipeline, split):
            return EvalFoldResult(
                portfolio_returns=pd.Series(dtype="float64"),
                weights_history=pd.DataFrame(),
            )

        runner = BacktestRunner(
            cv=WalkForwardCV(eval_window=63, step=63, min_train=252),
            objectives=[SharpeObjective()],
            optimizer=RandomSearchOptimizer(seed=42),
        )
        result = runner.run(_EmptyPipeline(), empty_eval, _DATES, n_trials=3)
        assert result.n_folds >= 0


class TestRunnerObjectiveContext:
    def test_runner_passes_context_and_preserves_artifacts(self) -> None:
        captured: list[dict[str, object]] = []

        @dataclass
        class _ContextObjective:
            name: str = "context"
            direction: str = "maximize"

            def compute(self, portfolio_returns, **context):
                captured.append(context)
                weights_history = context.get("weights_history")
                return float(len(weights_history)) if weights_history is not None else 0.0

        def evaluate(pipeline, split: CVSplit) -> EvalFoldResult:
            scores = pipeline.run(pd.DataFrame())
            eval_ret = _RETURNS.loc[split.eval_dates].reindex(columns=scores.index).fillna(0)
            port_ret = eval_ret.mul(scores, axis=1).sum(axis=1)
            weights = pd.DataFrame(
                [scores.values] * len(split.eval_dates),
                index=split.eval_dates,
                columns=scores.index,
            )
            return EvalFoldResult(
                portfolio_returns=port_ret,
                weights_history=weights,
                signal_scores={"default": scores},
                artifacts={"fold_id": split.fold_id},
            )

        runner = BacktestRunner(
            cv=WalkForwardCV(eval_window=63, step=63, min_train=252),
            objectives=[_ContextObjective()],
            optimizer=NoOpOptimizer(),
        )
        result = runner.run(_StubPipeline(), evaluate, _DATES, n_trials=1)

        assert result.fold_artifacts is not None
        assert len(result.fold_artifacts) == result.n_folds
        assert all("fold_id" in artifact for artifact in result.fold_artifacts)
        assert len(captured) >= result.n_folds + 1

        fold_contexts = [context for context in captured if context.get("split") is not None]
        assert fold_contexts
        assert all(context.get("weights_history") is not None for context in fold_contexts)
        assert all(context.get("signal_scores") is not None for context in fold_contexts)
        assert all(context.get("artifacts") is not None for context in fold_contexts)
