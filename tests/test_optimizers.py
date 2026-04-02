"""Tests for backtester.optimizers module."""

from __future__ import annotations

from backtester.optimizers import (
    BacktestOptimizer,
    GradientDescentOptimizer,
    GridSearchOptimizer,
    NoOpOptimizer,
    OptimizeResult,
    RandomSearchOptimizer,
)


def _simple_eval(params: dict[str, dict[str, float]]) -> float:
    """Higher window → higher score (simple planted optimum)."""
    w = params.get("comp", {}).get("window", 50)
    return -((w - 80) ** 2)  # optimum at w=80


class TestRandomSearch:
    def test_random_search(self) -> None:
        opt = RandomSearchOptimizer(seed=42)
        result = opt.optimize(
            {"comp": {"window": 50.0}},
            _simple_eval,
            n_trials=20,
        )
        assert isinstance(result, OptimizeResult)
        assert result.n_evaluations == 20
        assert len(result.all_trials) == 20


class TestNoOpOptimizer:
    def test_no_op_optimizer(self) -> None:
        base_params = {"comp": {"window": 50.0, "weight": 0.2}}
        calls: list[dict[str, dict[str, float]]] = []

        def _eval(params: dict[str, dict[str, float]]) -> float:
            calls.append(params)
            return 1.23

        opt = NoOpOptimizer()
        result = opt.optimize(base_params, _eval, n_trials=99)

        assert result.best_params == base_params
        assert result.best_score == 1.23
        assert result.n_evaluations == 1
        assert len(result.all_trials) == 1
        assert calls == [base_params]


class TestGridSearchExhaustive:
    def test_grid_search_exhaustive(self) -> None:
        opt = GridSearchOptimizer(param_grid={"comp": {"window": [40.0, 60.0, 80.0]}})
        result = opt.optimize(
            {"comp": {"window": 50.0}},
            _simple_eval,
            n_trials=100,  # ignored for grid
        )
        assert result.n_evaluations == 3
        # Best should be close to 80
        assert abs(result.best_params["comp"]["window"] - 80.0) < 1e-6


class TestBayesianFindsOptimum:
    def test_bayesian_finds_optimum(self) -> None:
        # Falls back to random search if optuna not installed
        from backtester.optimizers import BayesianOptimizer

        opt = BayesianOptimizer(seed=42)
        result = opt.optimize(
            {"comp": {"window": 50.0}},
            _simple_eval,
            n_trials=15,
        )
        assert isinstance(result, OptimizeResult)
        assert result.best_score > _simple_eval({"comp": {"window": 50.0}})


class TestGradientDescentCallback:
    def test_gradient_descent_callback(self) -> None:
        called = []

        def _train(base, evaluate_fn, n_trials):
            called.append(True)
            return OptimizeResult(base, 1.0, __import__("pandas").DataFrame(), 1)

        opt = GradientDescentOptimizer(train_fn=_train)
        opt.optimize({"comp": {"window": 50.0}}, _simple_eval, 10)
        assert len(called) == 1


class TestOptimizerSatisfiesProtocol:
    def test_optimizer_satisfies_protocol(self) -> None:
        assert isinstance(NoOpOptimizer(), BacktestOptimizer)
        assert isinstance(RandomSearchOptimizer(), BacktestOptimizer)
        assert isinstance(GridSearchOptimizer(), BacktestOptimizer)
