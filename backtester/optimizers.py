"""Parameter search strategies for backtesting."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OptimizeResult:
    best_params: dict[str, dict[str, float]]
    best_score: float
    all_trials: pd.DataFrame
    n_evaluations: int


@runtime_checkable
class BacktestOptimizer(Protocol):
    name: str

    def optimize(
        self,
        base_params: dict[str, dict[str, float]],
        evaluate_fn: Callable[[dict[str, dict[str, float]]], float],
        n_trials: int,
    ) -> OptimizeResult: ...


@dataclass
class RandomSearchOptimizer:
    name: str = "random_search"
    seed: int = 42
    window_range: tuple[float, float] = (0.3, 2.5)
    weight_std: float = 0.3

    def optimize(
        self,
        base_params: dict[str, dict[str, float]],
        evaluate_fn: Callable[[dict[str, dict[str, float]]], float],
        n_trials: int,
    ) -> OptimizeResult:
        rng = np.random.default_rng(self.seed)
        trials: list[dict] = []

        for trial_id in range(n_trials):
            sampled: dict[str, dict[str, float]] = {}
            for comp, params in base_params.items():
                sampled[comp] = {}
                for pname, pval in params.items():
                    if "window" in pname or "half_life" in pname:
                        lo = max(2, int(pval * self.window_range[0]))
                        hi = int(pval * self.window_range[1]) + 1
                        sampled[comp][pname] = float(rng.integers(lo, hi))
                    else:
                        sampled[comp][pname] = pval + rng.normal(0, self.weight_std)

            try:
                score = evaluate_fn(sampled)
            except Exception:
                score = float("nan")

            trials.append({
                "trial_id": trial_id,
                "params_json": json.dumps(sampled),
                "score": score,
            })

        trials_df = pd.DataFrame(trials)
        valid = trials_df.dropna(subset=["score"])
        if valid.empty:
            return OptimizeResult(base_params, float("nan"), trials_df, n_trials)

        best_idx = valid["score"].idxmax()
        best_row = valid.loc[best_idx]
        return OptimizeResult(
            best_params=json.loads(best_row["params_json"]),
            best_score=float(best_row["score"]),
            all_trials=trials_df,
            n_evaluations=n_trials,
        )


@dataclass
class GridSearchOptimizer:
    name: str = "grid_search"
    param_grid: dict[str, dict[str, list[float]]] = field(default_factory=dict)

    def optimize(
        self,
        base_params: dict[str, dict[str, float]],
        evaluate_fn: Callable[[dict[str, dict[str, float]]], float],
        n_trials: int,
    ) -> OptimizeResult:
        # Build all combinations from param_grid
        combos = [{}]
        for comp, params in self.param_grid.items():
            for pname, values in params.items():
                new_combos = []
                for c in combos:
                    for v in values:
                        nc = dict(c)
                        nc[(comp, pname)] = v
                        new_combos.append(nc)
                combos = new_combos

        trials: list[dict] = []
        for trial_id, combo in enumerate(combos):
            sampled: dict[str, dict[str, float]] = {
                c: dict(p) for c, p in base_params.items()
            }
            for (comp, pname), val in combo.items():
                if comp in sampled:
                    sampled[comp][pname] = val

            try:
                score = evaluate_fn(sampled)
            except Exception:
                score = float("nan")

            trials.append({
                "trial_id": trial_id,
                "params_json": json.dumps(sampled),
                "score": score,
            })

        trials_df = pd.DataFrame(trials)
        valid = trials_df.dropna(subset=["score"])
        if valid.empty:
            return OptimizeResult(base_params, float("nan"), trials_df, len(combos))

        best_idx = valid["score"].idxmax()
        best_row = valid.loc[best_idx]
        return OptimizeResult(
            best_params=json.loads(best_row["params_json"]),
            best_score=float(best_row["score"]),
            all_trials=trials_df,
            n_evaluations=len(combos),
        )


@dataclass
class BayesianOptimizer:
    """Optuna TPE optimizer. Falls back to RandomSearch if optuna not installed."""

    name: str = "bayesian_tpe"
    seed: int = 42
    pruning: bool = True

    def optimize(
        self,
        base_params: dict[str, dict[str, float]],
        evaluate_fn: Callable[[dict[str, dict[str, float]]], float],
        n_trials: int,
    ) -> OptimizeResult:
        try:
            import optuna
        except ImportError:
            return RandomSearchOptimizer(seed=self.seed).optimize(
                base_params, evaluate_fn, n_trials
            )

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        flat_params: list[tuple[str, str, float]] = []
        for comp, params in base_params.items():
            for pname, pval in params.items():
                flat_params.append((comp, pname, pval))

        def _objective(trial: optuna.Trial) -> float:
            sampled: dict[str, dict[str, float]] = {}
            for comp, pname, pval in flat_params:
                if comp not in sampled:
                    sampled[comp] = {}
                if "window" in pname or "half_life" in pname:
                    lo = max(2, int(pval * 0.3))
                    hi = int(pval * 2.5) + 1
                    sampled[comp][pname] = float(trial.suggest_int(f"{comp}.{pname}", lo, hi))
                else:
                    sampled[comp][pname] = trial.suggest_float(
                        f"{comp}.{pname}", pval - 1.0, pval + 1.0
                    )
            return evaluate_fn(sampled)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )
        study.optimize(_objective, n_trials=n_trials)

        trials_data = []
        for t in study.trials:
            trials_data.append({
                "trial_id": t.number,
                "params_json": json.dumps(t.params),
                "score": t.value if t.value is not None else float("nan"),
            })

        best = study.best_trial
        # Reconstruct best_params in nested format
        best_params: dict[str, dict[str, float]] = {}
        for comp, pname, _ in flat_params:
            if comp not in best_params:
                best_params[comp] = {}
            key = f"{comp}.{pname}"
            best_params[comp][pname] = best.params.get(key, base_params[comp][pname])

        return OptimizeResult(
            best_params=best_params,
            best_score=float(best.value) if best.value is not None else float("nan"),
            all_trials=pd.DataFrame(trials_data),
            n_evaluations=n_trials,
        )


@dataclass
class GradientDescentOptimizer:
    """Adapter for gradient-based training via callback."""

    name: str = "gradient_descent"
    train_fn: Callable[..., OptimizeResult] | None = None

    def optimize(
        self,
        base_params: dict[str, dict[str, float]],
        evaluate_fn: Callable[[dict[str, dict[str, float]]], float],
        n_trials: int,
    ) -> OptimizeResult:
        if self.train_fn is None:
            raise ValueError("GradientDescentOptimizer requires a train_fn callback")
        return self.train_fn(base_params, evaluate_fn, n_trials)
