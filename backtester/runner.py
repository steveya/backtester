"""BacktestRunner: generic orchestrator for backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
from alphaforge.logging import get_logger
from alphaforge.pipeline.protocols import Pipeline, PipelineVariant

from .cv import CVScheme, CVSplit
from .objectives import Objective
from .optimizers import BacktestOptimizer
from .result import BacktestResult, EvalFoldResult, VariantBacktestResult

log = get_logger(__name__)


@dataclass
class BacktestRunner:
    """Generic backtest orchestrator. Composes CV + Optimizer + Objectives."""

    cv: CVScheme
    objectives: list[Objective]
    optimizer: BacktestOptimizer

    def run(
        self,
        pipeline: Pipeline,
        evaluate_fn: Callable[[Pipeline, CVSplit], EvalFoldResult],
        dates: pd.DatetimeIndex,
        n_trials: int = 50,
    ) -> BacktestResult:
        """Run the full backtest."""
        all_returns: list[pd.Series] = []
        all_weights: list[pd.DataFrame] = []
        per_fold_rows: list[dict] = []
        params_per_fold: list[dict] = []

        splits = list(self.cv.splits(dates))
        if not splits:
            return self._empty_result(dates)

        for split in splits:
            log.info("backtest_fold", fold=split.fold_id, eval_dates=len(split.eval_dates))

            # Optimize on train fold
            base_params = pipeline.get_all_params()

            def _eval_params(params: dict[str, dict[str, float]]) -> float:
                pipeline.set_all_params(params)
                try:
                    result = evaluate_fn(pipeline, split)
                    if result.portfolio_returns.empty:
                        return float("nan")
                    return self.objectives[0].compute(result.portfolio_returns)
                except Exception:
                    return float("nan")

            opt_result = self.optimizer.optimize(base_params, _eval_params, n_trials)
            pipeline.set_all_params(opt_result.best_params)
            params_per_fold.append(opt_result.best_params)

            # Evaluate with optimized params
            try:
                fold_result = evaluate_fn(pipeline, split)
            except Exception:
                continue

            all_returns.append(fold_result.portfolio_returns)
            all_weights.append(fold_result.weights_history)

            # Compute all objectives
            row = {"fold_id": split.fold_id}
            for obj in self.objectives:
                row[obj.name] = obj.compute(fold_result.portfolio_returns)
            per_fold_rows.append(row)

            # Restore base params for next fold
            pipeline.set_all_params(base_params)

        if not all_returns:
            return self._empty_result(dates)

        combined_returns = pd.concat(all_returns).sort_index()
        combined_weights = pd.concat(all_weights).sort_index()
        per_fold_df = pd.DataFrame(per_fold_rows)

        # Aggregate metrics
        metrics: dict[str, float] = {}
        for obj in self.objectives:
            metrics[obj.name] = obj.compute(combined_returns)

        return BacktestResult(
            weights_history=combined_weights,
            returns_history=combined_returns,
            signal_history=None,
            metrics=metrics,
            per_fold_metrics=per_fold_df,
            optimized_params=params_per_fold[-1] if params_per_fold else {},
            params_per_fold=params_per_fold,
            cv_scheme=self.cv.name,
            optimizer=self.optimizer.name,
            n_folds=len(splits),
            n_eval_dates=len(combined_returns),
            date_range=(dates[0], dates[-1]),
        )

    def run_variants(
        self,
        variants: list[PipelineVariant],
        evaluate_fn: Callable[[Pipeline, CVSplit], EvalFoldResult],
        dates: pd.DatetimeIndex,
        n_trials: int = 50,
    ) -> VariantBacktestResult:
        """Run backtest for each variant, compare."""
        per_variant: dict[str, BacktestResult] = {}
        for v in variants:
            log.info("backtest_variant", variant=v.name)
            result = self.run(v.pipeline, evaluate_fn, dates, n_trials)
            per_variant[v.name] = result

        # Build rankings
        ranking_rows = []
        for name, r in per_variant.items():
            row = {"variant_name": name}
            row.update(r.metrics)
            ranking_rows.append(row)

        rankings = pd.DataFrame(ranking_rows)
        primary = self.objectives[0].name
        if primary in rankings.columns and rankings[primary].notna().any():
            asc = self.objectives[0].direction == "minimize"
            rankings["rank"] = rankings[primary].rank(ascending=asc)
            best = rankings.loc[
                rankings[primary].idxmax() if not asc else rankings[primary].idxmin(),
                "variant_name",
            ]
        else:
            rankings["rank"] = float("nan")
            best = variants[0].name if variants else ""

        return VariantBacktestResult(
            per_variant=per_variant,
            rankings=rankings,
            best_variant=best,
        )

    def _empty_result(self, dates: pd.DatetimeIndex) -> BacktestResult:
        return BacktestResult(
            weights_history=pd.DataFrame(),
            returns_history=pd.Series(dtype="float64"),
            signal_history=None,
            metrics={obj.name: float("nan") for obj in self.objectives},
            per_fold_metrics=pd.DataFrame(),
            optimized_params={},
            params_per_fold=[],
            cv_scheme=self.cv.name,
            optimizer=self.optimizer.name,
            n_folds=0,
            n_eval_dates=0,
            date_range=(dates[0], dates[-1]) if len(dates) > 0 else (pd.NaT, pd.NaT),
        )
