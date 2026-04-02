"""Generic portfolio accounting utilities."""

from __future__ import annotations

import pandas as pd


def _normalize_execution_date(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.normalize()


def held_weights_from_rebalances(
    rebalance_weights: pd.DataFrame,
    execution_dates: pd.Series,
    execution_index: pd.DatetimeIndex,
    *,
    apply_next_bar: bool = True,
) -> pd.DataFrame:
    """Expand rebalance weights onto a denser execution index.

    Parameters
    ----------
    rebalance_weights : pd.DataFrame
        Sparse weight schedule indexed by rebalance decision dates.
    execution_dates : pd.Series
        Mapping from rebalance date to execution timestamp/date.
    execution_index : pd.DatetimeIndex
        Dense index on which held weights should be expressed.
    apply_next_bar : bool, default True
        If True and the execution timestamp falls exactly on the execution index,
        begin holding from the following bar/session.

    Returns
    -------
    pd.DataFrame
        Dense held-weight history aligned to ``execution_index``.
    """
    history = pd.DataFrame(
        0.0,
        index=pd.DatetimeIndex(execution_index),
        columns=rebalance_weights.columns,
        dtype="float64",
    )
    active_dates = rebalance_weights.index[rebalance_weights.notna().any(axis=1)]
    if active_dates.empty:
        return history

    start_positions: list[int] = []
    for date in active_dates:
        execution_date = _normalize_execution_date(execution_dates.loc[date])
        start = int(history.index.searchsorted(execution_date))
        if (
            apply_next_bar
            and start < len(history.index)
            and history.index[start] == execution_date
        ):
            start += 1
        start_positions.append(start)

    for idx, date in enumerate(active_dates):
        start = start_positions[idx]
        end = start_positions[idx + 1] if idx + 1 < len(start_positions) else len(history.index)
        if start >= len(history.index) or start >= end:
            continue
        history.iloc[start:end] = rebalance_weights.loc[date].to_numpy(dtype="float64")
    return history


def per_asset_pnl(
    weights_history: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-asset additive PnL from held weights and asset returns."""
    aligned_returns = returns.reindex(index=weights_history.index, columns=weights_history.columns)
    return weights_history * aligned_returns.fillna(0.0)


def linear_turnover_costs(
    rebalance_weights: pd.DataFrame,
    execution_dates: pd.Series,
    execution_index: pd.DatetimeIndex,
    *,
    half_spread_bps: float,
    multiplier: float = 1.0,
    apply_next_bar: bool = True,
) -> pd.Series:
    """Compute linear turnover costs posted on execution dates."""
    costs = pd.Series(0.0, index=pd.DatetimeIndex(execution_index), dtype="float64")
    active_dates = rebalance_weights.index[rebalance_weights.notna().any(axis=1)]
    if active_dates.empty:
        return costs

    previous = pd.Series(0.0, index=rebalance_weights.columns, dtype="float64")
    for date in active_dates:
        execution_date = _normalize_execution_date(execution_dates.loc[date])
        start = int(costs.index.searchsorted(execution_date))
        if (
            apply_next_bar
            and start < len(costs.index)
            and costs.index[start] == execution_date
        ):
            start += 1
        if start >= len(costs.index):
            continue
        current = rebalance_weights.loc[date].fillna(0.0)
        turnover = float((current - previous).abs().sum())
        costs.iloc[start] = multiplier * half_spread_bps * 1e-4 * turnover
        previous = current
    return costs


def scenario_returns(
    weights_history: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    rebalance_weights: pd.DataFrame,
    execution_dates: pd.Series,
    cost_multipliers: tuple[float, ...],
    base_half_spread_bps: float,
    apply_next_bar: bool = True,
) -> tuple[dict[str, pd.Series], dict[str, pd.DataFrame]]:
    """Compute portfolio returns under multiple linear-cost scenarios."""
    pnl_by_asset = per_asset_pnl(weights_history, returns)
    scenario_total: dict[str, pd.Series] = {}
    scenario_by_asset: dict[str, pd.DataFrame] = {}
    for multiplier in cost_multipliers:
        label = f"{multiplier:.1f}x"
        costs = linear_turnover_costs(
            rebalance_weights,
            execution_dates,
            weights_history.index,
            half_spread_bps=base_half_spread_bps,
            multiplier=multiplier,
            apply_next_bar=apply_next_bar,
        )
        scenario_total[label] = pnl_by_asset.sum(axis=1) - costs
        scenario_by_asset[label] = pnl_by_asset.copy()
    return scenario_total, scenario_by_asset
