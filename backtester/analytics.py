"""Performance analytics: pure functions on return series."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def performance_table(
    returns: pd.Series, annualization: float = 252.0
) -> pd.DataFrame:
    """Comprehensive performance summary."""
    if returns.empty:
        return pd.DataFrame()

    ann_ret = returns.mean() * annualization
    ann_vol = returns.std() * np.sqrt(annualization)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(annualization) if len(downside) > 1 else 0.0
    sortino = ann_ret / downside_std if downside_std > 0 else 0.0

    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = float(dd.min())

    # Max drawdown duration
    underwater = dd < 0
    if underwater.any():
        groups = (~underwater).cumsum()
        dd_lengths = underwater.groupby(groups).sum()
        max_dd_dur = int(dd_lengths.max()) if not dd_lengths.empty else 0
    else:
        max_dd_dur = 0

    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0
    win_rate = (returns > 0).mean()
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    metrics = {
        "annualized_return": ann_ret,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "max_drawdown_duration_days": max_dd_dur,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "skew": float(returns.skew()),
        "kurtosis": float(returns.kurtosis()),
        "best_day": float(returns.max()),
        "worst_day": float(returns.min()),
    }
    return pd.DataFrame([metrics])


def rolling_metrics(
    returns: pd.Series, window: int = 63
) -> pd.DataFrame:
    """Rolling Sharpe, vol, drawdown."""
    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    roll_sharpe = (roll_mean / roll_std.replace(0, float("nan"))) * np.sqrt(252)

    cum = (1 + returns).cumprod()
    roll_max = cum.rolling(window, min_periods=1).max()
    roll_dd = (cum - roll_max) / roll_max

    return pd.DataFrame({
        "rolling_sharpe": roll_sharpe,
        "rolling_vol": roll_std * np.sqrt(252),
        "rolling_drawdown": roll_dd,
    })


def drawdown_table(returns: pd.Series, top_n: int = 5) -> pd.DataFrame:
    """Top N drawdowns: start, trough, depth, duration."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak

    # Find drawdown periods
    is_dd = dd < 0
    if not is_dd.any():
        return pd.DataFrame(columns=["start", "trough", "depth", "duration"])

    # Group consecutive drawdown periods
    groups = (~is_dd).cumsum()
    dd_periods = []
    for g, grp in dd[is_dd].groupby(groups[is_dd]):
        dd_periods.append({
            "start": grp.index[0],
            "trough": grp.idxmin(),
            "depth": float(grp.min()),
            "duration": len(grp),
        })

    result = pd.DataFrame(dd_periods).sort_values("depth").head(top_n).reset_index(drop=True)
    return result


def param_stability(
    params_per_fold: list[dict[str, dict[str, float]]],
) -> pd.DataFrame:
    """Coefficient of variation per parameter across folds."""
    if not params_per_fold:
        return pd.DataFrame(columns=["component", "param", "mean", "std", "cv"])

    rows = []
    # Collect all (comp, param) keys
    all_keys: set[tuple[str, str]] = set()
    for pf in params_per_fold:
        for comp, params in pf.items():
            for pname in params:
                all_keys.add((comp, pname))

    for comp, pname in sorted(all_keys):
        values = [
            pf[comp][pname]
            for pf in params_per_fold
            if comp in pf and pname in pf[comp]
        ]
        if not values:
            continue
        arr = np.array(values)
        mu = arr.mean()
        sigma = arr.std()
        cv = sigma / abs(mu) if abs(mu) > 1e-10 else float("nan")
        rows.append({
            "component": comp,
            "param": pname,
            "mean": mu,
            "std": sigma,
            "cv": cv,
        })
    return pd.DataFrame(rows)


def param_sensitivity(
    evaluate_fn: Callable[[dict[str, dict[str, float]]], float],
    base_params: dict[str, dict[str, float]],
    param_name: str,
    component_name: str,
    n_points: int = 20,
    range_frac: float = 0.5,
) -> pd.DataFrame:
    """Partial dependence: vary one param, measure metric."""
    base_val = base_params[component_name][param_name]
    lo = base_val * (1 - range_frac)
    hi = base_val * (1 + range_frac)
    values = np.linspace(lo, hi, n_points)

    rows = []
    for v in values:
        params = {c: dict(p) for c, p in base_params.items()}
        params[component_name][param_name] = float(v)
        try:
            score = evaluate_fn(params)
        except Exception:
            score = float("nan")
        rows.append({"value": float(v), "score": score})

    return pd.DataFrame(rows)
