"""PnL attribution: signal, instrument, and sector decomposition."""

from __future__ import annotations

import pandas as pd


def signal_attribution(
    weights_history: pd.DataFrame,
    returns: pd.DataFrame,
    signal_weights: dict[str, float],
    signal_scores: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Decompose daily PnL by signal source.

    Returns: date x [signal_1, ..., residual, total].
    """
    total_pnl = (
        weights_history
        * returns.reindex(index=weights_history.index, columns=weights_history.columns).fillna(0)
    ).sum(axis=1)
    total_weight = sum(signal_weights.values()) or 1.0

    result = pd.DataFrame(index=total_pnl.index)

    attributed = pd.Series(0.0, index=total_pnl.index)
    for sig_name, sig_w in signal_weights.items():
        # Proportion of PnL attributed to this signal
        frac = sig_w / total_weight
        result[sig_name] = total_pnl * frac
        attributed = attributed + result[sig_name]

    result["residual"] = total_pnl - attributed
    result["total"] = total_pnl
    return result


def instrument_attribution(
    weights_history: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Decompose PnL by instrument.

    Returns: date x [instruments..., total].
    """
    aligned_returns = returns.reindex(
        index=weights_history.index, columns=weights_history.columns
    ).fillna(0)
    per_instr = weights_history * aligned_returns
    per_instr["total"] = per_instr.sum(axis=1)
    return per_instr


def sector_attribution(
    weights_history: pd.DataFrame,
    returns: pd.DataFrame,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    """Decompose PnL by sector.

    Returns: date x [sectors..., total].
    """
    instr_pnl = instrument_attribution(weights_history, returns)
    total = instr_pnl["total"]
    instr_pnl = instr_pnl.drop(columns=["total"])

    sectors: dict[str, list[str]] = {}
    for instr, sector in sector_map.items():
        sectors.setdefault(sector, []).append(instr)

    result = pd.DataFrame(index=instr_pnl.index)
    for sector, instruments in sorted(sectors.items()):
        cols = [c for c in instruments if c in instr_pnl.columns]
        result[sector] = instr_pnl[cols].sum(axis=1) if cols else 0.0

    result["total"] = total
    return result
