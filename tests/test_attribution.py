"""Tests for backtester.attribution module."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.attribution import (
    instrument_attribution,
    sector_attribution,
    signal_attribution,
)

_DATES = pd.bdate_range("2024-01-02", periods=10)
_INSTRUMENTS = ["eur", "gbp", "jpy"]
_RETURNS = pd.DataFrame(
    np.random.default_rng(42).normal(0, 0.01, (10, 3)),
    index=_DATES,
    columns=_INSTRUMENTS,
)
_WEIGHTS = pd.DataFrame(
    [[0.4, 0.3, 0.3]] * 10,
    index=_DATES,
    columns=_INSTRUMENTS,
)


class TestSignalAttributionSums:
    def test_signal_attribution_sums(self) -> None:
        result = signal_attribution(
            _WEIGHTS, _RETURNS,
            {"sig1": 0.6, "sig2": 0.4},
            {"sig1": pd.DataFrame(), "sig2": pd.DataFrame()},
        )
        # Per-signal + residual = total
        check = result["sig1"] + result["sig2"] + result["residual"]
        np.testing.assert_allclose(check.values, result["total"].values, atol=1e-12)


class TestInstrumentAttributionSums:
    def test_instrument_attribution_sums(self) -> None:
        result = instrument_attribution(_WEIGHTS, _RETURNS)
        per_instr = result[_INSTRUMENTS].sum(axis=1)
        np.testing.assert_allclose(per_instr.values, result["total"].values, atol=1e-12)


class TestSectorAttribution:
    def test_sector_attribution(self) -> None:
        sector_map = {"eur": "europe", "gbp": "europe", "jpy": "asia"}
        result = sector_attribution(_WEIGHTS, _RETURNS, sector_map)
        assert "europe" in result.columns
        assert "asia" in result.columns
        sector_sum = result["europe"] + result["asia"]
        np.testing.assert_allclose(sector_sum.values, result["total"].values, atol=1e-12)


class TestSingleSignalAttribution:
    def test_single_signal_attribution(self) -> None:
        result = signal_attribution(
            _WEIGHTS, _RETURNS,
            {"only": 1.0},
            {"only": pd.DataFrame()},
        )
        # All PnL in one signal, residual ≈ 0
        np.testing.assert_allclose(result["residual"].values, 0.0, atol=1e-12)
