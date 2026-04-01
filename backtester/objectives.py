"""Performance objectives for backtesting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class Objective(Protocol):
    name: str
    direction: str  # "maximize" or "minimize"

    def compute(self, portfolio_returns: pd.Series, **context: Any) -> float: ...


@dataclass
class SharpeObjective:
    name: str = "sharpe"
    direction: str = "maximize"
    annualization: float = 252.0

    def compute(self, portfolio_returns: pd.Series, **context: Any) -> float:
        if portfolio_returns.empty:
            return 0.0
        std = portfolio_returns.std()
        if std < 1e-12:
            return 0.0
        return float(portfolio_returns.mean() / std * np.sqrt(self.annualization))


@dataclass
class SortinoObjective:
    name: str = "sortino"
    direction: str = "maximize"
    annualization: float = 252.0

    def compute(self, portfolio_returns: pd.Series, **context: Any) -> float:
        if portfolio_returns.empty:
            return 0.0
        downside = portfolio_returns[portfolio_returns < 0]
        downside_std = downside.std() if len(downside) > 1 else 0.0
        if downside_std == 0:
            return 0.0
        return float(portfolio_returns.mean() / downside_std * np.sqrt(self.annualization))


@dataclass
class MaxDrawdownObjective:
    name: str = "max_drawdown"
    direction: str = "minimize"

    def compute(self, portfolio_returns: pd.Series, **context: Any) -> float:
        if portfolio_returns.empty:
            return 0.0
        cum = (1 + portfolio_returns).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return float(dd.min())  # negative value


@dataclass
class TurnoverObjective:
    name: str = "turnover"
    direction: str = "minimize"

    def compute(self, portfolio_returns: pd.Series, **context: Any) -> float:
        turnover_history = context.get("turnover_history")
        if turnover_history is not None:
            if turnover_history.empty:
                return 0.0
            return float(turnover_history.mean())
        weights_history = context.get("weights_history")
        if weights_history is None or weights_history.empty:
            return 0.0
        diffs = weights_history.diff().abs().sum(axis=1)
        return float(diffs.mean())


@dataclass
class CalmarObjective:
    name: str = "calmar"
    direction: str = "maximize"
    annualization: float = 252.0

    def compute(self, portfolio_returns: pd.Series, **context: Any) -> float:
        if portfolio_returns.empty:
            return 0.0
        ann_return = portfolio_returns.mean() * self.annualization
        cum = (1 + portfolio_returns).cumprod()
        peak = cum.cummax()
        max_dd = float(((cum - peak) / peak).min())
        if max_dd == 0:
            return 0.0
        return float(ann_return / abs(max_dd))


@dataclass
class CompositeObjective:
    """Weighted combination of multiple objectives."""

    name: str = "composite"
    direction: str = "maximize"
    objectives: dict[str, Objective] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)

    def compute(self, portfolio_returns: pd.Series, **context: Any) -> float:
        total = 0.0
        for obj_name, obj in self.objectives.items():
            w = self.weights.get(obj_name, 1.0)
            val = obj.compute(portfolio_returns, **context)
            # Flip sign for "minimize" objectives so composite maximizes
            if obj.direction == "minimize":
                val = -val
            total += w * val
        return total
