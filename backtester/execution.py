"""Provider-agnostic execution-clock data abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import pandas as pd

TargetKind = Literal["weights", "exposures"]

_BAR_REQUIRED_COLUMNS = ("ts", "asset", "close")
_BAR_OPTIONAL_COLUMNS = ("open", "high", "low", "volume")
_BAR_STANDARD_COLUMNS = set(_BAR_REQUIRED_COLUMNS) | set(_BAR_OPTIONAL_COLUMNS)


def _normalize_asset_ids(values: pd.Series | pd.Index, *, name: str) -> pd.Index:
    if values.isna().any():
        raise ValueError(f"{name} asset identifiers must not contain null values.")
    return pd.Index(values.map(str), dtype="object")


def _normalize_timestamp_series(values: pd.Series, *, name: str) -> pd.Series:
    ts = pd.to_datetime(values, errors="raise")
    if getattr(ts.dt, "tz", None) is None:
        if len(ts) == 0:
            return ts.dt.tz_localize("UTC")
        raise ValueError(f"{name} timestamps must be timezone-aware and explicitly in UTC.")
    return ts.dt.tz_convert("UTC")


def _normalize_timestamp_index(values: pd.Index, *, name: str) -> pd.DatetimeIndex:
    if isinstance(values, pd.DatetimeIndex):
        ts = values
    else:
        ts = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    if ts.tz is None:
        if len(ts) == 0:
            return ts.tz_localize("UTC")
        raise ValueError(f"{name} timestamps must be timezone-aware and explicitly in UTC.")
    return ts.tz_convert("UTC")


def _normalize_boundary(value: pd.Timestamp | str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise ValueError("Boundary timestamps must be timezone-aware and explicitly in UTC.")
    return ts.tz_convert("UTC")


def _coerce_numeric(series: pd.Series, *, name: str) -> pd.Series:
    try:
        return pd.to_numeric(series)
    except Exception as exc:  # pragma: no cover - defensive wrapper
        raise ValueError(f"{name} values must be numeric.") from exc


@dataclass(frozen=True)
class ExecutionBarFrame:
    """Canonical execution-bar data on the execution clock.

    The canonical representation is a long-form DataFrame with columns:

    - ``ts``: timezone-aware UTC execution timestamp
    - ``asset``: string asset identifier
    - ``close``: executable price field required by the execution clock

    Optional standardized fields are ``open``, ``high``, ``low``, and
    ``volume``. Additional metadata columns are preserved.
    """

    data: pd.DataFrame

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", self._normalize(self.data))

    @classmethod
    def from_frame(
        cls,
        data: pd.DataFrame,
        *,
        timestamp_col: str,
        asset_col: str,
        close_col: str,
        open_col: str | None = None,
        high_col: str | None = None,
        low_col: str | None = None,
        volume_col: str | None = None,
    ) -> "ExecutionBarFrame":
        """Build execution bars from arbitrary provider column names."""

        required = [timestamp_col, asset_col, close_col]
        missing = [column for column in required if column not in data.columns]
        if missing:
            raise ValueError(f"Missing required execution-bar columns: {missing}")

        rename_map = {
            timestamp_col: "ts",
            asset_col: "asset",
            close_col: "close",
        }
        optional_map = {
            open_col: "open",
            high_col: "high",
            low_col: "low",
            volume_col: "volume",
        }
        for source, target in optional_map.items():
            if source is not None:
                if source not in data.columns:
                    raise ValueError(f"Optional execution-bar column '{source}' was not found.")
                rename_map[source] = target

        canonical = data.rename(columns=rename_map).copy()
        keep = [
            column
            for column in canonical.columns
            if column not in rename_map or column in rename_map.values()
        ]
        return cls(canonical[keep])

    @staticmethod
    def _normalize(data: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in _BAR_REQUIRED_COLUMNS if column not in data.columns]
        if missing:
            raise ValueError(
                "Execution bars must include canonical columns "
                f"{list(_BAR_REQUIRED_COLUMNS)}; missing {missing}."
            )

        normalized = data.copy()
        normalized["ts"] = _normalize_timestamp_series(normalized["ts"], name="Execution-bar")
        normalized["asset"] = _normalize_asset_ids(normalized["asset"], name="Execution-bar")
        normalized["close"] = _coerce_numeric(normalized["close"], name="Close price")
        if normalized["close"].isna().any():
            raise ValueError("Execution-bar close prices must not contain missing values.")

        for column in _BAR_OPTIONAL_COLUMNS:
            if column in normalized.columns:
                normalized[column] = _coerce_numeric(normalized[column], name=column.capitalize())

        normalized = normalized.sort_values(["ts", "asset"], kind="stable").reset_index(drop=True)
        if normalized.duplicated(subset=["ts", "asset"]).any():
            raise ValueError("Execution bars must be unique on ('ts', 'asset').")

        return normalized

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        """Sorted unique execution timestamps in UTC."""

        return pd.DatetimeIndex(self.data["ts"].drop_duplicates())

    @property
    def asset_ids(self) -> list[str]:
        """Sorted unique asset identifiers present in the frame."""

        return sorted(self.data["asset"].unique().tolist())

    @property
    def value_columns(self) -> list[str]:
        """Available standardized and metadata value fields."""

        return [column for column in self.data.columns if column not in {"ts", "asset"}]

    @property
    def metadata_columns(self) -> list[str]:
        """Non-standard metadata columns preserved from the input."""

        return [column for column in self.value_columns if column not in _BAR_STANDARD_COLUMNS]

    def to_matrix(self, field: str = "close") -> pd.DataFrame:
        """Pivot a value field to ``ts x asset`` matrix form."""

        if field not in self.data.columns:
            raise KeyError(f"Execution-bar field '{field}' is not available.")
        matrix = self.data.pivot(index="ts", columns="asset", values=field).sort_index()
        matrix.columns.name = None
        return matrix

    def slice(
        self,
        *,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        assets: Sequence[str] | None = None,
    ) -> "ExecutionBarFrame":
        """Return a filtered view of the execution-bar frame."""

        start_ts = _normalize_boundary(start)
        end_ts = _normalize_boundary(end)
        filtered = self.data
        if start_ts is not None:
            filtered = filtered[filtered["ts"] >= start_ts]
        if end_ts is not None:
            filtered = filtered[filtered["ts"] <= end_ts]
        if assets is not None:
            asset_ids = {str(asset) for asset in assets}
            filtered = filtered[filtered["asset"].isin(asset_ids)]
        return ExecutionBarFrame(filtered.reset_index(drop=True))


@dataclass(frozen=True)
class TargetSchedule:
    """Sparse decision-time targets aligned to a lower-frequency decision clock.

    The canonical representation is a wide DataFrame:

    - index: ``decision_ts`` as timezone-aware UTC timestamps
    - columns: string asset identifiers
    - values: numeric target magnitudes

    Missing decision timestamps imply carry-forward behavior at the strategy
    level: the most recent target remains in force until the next decision
    timestamp. Within a decision timestamp, partial snapshots are rejected to
    avoid ambiguity.

    ``target_kind="weights"`` means values are portfolio weights and residual
    ``1.0 - sum(weights)`` is interpreted as implicit cash by later execution
    layers. ``target_kind="exposures"`` means values are caller-defined
    exposure levels and no implicit cash inference should be assumed.
    """

    data: pd.DataFrame
    target_kind: TargetKind = "weights"

    def __post_init__(self) -> None:
        if self.target_kind not in ("weights", "exposures"):
            raise ValueError("target_kind must be either 'weights' or 'exposures'.")
        object.__setattr__(self, "data", self._normalize(self.data))

    @classmethod
    def from_frame(
        cls,
        data: pd.DataFrame,
        *,
        timestamp_col: str,
        asset_col: str,
        target_col: str,
        target_kind: TargetKind = "weights",
    ) -> "TargetSchedule":
        """Build a target schedule from long-form decision data."""

        required = [timestamp_col, asset_col, target_col]
        missing = [column for column in required if column not in data.columns]
        if missing:
            raise ValueError(f"Missing required target-schedule columns: {missing}")

        schedule = data[[timestamp_col, asset_col, target_col]].copy()
        schedule = schedule.rename(
            columns={
                timestamp_col: "decision_ts",
                asset_col: "asset",
                target_col: "target",
            }
        )
        schedule["decision_ts"] = _normalize_timestamp_series(
            schedule["decision_ts"], name="Target-schedule"
        )
        schedule["asset"] = _normalize_asset_ids(schedule["asset"], name="Target-schedule")
        schedule["target"] = _coerce_numeric(schedule["target"], name="Target")
        if schedule["target"].isna().any():
            raise ValueError("Target values must not contain missing values.")
        if schedule.duplicated(subset=["decision_ts", "asset"]).any():
            raise ValueError("Target schedule rows must be unique on ('decision_ts', 'asset').")

        wide = schedule.pivot(index="decision_ts", columns="asset", values="target").sort_index()
        wide.columns.name = None
        if wide.isna().any().any():
            raise ValueError(
                "Each decision timestamp must define a complete target snapshot for every "
                "asset present in the schedule."
            )
        return cls(wide, target_kind=target_kind)

    @staticmethod
    def _normalize(data: pd.DataFrame) -> pd.DataFrame:
        normalized = data.copy()
        normalized.index = _normalize_timestamp_index(normalized.index, name="Target-schedule")
        normalized.index.name = normalized.index.name or "decision_ts"
        if normalized.index.has_duplicates:
            raise ValueError("Target schedule decision timestamps must be unique.")

        normalized = normalized.sort_index(kind="stable")
        normalized.columns = _normalize_asset_ids(
            pd.Index(normalized.columns), name="Target-schedule"
        )
        if normalized.columns.has_duplicates:
            raise ValueError("Target schedule asset identifiers must be unique.")

        for column in normalized.columns:
            normalized[column] = _coerce_numeric(normalized[column], name=f"Target '{column}'")
        if normalized.isna().any().any():
            raise ValueError(
                "Target schedules must not contain missing values within a decision snapshot."
            )

        return normalized

    @property
    def decision_times(self) -> pd.DatetimeIndex:
        """Sorted decision timestamps in UTC."""

        return pd.DatetimeIndex(self.data.index)

    @property
    def asset_ids(self) -> list[str]:
        """Asset identifiers present in the schedule."""

        return self.data.columns.tolist()

    def slice(
        self,
        *,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
    ) -> "TargetSchedule":
        """Return a filtered view of the target schedule."""

        start_ts = _normalize_boundary(start)
        end_ts = _normalize_boundary(end)
        filtered = self.data
        if start_ts is not None:
            filtered = filtered[filtered.index >= start_ts]
        if end_ts is not None:
            filtered = filtered[filtered.index <= end_ts]
        return TargetSchedule(filtered.copy(), target_kind=self.target_kind)

    def validate_for(self, bars: ExecutionBarFrame) -> None:
        """Validate that the schedule does not reference assets missing from execution bars."""

        bar_assets = set(bars.asset_ids)
        missing_assets = sorted(set(self.asset_ids) - bar_assets)
        if missing_assets:
            raise ValueError(
                f"Target schedule contains assets not present in execution bars: {missing_assets}"
            )

    def align_to_assets(
        self,
        asset_ids: Sequence[str],
        *,
        fill_value: float = 0.0,
    ) -> "TargetSchedule":
        """Return a schedule reindexed to an explicit asset universe.

        Missing asset columns are added with ``fill_value``. Existing schedule
        assets must all be present in ``asset_ids``; this method is a deliberate
        normalization step rather than an implicit lossy projection.
        """

        normalized_assets = [str(asset) for asset in asset_ids]
        missing_assets = sorted(set(self.asset_ids) - set(normalized_assets))
        if missing_assets:
            raise ValueError(
                "Cannot align target schedule to an asset universe that omits "
                f"scheduled assets: {missing_assets}"
            )
        aligned = self.data.reindex(columns=normalized_assets, fill_value=fill_value)
        return TargetSchedule(aligned, target_kind=self.target_kind)
