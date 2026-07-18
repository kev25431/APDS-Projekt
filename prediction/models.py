from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import normalize_station_name


class GroupFallbackRegressor:
    """Fallback model used when scikit-learn is not installed."""

    def __init__(self) -> None:
        self.by_station_hour = pd.DataFrame()
        self.by_station = pd.DataFrame()
        self.global_mean = np.array([0.0, 0.0], dtype=float)

    def fit(self, frame: pd.DataFrame) -> None:
        self.by_station_hour = (
            frame.groupby(["line", "station_key", "hour", "weekday"], as_index=False)
            .agg(boardings=("boardings", "mean"), exiting=("exiting", "mean"))
        )
        self.by_station = (
            frame.groupby(["line", "station_key"], as_index=False)
            .agg(boardings=("boardings", "mean"), exiting=("exiting", "mean"))
        )
        self.global_mean = frame[["boardings", "exiting"]].mean().fillna(0).to_numpy(dtype=float)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.empty((0, 2), dtype=float)
        lookup = self.by_station_hour.rename(columns={"boardings": "pred_boardings", "exiting": "pred_exiting"})
        merged = frame[["line", "station_key", "hour", "weekday"]].merge(
            lookup,
            on=["line", "station_key", "hour", "weekday"],
            how="left",
        )
        missing = merged["pred_boardings"].isna()
        if missing.any():
            station_lookup = self.by_station.rename(columns={"boardings": "pred_boardings", "exiting": "pred_exiting"})
            fallback = frame.loc[missing, ["line", "station_key"]].merge(
                station_lookup,
                on=["line", "station_key"],
                how="left",
            )
            merged.loc[missing, "pred_boardings"] = fallback["pred_boardings"].fillna(self.global_mean[0]).to_numpy()
            merged.loc[missing, "pred_exiting"] = fallback["pred_exiting"].fillna(self.global_mean[1]).to_numpy()
        merged["pred_boardings"] = merged["pred_boardings"].fillna(self.global_mean[0])
        merged["pred_exiting"] = merged["pred_exiting"].fillna(self.global_mean[1])
        return merged[["pred_boardings", "pred_exiting"]].to_numpy(dtype=float)


class GraphMessagePassingRegressor:
    """Graph-light demand model: station profile plus neighbor message passing."""

    def __init__(self) -> None:
        self.neighbors: dict[tuple[int, str], set[str]] = {}
        self.profile_exact = pd.DataFrame()
        self.profile_hour = pd.DataFrame()
        self.profile_station = pd.DataFrame()
        self.global_mean = np.array([0.0, 0.0], dtype=float)

    def fit(self, feature_frame: pd.DataFrame, raw_frame: pd.DataFrame) -> None:
        self.profile_exact = (
            feature_frame.groupby(["line", "station_key", "hour", "weekday"], as_index=False)
            .agg(boardings=("boardings", "mean"), exiting=("exiting", "mean"))
        )
        self.profile_hour = (
            feature_frame.groupby(["line", "station_key", "hour"], as_index=False)
            .agg(boardings=("boardings", "mean"), exiting=("exiting", "mean"))
        )
        self.profile_station = (
            feature_frame.groupby(["line", "station_key"], as_index=False)
            .agg(boardings=("boardings", "mean"), exiting=("exiting", "mean"))
        )
        self.global_mean = feature_frame[["boardings", "exiting"]].mean().fillna(0).to_numpy(dtype=float)

        raw = raw_frame.sort_values(["line", "journey", "departure_plan_station"]).copy()
        raw["station_key"] = raw["station"].map(normalize_station_name)
        for (line, _journey), group in raw.groupby(["line", "journey"], sort=False):
            keys = [key for key in group["station_key"].tolist() if key]
            for left, right in zip(keys, keys[1:]):
                self.neighbors.setdefault((int(line), left), set()).add(right)
                self.neighbors.setdefault((int(line), right), set()).add(left)

    def _lookup(self, line: int, station_key: str, hour: int, weekday: int) -> np.ndarray:
        row = self.profile_exact[
            (self.profile_exact["line"] == line)
            & (self.profile_exact["station_key"] == station_key)
            & (self.profile_exact["hour"] == hour)
            & (self.profile_exact["weekday"] == weekday)
        ]
        if not row.empty:
            return row.iloc[0][["boardings", "exiting"]].to_numpy(dtype=float)

        row = self.profile_hour[
            (self.profile_hour["line"] == line)
            & (self.profile_hour["station_key"] == station_key)
            & (self.profile_hour["hour"] == hour)
        ]
        if not row.empty:
            return row.iloc[0][["boardings", "exiting"]].to_numpy(dtype=float)

        row = self.profile_station[
            (self.profile_station["line"] == line)
            & (self.profile_station["station_key"] == station_key)
        ]
        if not row.empty:
            return row.iloc[0][["boardings", "exiting"]].to_numpy(dtype=float)
        return self.global_mean.copy()

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        rows: list[np.ndarray] = []
        for _, row in frame.iterrows():
            line = int(row["line"])
            station_key = str(row["station_key"])
            hour = int(row["hour"])
            weekday = int(row["weekday"])
            own = self._lookup(line, station_key, hour, weekday)
            neighbor_values = [
                self._lookup(line, neighbor, hour, weekday)
                for neighbor in sorted(self.neighbors.get((line, station_key), set()))
            ]
            if neighbor_values:
                rows.append(0.72 * own + 0.28 * np.vstack(neighbor_values).mean(axis=0))
            else:
                rows.append(own)
        return np.vstack(rows) if rows else np.empty((0, 2), dtype=float)
