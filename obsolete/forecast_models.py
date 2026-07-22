from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


RANDOM_FOREST_FEATURE_COLUMNS = [
    "line_code",
    "direction_code",
    "station_code",
    "hour",
    "weekday",
    "station_mean",
    "station_hourly_mean",
]


@dataclass
class TrainedForecastModel:
    """Stores a trained tabular forecast model and feature encodings."""

    name: str
    model: RandomForestRegressor
    feature_columns: list[str]
    line_encoder: dict[int, int]
    direction_encoder: dict[str, int]
    station_encoder: dict[str, int]
    station_mean: dict[str, float]
    station_hourly_mean: dict[tuple[str, int], float]

    def _feature_values(
        self,
        line: int,
        direction: str,
        station: str,
        hour: int,
        weekday: int,
    ) -> dict[str, float]:
        return {
            "line_code": float(self.line_encoder.get(line, -1)),
            "direction_code": float(self.direction_encoder.get(direction, -1)),
            "station_code": float(self.station_encoder.get(station, -1)),
            "hour": float(hour),
            "weekday": float(weekday),
            "station_mean": float(self.station_mean.get(station, 0.0)),
            "station_hourly_mean": float(self.station_hourly_mean.get((station, hour), 0.0)),
        }

    def _encode_features(self, line: int, direction: str, station: str, hour: int, weekday: int) -> np.ndarray:
        feature_values = self._feature_values(line, direction, station, hour, weekday)
        values = [feature_values[column] for column in self.feature_columns]
        return np.array(values, dtype=float).reshape(1, -1)

    def predict_station(self, route, station: str, hour: int, weekday: int) -> float:
        features = self._encode_features(
            route.line,
            route.direction,
            station,
            hour,
            weekday,
        )
        return float(self.model.predict(features)[0])

    def predict_row(self, row: pd.Series) -> float:
        features = self._encode_features(
            int(row["line"]),
            str(row["direction"]),
            str(row["station_short"]),
            int(row["hour"]),
            int(row["weekday"]),
        )
        return float(self.model.predict(features)[0])

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        rows = []
        for row in frame.itertuples(index=False):
            feature_values = self._feature_values(
                int(row.line),
                str(row.direction),
                str(row.station_short),
                int(row.hour),
                int(row.weekday),
            )
            rows.append([feature_values[column] for column in self.feature_columns])
        return np.clip(self.model.predict(np.asarray(rows, dtype=float)), 0.0, None)

    def predict_route_demand(self, route, hour: int) -> float:
        return sum(self.predict_station(route, station, hour, 0) for station in route.stations)


def train_random_forest_forecast(training_data: pd.DataFrame) -> TrainedForecastModel:
    line_encoder = {line: idx for idx, line in enumerate(sorted(training_data["line"].unique()))}
    direction_encoder = {direction: idx for idx, direction in enumerate(sorted(training_data["direction"].unique()))}
    station_encoder = {station: idx for idx, station in enumerate(sorted(training_data["station_short"].unique()))}

    station_mean = training_data.groupby("station_short")["target"].mean().to_dict()
    station_hourly_mean = training_data.groupby(["station_short", "hour"])["target"].mean().to_dict()

    rows = []
    for _, row in training_data.iterrows():
        rows.append(
            [
                line_encoder.get(row["line"], -1),
                direction_encoder.get(row["direction"], -1),
                station_encoder.get(row["station_short"], -1),
                row["hour"],
                row["weekday"],
                station_mean.get(row["station_short"], 0.0),
                station_hourly_mean.get((row["station_short"], row["hour"]), 0.0),
            ]
        )

    rf_model = RandomForestRegressor(n_estimators=80, max_depth=12, random_state=42, n_jobs=-1)
    rf_model.fit(np.asarray(rows, dtype=float), training_data["target"].to_numpy(dtype=float))

    return TrainedForecastModel(
        name="random_forest",
        model=rf_model,
        feature_columns=list(RANDOM_FOREST_FEATURE_COLUMNS),
        line_encoder=line_encoder,
        direction_encoder=direction_encoder,
        station_encoder=station_encoder,
        station_mean=station_mean,
        station_hourly_mean=station_hourly_mean,
    )
