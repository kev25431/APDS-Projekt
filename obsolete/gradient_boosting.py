from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


MODEL_OUTPUT_DIR = Path("simulationresults/models")
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_COLUMNS: tuple[str, ...] = (
    "line",
    "direction_code",
    "station_code",
    "hour",
    "weekday",
    "month",
    "is_weekend",
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_24",
    "rolling_mean_4",
    "rolling_mean_24",
)


@dataclass
class TrainedGradientBoostingForecast:
    model: HistGradientBoostingRegressor
    feature_columns: list[str]
    station_lookup: dict[tuple[int, str, str], int]
    target_name: str = "target"

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        prepared = prepare_prediction_frame(
            frame=frame,
            feature_columns=self.feature_columns,
            station_lookup=self.station_lookup,
        )
        return self.model.predict(prepared[self.feature_columns])

    def predict_station(
        self,
        route,
        station: str,
        hour: int,
        weekday: int,
        month: int | None = None,
        lag_1: float = 0.0,
        lag_2: float = 0.0,
        lag_4: float = 0.0,
        lag_24: float = 0.0,
        rolling_mean_4: float = 0.0,
        rolling_mean_24: float = 0.0,
    ) -> float:
        key = (int(route.line), str(route.direction), str(station))
        station_code = self.station_lookup.get(key, -1)

        row = pd.DataFrame(
            [
                {
                    "line": int(route.line),
                    "direction_code": _direction_to_code(str(route.direction)),
                    "station_code": int(station_code),
                    "hour": int(hour),
                    "weekday": int(weekday),
                    "month": int(month if month is not None else 1),
                    "is_weekend": int(weekday >= 5),
                    "lag_1": float(lag_1),
                    "lag_2": float(lag_2),
                    "lag_4": float(lag_4),
                    "lag_24": float(lag_24),
                    "rolling_mean_4": float(rolling_mean_4),
                    "rolling_mean_24": float(rolling_mean_24),
                }
            ]
        )
        prediction = self.model.predict(row[self.feature_columns])[0]
        return float(max(0.0, prediction))

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "feature_columns": self.feature_columns,
                "station_lookup": self.station_lookup,
                "target_name": self.target_name,
            },
            path,
        )
        return path


def load_gradient_boosting_forecast(path: Path | str) -> TrainedGradientBoostingForecast:
    payload = joblib.load(path)
    return TrainedGradientBoostingForecast(
        model=payload["model"],
        feature_columns=list(payload["feature_columns"]),
        station_lookup=dict(payload["station_lookup"]),
        target_name=payload.get("target_name", "target"),
    )


def train_gradient_boosting_forecast(
    hourly_data: pd.DataFrame,
    save_path: Path | str | None = None,
) -> TrainedGradientBoostingForecast:
    prepared, station_lookup = build_gradient_boosting_training_data(hourly_data)

    if prepared.empty:
        raise ValueError("Keine Trainingsdaten fuer Gradient Boosting vorhanden.")

    X = prepared[list(FEATURE_COLUMNS)].copy()
    y = prepared["target"].astype(float).to_numpy()

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )
    model.fit(X, y)

    trained = TrainedGradientBoostingForecast(
        model=model,
        feature_columns=list(FEATURE_COLUMNS),
        station_lookup=station_lookup,
        target_name="target",
    )

    if save_path is not None:
        trained.save(save_path)

    return trained


def evaluate_gradient_boosting_forecast(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> tuple[dict[str, float], TrainedGradientBoostingForecast]:
    trained = train_gradient_boosting_forecast(train_frame)

    test_prepared, _ = build_gradient_boosting_training_data(
        hourly_data=test_frame,
        station_lookup=trained.station_lookup,
        drop_unknown_stations=False,
    )
    test_prepared = test_prepared.dropna(subset=["target"]).copy()

    if test_prepared.empty:
        raise ValueError("Keine Testdaten fuer die Bewertung des Gradient-Boosting-Modells vorhanden.")

    predictions = trained.model.predict(test_prepared[list(FEATURE_COLUMNS)])
    actual = test_prepared["target"].to_numpy(dtype=float)

    metrics = {
        "forecastmodel": "gradientboosting",
        "ntrain": int(len(train_frame)),
        "ntest": int(len(test_prepared)),
        "mae": round(float(mean_absolute_error(actual, predictions)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, predictions))), 4),
        "r2": round(float(r2_score(actual, predictions)), 4),
    }
    return metrics, trained


def build_gradient_boosting_training_data(
    hourly_data: pd.DataFrame,
    station_lookup: dict[tuple[int, str, str], int] | None = None,
    drop_unknown_stations: bool = True,
) -> tuple[pd.DataFrame, dict[tuple[int, str, str], int]]:
    hourly_data = hourly_data.rename(
        columns={"service_date": "servicedate", "station_short": "stationshort"}
    )
    required = {"line", "direction", "stationshort", "hour", "servicedate", "target"}
    missing = required - set(hourly_data.columns)
    if missing:
        raise ValueError(f"Fehlende Trainingsspalten fuer Gradient Boosting: {sorted(missing)}")

    frame = hourly_data.copy()
    frame["servicedate"] = pd.to_datetime(frame["servicedate"])
    frame["line"] = pd.to_numeric(frame["line"], errors="coerce")
    frame["hour"] = pd.to_numeric(frame["hour"], errors="coerce")
    frame["target"] = pd.to_numeric(frame["target"], errors="coerce")
    frame["direction"] = frame["direction"].astype(str)
    frame["stationshort"] = frame["stationshort"].astype(str)

    frame = frame.dropna(subset=["line", "hour", "target"]).copy()
    frame["line"] = frame["line"].astype(int)
    frame["hour"] = frame["hour"].astype(int)

    frame = frame.sort_values(["line", "direction", "stationshort", "servicedate", "hour"]).reset_index(drop=True)
    frame["timestamp"] = frame["servicedate"] + pd.to_timedelta(frame["hour"], unit="h")

    key_tuples = list(
        frame[["line", "direction", "stationshort"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    if station_lookup is None:
        station_lookup = {key: idx for idx, key in enumerate(sorted(key_tuples))}
    frame["station_code"] = frame.apply(
        lambda row: station_lookup.get((int(row["line"]), str(row["direction"]), str(row["stationshort"])), -1),
        axis=1,
    )

    if drop_unknown_stations:
        frame = frame.loc[frame["station_code"] >= 0].copy()

    group_keys = ["line", "direction", "stationshort"]

    frame["weekday"] = frame["servicedate"].dt.weekday
    frame["month"] = frame["servicedate"].dt.month
    frame["is_weekend"] = (frame["weekday"] >= 5).astype(int)
    frame["direction_code"] = frame["direction"].map(_direction_to_code)

    frame["lag_1"] = frame.groupby(group_keys)["target"].shift(1)
    frame["lag_2"] = frame.groupby(group_keys)["target"].shift(2)
    frame["lag_4"] = frame.groupby(group_keys)["target"].shift(4)
    frame["lag_24"] = frame.groupby(group_keys)["target"].shift(24)

    shifted_target = frame.groupby(group_keys)["target"].shift(1)
    frame["rolling_mean_4"] = frame.groupby(group_keys)["lag_1"].transform(
        lambda values: values.rolling(window=4, min_periods=1).mean()
    )
    frame["rolling_mean_24"] = frame.groupby(group_keys)["lag_1"].transform(
        lambda values: values.rolling(window=24, min_periods=1).mean()
    )

    frame = frame.dropna(subset=["lag_1", "lag_2", "lag_4", "lag_24"]).copy()

    for column in FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    return frame.reset_index(drop=True), station_lookup


def prepare_prediction_frame(
    frame: pd.DataFrame,
    feature_columns: list[str],
    station_lookup: dict[tuple[int, str, str], int],
) -> pd.DataFrame:
    prepared = frame.copy()

    if "direction_code" not in prepared.columns and "direction" in prepared.columns:
        prepared["direction_code"] = prepared["direction"].astype(str).map(_direction_to_code)

    if "station_code" not in prepared.columns:
        if {"line", "direction", "stationshort"}.issubset(prepared.columns):
            prepared["station_code"] = prepared.apply(
                lambda row: station_lookup.get(
                    (int(row["line"]), str(row["direction"]), str(row["stationshort"])),
                    -1,
                ),
                axis=1,
            )
        else:
            prepared["station_code"] = -1

    if "is_weekend" not in prepared.columns and "weekday" in prepared.columns:
        prepared["is_weekend"] = (pd.to_numeric(prepared["weekday"], errors="coerce").fillna(0) >= 5).astype(int)

    if "month" not in prepared.columns:
        prepared["month"] = 1

    for column in feature_columns:
        if column not in prepared.columns:
            prepared[column] = 0.0
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce").fillna(0.0)

    return prepared


def train_test_split_by_date(
    hourly_data: pd.DataFrame,
    split_date: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = hourly_data.copy()
    frame["servicedate"] = pd.to_datetime(frame["servicedate"])
    split_date = pd.to_datetime(split_date)

    train = frame.loc[frame["servicedate"] < split_date].copy()
    test = frame.loc[frame["servicedate"] >= split_date].copy()

    return train, test


def fit_and_save_gradient_boosting(
    hourly_data: pd.DataFrame,
    split_date: str | pd.Timestamp,
    output_dir: Path | str = MODEL_OUTPUT_DIR,
    model_name: str = "gradient_boosting_forecast.joblib",
) -> tuple[dict[str, float], TrainedGradientBoostingForecast, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_frame, test_frame = train_test_split_by_date(hourly_data, split_date=split_date)
    metrics, trained = evaluate_gradient_boosting_forecast(train_frame, test_frame)

    save_path = output_dir / model_name
    trained.save(save_path)
    return metrics, trained, save_path


def _direction_to_code(direction: str) -> int:
    value = str(direction).strip().lower()
    mapping = {
        "hin": 0,
        "rueck": 1,
        "rück": 1,
        "outbound": 0,
        "inbound": 1,
    }
    return mapping.get(value, abs(hash(value)) % 1000)