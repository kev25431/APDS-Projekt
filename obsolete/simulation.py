from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import math
import os
import re
from collections import deque, defaultdict

os.environ.setdefault("ARROW_USER_SIMD_LEVEL", "NONE")
MATPLOTLIB_CACHE_DIR = Path("simulation_results") / "matplotlib_cache"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np
import pandas as pd
import simpy

from forecast_models import TrainedForecastModel, train_random_forest_forecast
from gnn_forecast import train_gnn_forecast
from gradient_boosting import train_gradient_boosting_forecast

# Import data.
DATA_DIR = Path("Data_project_app/downloads/wvv-pjs-2026/model_ready_2025")
OUTPUT_DIR = Path("simulation_results")

PROJECT_LINES = [10, 20, 27, 33, 55, 214]
LINE_LABELS = {
    10: "Hubland / volatil",
    20: "Stabile Stadtachse",
    27: "Pendlerfokus",
    33: "Grundversorgung",
    55: "Eventgetrieben",
    214: "Uni Direkt",
}

SIMULATION_START_HOUR = 5
SIMULATION_TIME = 1320
DEMAND_INTERVAL = 15
BUS_CAPACITY = 80
VEHICLE_TYPE_CAPACITY = {
    "GKOM": 149,
    "SKOM": 102,
}
FLEET_SIZE = 22
TARGET_LOAD_FACTOR = 0.75
FORECAST_ALPHA = 0.3
FORECAST_WINDOW_SIZE = 4
FORECAST_MODELS = [
    "historical_mean",
    "moving_average",
    "exponential_smoothing",
    "random_forest",
    "gradient_boosting",
    "gnn",
]

TRAIN_TEST_SPLIT_DATE = "2025-10-01"
SIMULATION_START_TIMESTAMP = pd.Timestamp(TRAIN_TEST_SPLIT_DATE) + pd.Timedelta(hours=SIMULATION_START_HOUR)

MIN_HEADWAY = 5
MAX_HEADWAY = 35
DEFAULT_HEADWAY = 20
DEFAULT_TRAVEL_TIME = 2.0
MIN_DWELL_TIME = 0.4
DWELL_TIME_PER_PASSENGER = 0.04
MAX_QUALITY_FACTOR = 150


@dataclass(frozen=True)
class LineRoute:
    """Immutable route definition for one bus line and direction."""

    line: int
    direction: str
    stations: tuple[str, ...]
    station_names: dict[str, str]
    travel_times: tuple[float, ...]
    static_headway: float
    vehicle_type: str
    capacity: int


class EventLogger:
    """Collects simulation events and converts them into an analysis-ready DataFrame."""

    def __init__(self, verbose: bool = False):
        self.events: list[dict] = []
        self.verbose = verbose

    def log(self, **kwargs):
        if self.verbose:
            print("  ".join(f"{key}={value}" for key, value in kwargs.items()))
        self.events.append(kwargs)

    def get_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.events)


class PriorityFleet:
    """Shared limited bus fleet modeled as a SimPy priority resource."""

    def __init__(self, env: simpy.Environment, size: int):
        self.env = env
        self.size = size
        self.resource = simpy.PriorityResource(env, capacity=size)

    def request(self, priority: float = 0):
        return self.resource.request(priority=priority)

    def release(self, request):
        return self.resource.release(request)


class HistoricalNetworkData:
    """Loads historical WVV data and derives routes, travel times, and demand profiles."""

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        lines: list[int] | None = None,
        training_year: int = 2025,
    ):
        self.data_dir = data_dir
        self.lines = lines or PROJECT_LINES
        self.training_year = training_year
        self.raw = self._load_data()
        self.routes = self._build_routes()
        self.boarding_profile = self._build_hourly_profile("passenger_boarding_measured")
        self.exiting_profile = self._build_visit_profile("passenger_exiting_measured")

    def _load_data(self) -> pd.DataFrame:
        frames = []

        for line in self.lines:
            path = self._select_line_file(line)
            if path is None:
                print(f"Warnung: Keine Daten fuer Linie {line} gefunden.")
                continue

            frames.append(self._read_line_file(path))

        if not frames:
            raise FileNotFoundError("Keine Parquet-Dateien fuer die ausgewaehlten Linien gefunden.")

        df = pd.concat(frames, ignore_index=True)
        df["departure_plan_station"] = pd.to_datetime(df["departure_plan_station"])
        df["service_date"] = df["departure_plan_station"].dt.date
        df["hour"] = df["departure_plan_station"].dt.hour
        df["station_short"] = df["station_short"].astype(str)
        df["direction"] = df["direction"].astype(str)

        if "quality_factor" in df.columns:
            df = df[df["quality_factor"].isna() | (df["quality_factor"] <= MAX_QUALITY_FACTOR)]

        for column in ["passenger_boarding_measured", "passenger_exiting_measured"]:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).clip(lower=0)

        df["vehicle_type"] = df["vehicle_type"].fillna("unknown").astype(str)
        if "source_model_ready" not in df.columns:
            df["source_model_ready"] = False
        for column in ["occupancy_departure", "vehicle_utilization"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        return df

    def _read_line_file(self, path: Path) -> pd.DataFrame:
        frame = pd.read_parquet(path)
        if "departure_plan_station" in frame.columns:
            return self._normalize_raw_frame(frame)
        return self._normalize_model_ready_frame(frame)

    def _normalize_raw_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        expected_columns = [
            "departure_plan_station",
            "line",
            "direction",
            "route",
            "journey",
            "quality_factor",
            "stop_sequence",
            "station_short",
            "station",
            "vehicle_type",
            "occupancy_departure",
            "vehicle_utilization",
            "passenger_boarding_measured",
            "passenger_exiting_measured",
        ]
        normalized = frame[[column for column in expected_columns if column in frame.columns]].copy()
        normalized["source_model_ready"] = False
        return normalized

    def _normalize_model_ready_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        # The model-ready files are feature tables, so map them into the simulation schema.
        df = frame.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"].dt.year == self.training_year]
        df["departure_plan_station"] = df["date"] + pd.to_timedelta(df["departure_minute_of_day"], unit="m")
        df["journey"] = (
            df["date"].dt.strftime("%Y-%m-%d")
            + "_"
            + df["line"].astype(str)
            + "_"
            + df["direction"].astype(str)
            + "_"
            + df["route"].astype(str)
        )
        df["station_short"] = df["station_key"].astype(str)
        df["passenger_boarding_measured"] = df["passenger_boarding"]
        df["passenger_exiting_measured"] = df["passenger_exiting"]
        df["quality_factor"] = np.nan
        df["occupancy_departure"] = np.nan
        df["vehicle_utilization"] = np.nan
        df["source_model_ready"] = True

        return df[
            [
                "departure_plan_station",
                "line",
                "direction",
                "route",
                "journey",
                "quality_factor",
                "stop_sequence",
                "station_short",
                "station",
                "vehicle_type",
                "occupancy_departure",
                "vehicle_utilization",
                "passenger_boarding_measured",
                "passenger_exiting_measured",
                "source_model_ready",
            ]
        ]

    def _select_line_file(self, line: int) -> Path | None:
        pattern = re.compile(
            rf"data_(\d{{4}})-\d{{2}}-\d{{2}}_(\d{{4}})-\d{{2}}-\d{{2}}_line_{line}(?:_model_ready_2025)?\.parquet$"
        )
        candidates = []

        for path in self.data_dir.glob(f"*_line_{line}*.parquet"):
            match = pattern.match(path.name)
            if not match:
                continue
            start_year = int(match.group(1))
            end_year = int(match.group(2))
            candidates.append((start_year, end_year, path))

        if not candidates:
            return None

        same_year = [item for item in candidates if item[0] == self.training_year]
        if same_year:
            return sorted(same_year, key=lambda item: item[1])[-1][2]

        return sorted(candidates, key=lambda item: (item[0], item[1]))[-1][2]

    def _build_routes(self) -> dict[tuple[int, str], LineRoute]:
        routes = {}
        station_names = (
            self.raw.sort_values("departure_plan_station")
            .drop_duplicates("station_short")
            .set_index("station_short")["station"]
            .to_dict()
        )

        for (line, direction), group in self.raw.groupby(["line", "direction"]):
            if group["source_model_ready"].all():
                best_sequence = self._model_ready_station_sequence(group)
            else:
                journey_sequences = []
                for _, journey in group.groupby("journey"):
                    ordered = journey.sort_values("stop_sequence")
                    stations = tuple(ordered["station_short"].dropna().astype(str))
                    if len(stations) >= 2:
                        journey_sequences.append(stations)

                best_sequence = pd.Series(journey_sequences).value_counts().idxmax() if journey_sequences else ()

            if len(best_sequence) < 2:
                continue

            travel_times = (
                tuple([DEFAULT_TRAVEL_TIME] * (len(best_sequence) - 1))
                if group["source_model_ready"].all()
                else self._median_travel_times(group, best_sequence)
            )
            static_headway = self._median_headway(group)
            vehicle_type = self._most_common_vehicle_type(group)
            inferred_capacity = self._infer_capacity(group)
            capacity = VEHICLE_TYPE_CAPACITY.get(vehicle_type, inferred_capacity)

            routes[(int(line), str(direction))] = LineRoute(
                line=int(line),
                direction=str(direction),
                stations=best_sequence,
                station_names={
                    station: station_names.get(station, station)
                    for station in best_sequence
                },
                travel_times=travel_times,
                static_headway=static_headway,
                vehicle_type=vehicle_type,
                capacity=capacity,
            )

        return routes

    def _model_ready_station_sequence(self, group: pd.DataFrame) -> tuple[str, ...]:
        sequence = []
        for _, stop_group in group.groupby("stop_sequence"):
            stations = stop_group["station_short"].dropna().astype(str)
            if stations.empty:
                continue
            sequence.append(str(stations.mode().iloc[0]))
        return tuple(dict.fromkeys(sequence))

    def _median_travel_times(self, group: pd.DataFrame, stations: tuple[str, ...]) -> tuple[float, ...]:
        leg_keys = [(stations[index - 1], stations[index]) for index in range(1, len(stations))]
        leg_times: dict[tuple[str, str], list[float]] = {key: [] for key in leg_keys}

        for _, journey in group.groupby("journey"):
            ordered = journey.sort_values("stop_sequence")
            previous_station = None
            previous_time = None

            for row in ordered.itertuples(index=False):
                station = str(row.station_short)
                current_time = row.departure_plan_station

                if previous_station is not None:
                    key = (previous_station, station)
                    if key in leg_times:
                        minutes = (current_time - previous_time).total_seconds() / 60
                        if 0 < minutes <= 30:
                            leg_times[key].append(minutes)

                previous_station = station
                previous_time = current_time

        result = []
        for key in leg_keys:
            values = leg_times[key]
            result.append(float(np.median(values)) if values else DEFAULT_TRAVEL_TIME)

        return tuple(result)

    def _median_headway(self, group: pd.DataFrame) -> float:
        if group["source_model_ready"].all():
            first_sequence = group["stop_sequence"].min()
            first_stops = group[group["stop_sequence"] == first_sequence].copy()
        else:
            first_stops = group.sort_values("stop_sequence").groupby("journey", as_index=False).first()
        diffs = []

        for _, day in first_stops.groupby("service_date"):
            times = day["departure_plan_station"].sort_values()
            day_diffs = times.diff().dt.total_seconds().div(60).dropna()
            diffs.extend(day_diffs[(day_diffs >= 3) & (day_diffs <= 120)].tolist())

        if not diffs:
            return DEFAULT_HEADWAY

        return float(np.clip(np.median(diffs), MIN_HEADWAY, MAX_HEADWAY))

    def _most_common_vehicle_type(self, group: pd.DataFrame) -> str:
        values = group["vehicle_type"].dropna().astype(str)
        if values.empty:
            return "unknown"
        return str(values.mode().iloc[0])

    def _infer_capacity(self, group: pd.DataFrame) -> int:
        # Capacity is not stored directly, so infer it from observed load and utilization.
        usable = group[
            (group["occupancy_departure"] > 0)
            & (group["vehicle_utilization"] > 0)
        ].copy()
        if usable.empty:
            return VEHICLE_TYPE_CAPACITY.get(self._most_common_vehicle_type(group), BUS_CAPACITY)

        inferred = usable["occupancy_departure"] / usable["vehicle_utilization"]
        inferred = inferred.replace([np.inf, -np.inf], np.nan).dropna()
        if inferred.empty:
            return VEHICLE_TYPE_CAPACITY.get(self._most_common_vehicle_type(group), BUS_CAPACITY)

        return int(round(float(inferred.median())))

    def _build_hourly_profile(self, value_column: str) -> pd.DataFrame:
        daily = (
            self.raw.groupby(["line", "direction", "station_short", "hour", "service_date"], as_index=False)
            .agg(value=(value_column, "sum"))
        )
        profile = (
            daily.groupby(["line", "direction", "station_short", "hour"], as_index=False)
            .agg(mean=("value", "mean"), variance=("value", "var"))
        )
        profile["variance"] = profile["variance"].fillna(profile["mean"])
        return profile.set_index(["line", "direction", "station_short", "hour"])

    def _build_visit_profile(self, value_column: str) -> pd.DataFrame:
        profile = (
            self.raw.groupby(["line", "direction", "station_short", "hour"], as_index=False)
            .agg(mean=(value_column, "mean"), variance=(value_column, "var"))
        )
        profile["variance"] = profile["variance"].fillna(profile["mean"])
        return profile.set_index(["line", "direction", "station_short", "hour"])

    def train_forecast_models(self, save_path: str | Path | None = None) -> dict[str, TrainedForecastModel]:
        if hasattr(self, "trained_models") and self.trained_models:
            return self.trained_models

        training_data = self._prepare_forecast_training_data(training_only=True)
        self.trained_models = {
            "random_forest": train_random_forest_forecast(training_data),
            "gradient_boosting": train_gradient_boosting_forecast(training_data, save_path=save_path),
        }
        return self.trained_models

    def get_trained_model(self, name: str) -> TrainedForecastModel | None:
        if not hasattr(self, "trained_models"):
            return None
        return self.trained_models.get(name)

    def compare_forecast_models(self, save_path: str | Path | None = None) -> pd.DataFrame:
        self.train_forecast_models(save_path=save_path)
        hourly = self._prepare_forecast_training_data(training_only=False)
        split_date = pd.to_datetime(TRAIN_TEST_SPLIT_DATE)
        train = hourly[hourly["service_date"] < split_date].copy()
        test = hourly[hourly["service_date"] >= split_date].copy()

        if test.empty:
            raise ValueError("No holdout data available for forecast comparison.")

        baseline_station_mean = train.groupby("station_short")["target"].mean().to_dict()
        baseline_station_hourly_mean = train.groupby(["station_short", "hour"])["target"].mean().to_dict()
        global_mean = float(train["target"].mean()) if not train.empty else 0.0

        # Compare lightweight statistical baselines and the trained tree model on the same holdout period.
        predictions = {
            "historical_mean": [
                self._baseline_prediction(row, baseline_station_hourly_mean, baseline_station_mean, global_mean)
                for _, row in test.iterrows()
            ],
            "moving_average": self._rolling_holdout_predictions(train, test, mode="moving_average"),
            "exponential_smoothing": self._rolling_holdout_predictions(train, test, mode="exponential_smoothing"),
        }

        for model_name in ["random_forest", "gradient_boosting"]:
            trained_model = self.get_trained_model(model_name)
            if trained_model is not None:
                predictions[model_name] = trained_model.predict_frame(test)

        rows = []
        actual = test["target"].to_numpy(dtype=float)
        for model_name, predicted in predictions.items():
            predicted_values = np.asarray(predicted, dtype=float)
            rows.append(
                {
                    "forecast_model": model_name,
                    "n_train": len(train),
                    "n_test": len(test),
                    "mae": round(mean_absolute_error(actual, predicted_values), 4),
                    "rmse": round(math.sqrt(mean_squared_error(actual, predicted_values)), 4),
                    "r2": round(r2_score(actual, predicted_values), 4),
                }
            )

        return pd.DataFrame(rows).sort_values(["mae", "rmse"]).reset_index(drop=True)

    def _baseline_prediction(
        self,
        row: pd.Series,
        station_hourly_mean: dict[tuple[str, int], float],
        station_mean: dict[str, float],
        global_mean: float,
    ) -> float:
        station = str(row["station_short"])
        hour = int(row["hour"])
        return float(station_hourly_mean.get((station, hour), station_mean.get(station, global_mean)))

    def _rolling_holdout_predictions(self, train: pd.DataFrame, test: pd.DataFrame, mode: str) -> list[float]:
        station_keys = ["line", "direction", "station_short"]
        train_sorted = train.sort_values(["service_date", "hour"])
        test_sorted = test.sort_values(["service_date", "hour"])

        history: dict[tuple[int, str, str], deque[float]] = defaultdict(lambda: deque(maxlen=FORECAST_WINDOW_SIZE))
        smooth_state: dict[tuple[int, str, str], float] = {}
        fallback_mean = float(train["target"].mean()) if not train.empty else 0.0

        for key_values, group in train_sorted.groupby(station_keys):
            key = (int(key_values[0]), str(key_values[1]), str(key_values[2]))
            values = group["target"].tail(FORECAST_WINDOW_SIZE).astype(float).tolist()
            history[key].extend(values)
            smooth_state[key] = float(group["target"].mean())

        predictions_by_index = {}
        for index, row in test_sorted.iterrows():
            key = (int(row["line"]), str(row["direction"]), str(row["station_short"]))
            if mode == "moving_average":
                prediction = float(np.mean(history[key])) if history[key] else fallback_mean
                history[key].append(float(row["target"]))
            else:
                previous = smooth_state.get(key, fallback_mean)
                prediction = previous
                smooth_state[key] = FORECAST_ALPHA * float(row["target"]) + (1 - FORECAST_ALPHA) * previous
            predictions_by_index[index] = max(0.0, prediction)

        return [predictions_by_index[index] for index in test.index]

    def _prepare_forecast_training_data(self, training_only: bool = True) -> pd.DataFrame:
        hourly = (
            self.raw.groupby(["line", "direction", "station_short", "hour", "service_date"], as_index=False)
            .agg(target=("passenger_boarding_measured", "sum"))
        )
        hourly["service_date"] = pd.to_datetime(hourly["service_date"])
        hourly["weekday"] = hourly["service_date"].dt.weekday
        if training_only and TRAIN_TEST_SPLIT_DATE:
            training_cutoff = pd.to_datetime(TRAIN_TEST_SPLIT_DATE)
            hourly = hourly[hourly["service_date"] < training_cutoff]
        return hourly

    def expected_hourly_demand(self, line: int, direction: str, station: str, hour: int) -> float:
        return self._profile_mean(self.boarding_profile, line, direction, station, hour)

    def expected_route_demand(self, route: LineRoute, hour: int) -> float:
        return sum(
            self.expected_hourly_demand(route.line, route.direction, station, hour)
            for station in route.stations
        )

    def fallback_hourly_demand(self, line: int, direction: str, station: str, hour: int) -> tuple[float, str]:
        # Use the most specific historical signal first, then widen the fallback scope.
        exact_key = (line, direction, station, hour)
        if exact_key in self.boarding_profile.index:
            row = self.boarding_profile.loc[exact_key]
            return float(row["mean"]), "exact_hour"

        adjacent_values = []
        for adjacent_hour in ((hour - 1) % 24, (hour + 1) % 24):
            adjacent_key = (line, direction, station, adjacent_hour)
            if adjacent_key in self.boarding_profile.index:
                adjacent_values.append(float(self.boarding_profile.loc[adjacent_key]["mean"]))

        if adjacent_values:
            return float(np.mean(adjacent_values)), "adjacent_hours"

        station_slice = self._safe_profile_slice(
            self.boarding_profile,
            line=line,
            direction=direction,
            station=station,
        )
        if not station_slice.empty:
            return float(station_slice["mean"].mean()), "historical_stop_mean"

        line_slice = self._safe_profile_slice(
            self.boarding_profile,
            line=line,
            direction=direction,
        )
        if not line_slice.empty:
            return float(line_slice["mean"].mean()), "line_average"

        return 0.0, "zero_fallback"

    def sample_boardings(self, line: int, direction: str, station: str, hour: int, interval: float, rng) -> int:
        mean, variance = self._profile_stats(self.boarding_profile, line, direction, station, hour)
        scale = interval / 60
        return sample_count(mean * scale, max(mean * scale, variance * scale), rng)

    def sample_exits(self, line: int, direction: str, station: str, hour: int, rng) -> int:
        mean, variance = self._profile_stats(self.exiting_profile, line, direction, station, hour)
        return sample_count(mean, variance, rng)

    def _profile_mean(self, profile: pd.DataFrame, line: int, direction: str, station: str, hour: int) -> float:
        mean, _ = self._profile_stats(profile, line, direction, station, hour)
        return mean

    def _profile_stats(self, profile: pd.DataFrame, line: int, direction: str, station: str, hour: int):
        keys = [
            (line, direction, station, hour),
            (line, direction, station, SIMULATION_START_HOUR),
        ]

        for key in keys:
            if key in profile.index:
                row = profile.loc[key]
                return float(row["mean"]), float(row["variance"])

        return 0.0, 0.0

    def _safe_profile_slice(
        self,
        profile: pd.DataFrame,
        line: int,
        direction: str | None = None,
        station: str | None = None,
    ) -> pd.DataFrame:
        mask = profile.index.get_level_values("line") == line

        if direction is not None:
            mask &= profile.index.get_level_values("direction") == direction

        if station is not None:
            mask &= profile.index.get_level_values("station_short") == station

        return profile[mask]
    


class ForecastEngine:
    """Adaptive short-term forecast with selectable forecast models and historical fallbacks."""

    def __init__(
        self,
        network: HistoricalNetworkData,
        model: str = "exponential_smoothing",
        alpha: float = FORECAST_ALPHA,
        interval: float = DEMAND_INTERVAL,
        window_size: int = FORECAST_WINDOW_SIZE,
        save_path: str | Path | None = None,
    ):
        if model not in FORECAST_MODELS:
            raise ValueError(f"Unbekanntes Forecast-Modell: {model}. Erlaubt: {FORECAST_MODELS}")

        self.network = network
        self.model = model
        self.alpha = alpha
        self.interval = interval
        self.window_size = window_size
        self.forecasts: dict[tuple[int, str, str], float] = {}
        self.fallback_levels: dict[tuple[int, str, str], str] = {}
        self.observations: dict[tuple[int, str, str], deque[float]] = {}
        self.trained_forecast_cache: dict[tuple[int, str, str, int, int], float] = {}
        self.trained_model: object | None = None

        if self.model in {"random_forest", "gradient_boosting"}:
            self.network.train_forecast_models(save_path=save_path)
            self.trained_model = self.network.get_trained_model(self.model)
            if self.trained_model is None:
                raise RuntimeError(f"Trainiertes Modell {self.model} konnte nicht geladen werden.")
        elif self.model == "gnn":
            self.trained_model = train_gnn_forecast(
                data_dir=self.network.data_dir,
                lines=self.network.lines,
                split_date=TRAIN_TEST_SPLIT_DATE,
            )

    def forecast_station(self, route: LineRoute, station: str, sim_time: float) -> float:
        key = (route.line, route.direction, station)

        # Historical mean is intentionally stateless and follows the current simulation hour.
        if self.model == "historical_mean":
            forecast, fallback_level = self._historical_forecast(route, station, sim_time)
            self.forecasts[key] = forecast
            self.fallback_levels[key] = fallback_level
            return forecast

        # Moving average reacts only after the first observed demand interval.
        if self.model == "moving_average" and key in self.observations and self.observations[key]:
            forecast = float(np.mean(self.observations[key]))
            self.forecasts[key] = forecast
            return forecast

        # Supervised ML models use pre-trained station-hour forecasts.
        if self.model in {"random_forest", "gradient_boosting", "gnn"} and self.trained_model is not None:
            timestamp = simulation_timestamp(sim_time)
            hour = timestamp.hour
            weekday = timestamp.weekday()
            cache_key = (route.line, route.direction, station, hour, weekday)
            if cache_key not in self.trained_forecast_cache:
                if self.model == "gradient_boosting":
                    forecast = self.trained_model.predict_station(
                    route=route,
                    station=station,
                    hour=hour,
                    weekday=weekday,
                    month=timestamp.month,
                )
                else:
                    forecast = self.trained_model.predict_station(route, station, hour, weekday)
                self.trained_forecast_cache[cache_key] = float(np.clip(forecast, 0.0, None))
            self.forecasts[key] = self.trained_forecast_cache[cache_key]
            self.fallback_levels[key] = self.model
            return self.forecasts[key]

        # Other adaptive models start from the historical fallback before live observations arrive.
        if key not in self.forecasts:
            forecast, fallback_level = self._historical_forecast(route, station, sim_time)
            self.forecasts[key] = forecast
            self.fallback_levels[key] = fallback_level

        return self.forecasts[key]

    def forecast_route_demand(self, route: LineRoute, sim_time: float) -> float:
        return sum(self.forecast_station(route, station, sim_time) for station in route.stations)

    def update(self, route: LineRoute, station: str, observed_demand: int, sim_time: float) -> dict:
        key = (route.line, route.direction, station)
        previous = self.forecast_station(route, station, sim_time)
        observed_hourly_rate = observed_demand * (60 / self.interval)

        # All forecast models work in passengers per hour, even though demand arrives per interval.
        if self.model == "historical_mean":
            forecast, fallback_level = self._historical_forecast(route, station, sim_time)
            self.fallback_levels[key] = fallback_level
        elif self.model == "moving_average":
            history = self.observations.setdefault(key, deque(maxlen=self.window_size))
            history.append(observed_hourly_rate)
            forecast = float(np.mean(history))
        elif self.model in {"random_forest", "gradient_boosting", "gnn"} and self.trained_model is not None:
            forecast = self.forecast_station(route, station, sim_time)
            self.fallback_levels[key] = self.model
        else:
            forecast = self.alpha * observed_hourly_rate + (1 - self.alpha) * previous

        self.forecasts[key] = forecast

        return {
            "forecast_model": self.model,
            "previous_forecast": round(previous, 3),
            "observed_hourly_rate": round(observed_hourly_rate, 3),
            "forecast": round(forecast, 3),
            "fallback_level": self.fallback_levels.get(key, "exact_hour"),
        }

    def _historical_forecast(self, route: LineRoute, station: str, sim_time: float) -> tuple[float, str]:
        return self.network.fallback_hourly_demand(
            route.line,
            route.direction,
            station,
            simulation_hour(sim_time),
        )


class WaitingTimeTracker:
    """FIFO passenger queues per line, direction, and stop with exact wait aggregation."""

    def __init__(self, routes: dict[tuple[int, str], LineRoute]):
        self.queues = {
            (route.line, route.direction, station): deque()
            for route in routes.values()
            for station in route.stations
        }
        self.wait_time_sum = {key: 0.0 for key in self.queues}
        self.boarded_count = {key: 0 for key in self.queues}

    def add(self, key: tuple[int, str, str], count: int, arrival_time: float):
        if count <= 0:
            return
        self.queues.setdefault(key, deque()).append([arrival_time, int(count)])
        self.wait_time_sum.setdefault(key, 0.0)
        self.boarded_count.setdefault(key, 0)

    def waiting_count(self, key: tuple[int, str, str]) -> int:
        return sum(count for _, count in self.queues.get(key, ()))

    def board(self, key: tuple[int, str, str], capacity: int, now: float) -> dict:
        waiting_before = self.waiting_count(key)
        to_board = min(waiting_before, max(0, int(capacity)))
        remaining = to_board
        wait_time_sum = 0.0

        queue = self.queues.setdefault(key, deque())
        while remaining > 0 and queue:
            arrival_time, count = queue[0]
            boarded_from_batch = min(count, remaining)
            wait_time_sum += (now - arrival_time) * boarded_from_batch
            remaining -= boarded_from_batch
            count -= boarded_from_batch

            if count == 0:
                queue.popleft()
            else:
                queue[0][1] = count

        boarded = to_board - remaining
        self.wait_time_sum[key] = self.wait_time_sum.get(key, 0.0) + wait_time_sum
        self.boarded_count[key] = self.boarded_count.get(key, 0) + boarded

        return {
            "waiting_before": waiting_before,
            "boarding": boarded,
            "denied_boarding": waiting_before - boarded,
            "wait_time_sum": wait_time_sum,
            "avg_wait_this_stop": wait_time_sum / boarded if boarded else 0.0,
        }

    def left_waiting(self, key: tuple[int, str, str]) -> int:
        return self.waiting_count(key)

    def average_wait(self, key: tuple[int, str, str]) -> float:
        boarded = self.boarded_count.get(key, 0)
        if boarded == 0:
            return 0.0
        return self.wait_time_sum.get(key, 0.0) / boarded

    def keys(self):
        return self.queues.keys()


class DemandModel:
    """Samples stochastic passenger demand from historical network profiles."""

    def __init__(self, network: HistoricalNetworkData, seed: int = 42):
        self.network = network
        self.rng = np.random.default_rng(seed)

    def sample_new_demand(self, route: LineRoute, sim_time: float, interval: float) -> dict[str, int]:
        hour = simulation_hour(sim_time)
        return {
            station: self.network.sample_boardings(
                route.line,
                route.direction,
                station,
                hour,
                interval,
                self.rng,
            )
            for station in route.stations
        }

    def sample_exits(self, route: LineRoute, station: str, sim_time: float) -> int:
        return self.network.sample_exits(
            route.line,
            route.direction,
            station,
            simulation_hour(sim_time),
            self.rng,
        )


class Bus:
    """Active simulation entity that drives a route and serves passengers at stops."""

    def __init__(
        self,
        env: simpy.Environment,
        bus_id: str,
        route: LineRoute,
        waiting_tracker: WaitingTimeTracker,
        demand_model: DemandModel,
        logger: EventLogger,
        capacity: int = BUS_CAPACITY,
    ):
        self.env = env
        self.id = bus_id
        self.route = route
        self.waiting_tracker = waiting_tracker
        self.demand_model = demand_model
        self.logger = logger
        self.capacity = capacity
        self.occupancy = 0

    def run(self, fleet: PriorityFleet, planned_departure: float, priority: float, priority_demand: float):
        request = fleet.request(priority=priority)
        yield request

        dispatch_delay = self.env.now - planned_departure
        self.logger.log(
            time=round(self.env.now, 2),
            event="bus_start",
            bus_id=self.id,
            line=self.route.line,
            direction=self.route.direction,
            planned_departure=round(planned_departure, 2),
            dispatch_delay=round(dispatch_delay, 2),
            priority=round(priority, 3),
            priority_demand=round(priority_demand, 3),
            occupancy=self.occupancy,
        )

        for index, station in enumerate(self.route.stations):
            if index > 0:
                yield self.env.timeout(self.route.travel_times[index - 1])

            yield self.env.process(self.handle_stop(station))

        self.logger.log(
            time=round(self.env.now, 2),
            event="bus_end",
            bus_id=self.id,
            line=self.route.line,
            direction=self.route.direction,
            occupancy=self.occupancy,
        )
        fleet.release(request)

    def handle_stop(self, station: str):
        key = (self.route.line, self.route.direction, station)
        planned_exits = self.demand_model.sample_exits(self.route, station, self.env.now)
        alighting = min(self.occupancy, planned_exits)
        self.occupancy -= alighting

        available_seats = self.capacity - self.occupancy
        boarding_result = self.waiting_tracker.board(key, available_seats, self.env.now)
        waiting_before = boarding_result["waiting_before"]
        boarding = boarding_result["boarding"]
        denied_boarding = boarding_result["denied_boarding"]
        self.occupancy += boarding

        self.logger.log(
            time=round(self.env.now, 2),
            event="stop_served",
            bus_id=self.id,
            line=self.route.line,
            direction=self.route.direction,
            stop=station,
            station_name=self.route.station_names.get(station, station),
            waiting_before=waiting_before,
            boarding=boarding,
            denied_boarding=denied_boarding,
            wait_time_sum=round(boarding_result["wait_time_sum"], 3),
            avg_wait_this_stop=round(boarding_result["avg_wait_this_stop"], 3),
            alighting=alighting,
            occupancy=self.occupancy,
        )

        dwell_time = max(MIN_DWELL_TIME, (boarding + alighting) * DWELL_TIME_PER_PASSENGER)
        yield self.env.timeout(dwell_time)


def demand_generator(
    env: simpy.Environment,
    routes: dict[tuple[int, str], LineRoute],
    waiting_tracker: WaitingTimeTracker,
    demand_model: DemandModel,
    logger: EventLogger,
    forecast_engine: ForecastEngine | None = None,
):
    while True:
        for route in routes.values():
            generated = demand_model.sample_new_demand(route, env.now, DEMAND_INTERVAL)
            for station, demand in generated.items():
                key = (route.line, route.direction, station)
                waiting_tracker.add(key, demand, env.now)

                forecast_update = {}
                if forecast_engine is not None:
                    forecast_update = forecast_engine.update(route, station, demand, env.now)

                logger.log(
                    time=round(env.now, 2),
                    event="demand_generated",
                    line=route.line,
                    direction=route.direction,
                    stop=station,
                    station_name=route.station_names.get(station, station),
                    demand=demand,
                    waiting_after=waiting_tracker.waiting_count(key),
                    **forecast_update,
                )

        yield env.timeout(DEMAND_INTERVAL)


def dispatch_line(
    env: simpy.Environment,
    route: LineRoute,
    fleet: PriorityFleet,
    waiting_tracker: WaitingTimeTracker,
    demand_model: DemandModel,
    network: HistoricalNetworkData,
    forecast_engine: ForecastEngine | None,
    logger: EventLogger,
    until: float,
    policy: str,
):
    trip_number = 1

    while env.now < until:
        planned_departure = env.now
        bus = Bus(
            env=env,
            bus_id=f"{policy}-{route.line}-{route.direction}-{trip_number}",
            route=route,
            waiting_tracker=waiting_tracker,
            demand_model=demand_model,
            logger=logger,
            capacity=route.capacity,
        )
        priority_demand = priority_demand_for(policy, route, network, forecast_engine, env.now)
        env.process(
            bus.run(
                fleet=fleet,
                planned_departure=planned_departure,
                priority=-priority_demand,
                priority_demand=priority_demand,
            )
        )
        trip_number += 1

        yield env.timeout(next_headway(policy, route, network, forecast_engine, env.now))


def next_headway(
    policy: str,
    route: LineRoute,
    network: HistoricalNetworkData,
    forecast_engine: ForecastEngine | None,
    sim_time: float,
) -> float:
    if policy == "static":
        return route.static_headway

    if forecast_engine is not None:
        expected_demand = forecast_or_historical_route_demand(route, network, forecast_engine, sim_time)
    else:
        expected_demand = network.expected_route_demand(route, simulation_hour(sim_time))

    # Convert expected hourly demand into the number of trips needed at the target load factor.
    target_capacity = route.capacity * TARGET_LOAD_FACTOR
    trips_needed = max(1, math.ceil(expected_demand / target_capacity))
    adaptive_headway = 60 / trips_needed

    if route.line == 10:
        adaptive_headway *= 0.85

    adaptive_headway = min(adaptive_headway, route.static_headway)

    return float(np.clip(adaptive_headway, MIN_HEADWAY, MAX_HEADWAY))


def priority_demand_for(
    policy: str,
    route: LineRoute,
    network: HistoricalNetworkData,
    forecast_engine: ForecastEngine | None,
    sim_time: float,
) -> float:
    if forecast_engine is not None:
        return forecast_or_historical_route_demand(route, network, forecast_engine, sim_time)

    return network.expected_route_demand(route, simulation_hour(sim_time))


def forecast_or_historical_route_demand(
    route: LineRoute,
    network: HistoricalNetworkData,
    forecast_engine: ForecastEngine,
    sim_time: float,
) -> float:
    forecast_demand = forecast_engine.forecast_route_demand(route, sim_time)
    historical_demand = network.expected_route_demand(route, simulation_hour(sim_time))
    return max(forecast_demand, historical_demand)


def sample_count(mean: float, variance: float, rng: np.random.Generator) -> int:
    if mean <= 0:
        return 0

    if variance > mean:
        p = np.clip(mean / variance, 1e-6, 1 - 1e-6)
        r = mean * p / (1 - p)
        return int(rng.negative_binomial(r, p))

    return int(rng.poisson(mean))


def simulation_hour(sim_time: float) -> int:
    return simulation_timestamp(sim_time).hour


def simulation_timestamp(sim_time: float) -> pd.Timestamp:
    return SIMULATION_START_TIMESTAMP + pd.Timedelta(minutes=float(sim_time))


def forecast_model_for_policy(policy: str) -> str | None:
    if policy == "adaptive_forecast":
        return "exponential_smoothing"

    prefix = "adaptive_forecast_"
    if policy.startswith(prefix):
        model = policy[len(prefix):]
        if model in FORECAST_MODELS:
            return model

    if policy in FORECAST_MODELS:
        return policy

    return None


def create_station_summary(events: pd.DataFrame, waiting_tracker: WaitingTimeTracker) -> pd.DataFrame:
    served = events[events["event"] == "stop_served"] if not events.empty else pd.DataFrame()
    demand = events[events["event"] == "demand_generated"] if not events.empty else pd.DataFrame()

    rows = []
    for line, direction, stop in sorted(waiting_tracker.keys()):
        stop_served = filter_events(served, line, direction, stop)
        stop_demand = filter_events(demand, line, direction, stop)

        rows.append(
            {
                "line": line,
                "direction": direction,
                "stop": stop,
                "generated_demand": int(stop_demand["demand"].sum()) if "demand" in stop_demand else 0,
                "boarded_passengers": int(stop_served["boarding"].sum()) if "boarding" in stop_served else 0,
                "denied_boarding_events": int((stop_served.get("denied_boarding", 0) > 0).sum())
                if not stop_served.empty
                else 0,
                "left_waiting": int(waiting_tracker.left_waiting((line, direction, stop))),
                "total_wait_time": round(float(waiting_tracker.wait_time_sum.get((line, direction, stop), 0.0)), 3),
                "avg_wait_time": round(float(waiting_tracker.average_wait((line, direction, stop))), 3),
                "visits": int(len(stop_served)),
            }
        )

    return pd.DataFrame(rows)


def create_line_summary(
    events: pd.DataFrame,
    station_summary: pd.DataFrame,
    policy: str,
    routes: dict[tuple[int, str], LineRoute],
    forecast_model: str | None = None,
) -> pd.DataFrame:
    starts = events[events["event"] == "bus_start"] if not events.empty else pd.DataFrame()
    rows = []

    for (line, direction), group in station_summary.groupby(["line", "direction"]):
        route = routes[(line, direction)]
        line_starts = starts[(starts["line"] == line) & (starts["direction"] == direction)]
        rows.append(
            {
                "policy": policy,
                "forecast_model": forecast_model or "none",
                "line": line,
                "direction": direction,
                "vehicle_type": route.vehicle_type,
                "capacity": route.capacity,
                "generated_demand": int(group["generated_demand"].sum()),
                "boarded_passengers": int(group["boarded_passengers"].sum()),
                "left_waiting": int(group["left_waiting"].sum()),
                "avg_wait_time": round(
                    weighted_average(group["avg_wait_time"], group["boarded_passengers"]),
                    3,
                ),
                "trips_started": int(len(line_starts)),
                "mean_dispatch_delay": round(float(line_starts["dispatch_delay"].mean()), 2)
                if not line_starts.empty
                else 0.0,
            }
        )

    return pd.DataFrame(rows)


def filter_events(events: pd.DataFrame, line: int, direction: str, stop: str) -> pd.DataFrame:
    if events.empty:
        return events

    return events[
        (events["line"] == line)
        & (events["direction"] == direction)
        & (events["stop"] == stop)
    ]


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    total_weight = weights.sum()
    if total_weight == 0:
        return 0.0
    return float((values * weights).sum() / total_weight)


def run_simulation(
    network: HistoricalNetworkData,
    policy: str,
    forecast_model: str | None = None,
    until: int = SIMULATION_TIME,
    fleet_size: int = FLEET_SIZE,
    seed: int = 42,
    verbose: bool = False,
    save_path: str | Path | None = None,
):
    env = simpy.Environment()
    logger = EventLogger(verbose=verbose)
    fleet = PriorityFleet(env, size=fleet_size)
    demand_model = DemandModel(network, seed=seed)
    selected_forecast_model = forecast_model or forecast_model_for_policy(policy)
    forecast_engine = (
        ForecastEngine(network, model=selected_forecast_model, save_path=save_path)
        if selected_forecast_model is not None
        else None
    )
    waiting_tracker = WaitingTimeTracker(network.routes)

    env.process(
        demand_generator(
            env,
            network.routes,
            waiting_tracker,
            demand_model,
            logger,
            forecast_engine,
        )
    )
    for route in network.routes.values():
        env.process(
            dispatch_line(
                env=env,
                route=route,
                fleet=fleet,
                waiting_tracker=waiting_tracker,
                demand_model=demand_model,
                network=network,
                forecast_engine=forecast_engine,
                logger=logger,
                until=until,
                policy=policy,
            )
        )

    env.run(until=until)

    events = logger.get_df()
    station_summary = create_station_summary(events, waiting_tracker)
    line_summary = create_line_summary(
        events,
        station_summary,
        policy,
        network.routes,
        selected_forecast_model,
    )

    return events, station_summary, line_summary


def write_route_overview(network: HistoricalNetworkData):
    rows = []
    for route in network.routes.values():
        rows.append(
            {
                "line": route.line,
                "description": LINE_LABELS.get(route.line, ""),
                "direction": route.direction,
                "stops": len(route.stations),
                "static_headway_min": round(route.static_headway, 2),
                "vehicle_type": route.vehicle_type,
                "capacity": route.capacity,
                "route": " -> ".join(route.stations),
            }
        )

    return pd.DataFrame(rows).sort_values(["line", "direction"])


def print_scenario_output(title: str, summary: pd.DataFrame):
    columns = [
        "policy",
        "forecast_model",
        "line",
        "direction",
        "vehicle_type",
        "capacity",
        "generated_demand",
        "boarded_passengers",
        "left_waiting",
        "avg_wait_time",
        "trips_started",
    ]
    print(f"\n{title}")
    print(summary[columns].to_string(index=False))


def format_forecast_metrics_for_console(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return "Keine Forecast-Metriken verfügbar."

    ordered = metrics.sort_values(["mae", "rmse", "r2"], ascending=[True, True, False]).reset_index(drop=True)
    lines = [
        "Forecast model KPI comparison:",
        "model | n_train | n_test | MAE | RMSE | R2",
        "-" * 72,
    ]
    for _, row in ordered.iterrows():
        lines.append(
            f"{row['forecast_model']:<24} {int(row['n_train']):>7} {int(row['n_test']):>7} "
            f"{float(row['mae']):>7.4f} {float(row['rmse']):>7.4f} {float(row['r2']):>7.4f}"
        )
    return "\n".join(lines)


def plot_forecast_model_comparison(metrics: pd.DataFrame, output_path: Path):
    fig, axis = plt.subplots(figsize=(10, 5))
    ordered = metrics.sort_values("mae")
    axis.bar(ordered["forecast_model"], ordered["mae"], color="#2563eb")
    axis.set_title("Forecast model comparison on 2025 holdout data")
    axis.set_ylabel("MAE: passengers per station-hour")
    axis.tick_params(axis="x", rotation=25)
    for index, value in enumerate(ordered["mae"]):
        axis.text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_simulation_economic_comparison(comparison: pd.DataFrame, output_path: Path):
    summary = (
        comparison.groupby(["policy", "forecast_model"], as_index=False)
        .agg(
            boarded_passengers=("boarded_passengers", "sum"),
            left_waiting=("left_waiting", "sum"),
            trips_started=("trips_started", "sum"),
        )
    )
    wait_summary = (
        comparison.assign(wait_weight=comparison["avg_wait_time"] * comparison["boarded_passengers"])
        .groupby(["policy", "forecast_model"], as_index=False)
        .agg(wait_weight=("wait_weight", "sum"), boarded=("boarded_passengers", "sum"))
    )
    summary["avg_wait_time"] = wait_summary["wait_weight"] / wait_summary["boarded"].replace(0, np.nan)
    summary["label"] = summary["policy"] + "\n(" + summary["forecast_model"] + ")"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("SimPy economic comparison using forecasted demand", fontsize=15, fontweight="bold")
    metrics = [
        ("boarded_passengers", "Served demand", "passengers", "#2563eb"),
        ("left_waiting", "Unserved demand", "passengers", "#dc2626"),
        ("avg_wait_time", "Average wait time", "minutes", "#f59e0b"),
        ("trips_started", "Operating effort", "trips", "#16a34a"),
    ]
    for axis, (metric, title, ylabel, color) in zip(axes.flat, metrics):
        values = summary[metric].fillna(0)
        axis.bar(summary["label"], values, color=color, width=0.58)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=12)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


@contextmanager
def suppress_stderr_fd():
    original_stderr = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(original_stderr, 2)
        os.close(original_stderr)


def load_network_without_external_warnings() -> HistoricalNetworkData:
    # PyArrow can emit platform-specific CPU probing warnings before loading Parquet files.
    with suppress_stderr_fd():
        return HistoricalNetworkData(lines=PROJECT_LINES)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    network = load_network_without_external_warnings()

    route_overview = write_route_overview(network)
    route_overview.to_parquet(OUTPUT_DIR / "route_overview.parquet")

    model_save_path = OUTPUT_DIR / "models" / "gradient_boosting.joblib"
    forecast_metrics = network.compare_forecast_models(save_path=model_save_path)
    forecast_metrics_path = OUTPUT_DIR / "forecast_model_comparison.parquet"
    forecast_plot_path = OUTPUT_DIR / "forecast_model_comparison.png"
    forecast_metrics.to_parquet(forecast_metrics_path)
    plot_forecast_model_comparison(forecast_metrics, forecast_plot_path)
    print(format_forecast_metrics_for_console(forecast_metrics))

    best_forecast_model = str(forecast_metrics.iloc[0]["forecast_model"])
    best_forecast_policy = (
        f"adaptive_forecast_{best_forecast_model}"
        if best_forecast_model in {"historical_mean", "moving_average", "exponential_smoothing"}
        else best_forecast_model
    )

    all_line_summaries = []
    scenarios = [
        ("static", "static"),
        (best_forecast_model, best_forecast_policy),
    ]
    for title, policy in scenarios:
        events, station_summary, line_summary = run_simulation(network, policy=policy, save_path=model_save_path)

        events.to_parquet(OUTPUT_DIR / f"{policy}_event_log.parquet")
        station_summary.to_parquet(OUTPUT_DIR / f"{policy}_station_summary.parquet")
        line_summary.to_parquet(OUTPUT_DIR / f"{policy}_line_summary.parquet")
        all_line_summaries.append(line_summary)

    comparison = pd.concat(all_line_summaries, ignore_index=True)
    simulation_comparison_path = OUTPUT_DIR / "policy_comparison.parquet"
    simulation_plot_path = OUTPUT_DIR / "simpy_economic_comparison.png"
    comparison.to_parquet(simulation_comparison_path)
    plot_simulation_economic_comparison(comparison, simulation_plot_path)

    print(forecast_metrics_path)
    print(forecast_plot_path)
    print(simulation_comparison_path)
    print(simulation_plot_path)

if __name__ == "__main__":
    main()
