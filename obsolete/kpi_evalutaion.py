from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from simulation import (
    FLEET_SIZE,
    SIMULATION_TIME,
    HistoricalNetworkData,
    run_simulation,
)

from kpi_definitions import (
    DEFAULT_LOAD_FACTOR_TARGET,
    DEFAULT_LOAD_FACTOR_TOLERANCE,
    DEFAULT_PEAK_QUANTILE,
    DEFAULT_WEIGHTS,
    KPI_DEFINITIONS,
)

from kpi_repository import KpiRepository


@dataclass(frozen=True)
class KpiRunConfig:
    policy: str
    forecast_model: str | None = None
    until: int = SIMULATION_TIME
    fleet_size: int = FLEET_SIZE
    seed: int = 42
    peak_quantile: float = DEFAULT_PEAK_QUANTILE
    label: str | None = None


class KpiEvaluator:
    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self._validate_weights()

    def _validate_weights(self) -> None:
        required = set(KPI_DEFINITIONS)
        provided = set(self.weights)
        missing = required - provided
        if missing:
            raise ValueError(f"Fehlende Gewichte: {sorted(missing)}")

        total = float(sum(self.weights.values()))
        if total <= 0:
            raise ValueError("Die Summe der Gewichte muss groesser als 0 sein.")

    def evaluate_run(
        self,
        events: pd.DataFrame,
        stationsummary: pd.DataFrame,
        linesummary: pd.DataFrame,
        config: KpiRunConfig,
    ) -> pd.DataFrame:
        base = self._base_run_metrics(linesummary)
        peak = self._peak_metrics(events, quantile=config.peak_quantile)
        load = self._load_metrics(events, linesummary)

        row = {
            "scenario_label": config.label or self._scenario_label(config),
            "policy": config.policy,
            "forecast_model": config.forecast_model or "none",
            "seed": config.seed,
            "until": config.until,
            "fleet_size": config.fleet_size,
            "peak_quantile": config.peak_quantile,
            **base,
            **peak,
            **load,
        }
        return pd.DataFrame([row])

    def add_decision_score(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()

        scored = frame.copy()

        scored["score_demand_coverage"] = self._clip01(scored["demand_coverage"])
        scored["score_peak_demand_coverage"] = self._clip01(scored["peak_demand_coverage"])
        scored["score_unserved_demand_rate"] = 1.0 - self._clip01(scored["unserved_demand_rate"])
        scored["score_avg_load_factor"] = scored["avg_load_factor"].apply(
            lambda value: self._load_factor_score(
                value=value,
                target=DEFAULT_LOAD_FACTOR_TARGET,
                tolerance=DEFAULT_LOAD_FACTOR_TOLERANCE,
            )
        )
        scored["score_avg_wait_time"] = self._minmax_inverse(scored["avg_wait_time"])

        weighted_sum = (
            self.weights["demand_coverage"] * scored["score_demand_coverage"]
            + self.weights["peak_demand_coverage"] * scored["score_peak_demand_coverage"]
            + self.weights["unserved_demand_rate"] * scored["score_unserved_demand_rate"]
            + self.weights["avg_load_factor"] * scored["score_avg_load_factor"]
            + self.weights["avg_wait_time"] * scored["score_avg_wait_time"]
        )
        total_weight = float(sum(self.weights.values()))

        scored["decision_score"] = (weighted_sum / total_weight).round(6)
        scored["decision_rank"] = scored["decision_score"].rank(
            method="dense",
            ascending=False,
        ).astype(int)

        scored = scored.sort_values(
            ["decision_rank", "decision_score", "scenario_label"],
            ascending=[True, False, True],
        )
        return scored.reset_index(drop=True)

    def evaluate_and_store(
        self,
        configs: Iterable[KpiRunConfig],
        network: HistoricalNetworkData | None = None,
        repository: KpiRepository | None = None,
    ) -> pd.DataFrame:
        result = compare_kpi_runs(
            configs=configs,
            network=network,
            weights=self.weights,
        )

        repo = repository or KpiRepository()
        repo.save_with_upsert(result)
        repo.save_db_mapping_table()
        return result

    def _base_run_metrics(self, linesummary: pd.DataFrame) -> dict[str, float | int]:
        if linesummary.empty:
            return {
                "generated_demand": 0,
                "boarded_passengers": 0,
                "left_waiting": 0,
                "trips_started": 0,
                "avg_wait_time": 0.0,
                "mean_dispatch_delay": 0.0,
                "demand_coverage": 0.0,
                "unserved_demand": 0,
                "unserved_demand_rate": 0.0,
            }

        generated = int(pd.to_numeric(linesummary["generateddemand"], errors="coerce").fillna(0).sum())
        boarded = int(pd.to_numeric(linesummary["boardedpassengers"], errors="coerce").fillna(0).sum())
        left_waiting = int(pd.to_numeric(linesummary["leftwaiting"], errors="coerce").fillna(0).sum())
        trips_started = int(pd.to_numeric(linesummary["tripsstarted"], errors="coerce").fillna(0).sum())

        boarded_weights = pd.to_numeric(linesummary["boardedpassengers"], errors="coerce").fillna(0)
        avg_wait_values = pd.to_numeric(linesummary["avgwaittime"], errors="coerce").fillna(0)
        dispatch_values = pd.to_numeric(linesummary["meandispatchdelay"], errors="coerce").fillna(0)

        total_boarded = boarded_weights.sum()
        avg_wait_time = (
            float((avg_wait_values * boarded_weights).sum() / total_boarded)
            if total_boarded > 0
            else 0.0
        )
        mean_dispatch_delay = float(dispatch_values.mean()) if not dispatch_values.empty else 0.0

        demand_coverage = boarded / generated if generated > 0 else 0.0
        unserved_demand = max(0, generated - boarded)
        unserved_demand_rate = unserved_demand / generated if generated > 0 else 0.0

        return {
            "generated_demand": generated,
            "boarded_passengers": boarded,
            "left_waiting": left_waiting,
            "trips_started": trips_started,
            "avg_wait_time": round(avg_wait_time, 6),
            "mean_dispatch_delay": round(mean_dispatch_delay, 6),
            "demand_coverage": round(demand_coverage, 6),
            "unserved_demand": unserved_demand,
            "unserved_demand_rate": round(unserved_demand_rate, 6),
        }

    def _peak_metrics(
        self,
        events: pd.DataFrame,
        quantile: float = DEFAULT_PEAK_QUANTILE,
    ) -> dict[str, float | int]:
        if events.empty or "event" not in events.columns:
            return {
                "peak_threshold": 0.0,
                "peak_generated_demand": 0,
                "peak_boarded_passengers": 0,
                "peak_left_waiting": 0,
                "peak_demand_coverage": 0.0,
            }

        demand_events = events.loc[events["event"] == "demandgenerated"].copy()
        stop_events = events.loc[events["event"] == "stopserved"].copy()

        if demand_events.empty or "demand" not in demand_events.columns:
            return {
                "peak_threshold": 0.0,
                "peak_generated_demand": 0,
                "peak_boarded_passengers": 0,
                "peak_left_waiting": 0,
                "peak_demand_coverage": 0.0,
            }

        demand_events["demand"] = pd.to_numeric(demand_events["demand"], errors="coerce").fillna(0)
        peak_threshold = float(demand_events["demand"].quantile(quantile))
        peak_demand = demand_events.loc[demand_events["demand"] >= peak_threshold].copy()

        if peak_demand.empty:
            return {
                "peak_threshold": round(peak_threshold, 6),
                "peak_generated_demand": 0,
                "peak_boarded_passengers": 0,
                "peak_left_waiting": 0,
                "peak_demand_coverage": 0.0,
            }

        peak_generated_demand = int(peak_demand["demand"].sum())

        if stop_events.empty:
            return {
                "peak_threshold": round(peak_threshold, 6),
                "peak_generated_demand": peak_generated_demand,
                "peak_boarded_passengers": 0,
                "peak_left_waiting": 0,
                "peak_demand_coverage": 0.0,
            }

        stop_events["boarding"] = pd.to_numeric(stop_events.get("boarding"), errors="coerce").fillna(0)
        stop_events["deniedboarding"] = pd.to_numeric(stop_events.get("deniedboarding"), errors="coerce").fillna(0)

        peak_stop_keys = peak_demand[["line", "direction", "stop"]].drop_duplicates()
        peak_served = peak_stop_keys.merge(
            stop_events[["line", "direction", "stop", "boarding", "deniedboarding"]],
            on=["line", "direction", "stop"],
            how="left",
        ).fillna(0)

        peak_boarded_passengers = int(peak_served["boarding"].sum())
        peak_left_waiting = int(peak_served["deniedboarding"].sum())
        peak_demand_coverage = (
            peak_boarded_passengers / peak_generated_demand
            if peak_generated_demand > 0
            else 0.0
        )

        return {
            "peak_threshold": round(peak_threshold, 6),
            "peak_generated_demand": peak_generated_demand,
            "peak_boarded_passengers": peak_boarded_passengers,
            "peak_left_waiting": peak_left_waiting,
            "peak_demand_coverage": round(peak_demand_coverage, 6),
        }

    def _load_metrics(self, events: pd.DataFrame, linesummary: pd.DataFrame) -> dict[str, float]:
        if events.empty or "event" not in events.columns:
            return {
                "avg_occupancy": 0.0,
                "peak_occupancy": 0.0,
                "avg_load_factor": 0.0,
                "peak_load_factor": 0.0,
            }

        stop_events = events.loc[events["event"] == "stopserved"].copy()
        if stop_events.empty:
            return {
                "avg_occupancy": 0.0,
                "peak_occupancy": 0.0,
                "avg_load_factor": 0.0,
                "peak_load_factor": 0.0,
            }

        stop_events["occupancy"] = pd.to_numeric(stop_events.get("occupancy"), errors="coerce").fillna(0)

        if "capacity" in linesummary.columns and not linesummary.empty:
            line_capacity = linesummary.groupby("line", as_index=True)["capacity"].median()
            default_capacity = float(pd.to_numeric(linesummary["capacity"], errors="coerce").median())
            stop_events["capacity"] = stop_events["line"].map(line_capacity).fillna(default_capacity)
        else:
            stop_events["capacity"] = 80.0

        stop_events["load_factor"] = np.where(
            stop_events["capacity"] > 0,
            stop_events["occupancy"] / stop_events["capacity"],
            0.0,
        )

        return {
            "avg_occupancy": round(float(stop_events["occupancy"].mean()), 6),
            "peak_occupancy": round(float(stop_events["occupancy"].max()), 6),
            "avg_load_factor": round(float(stop_events["load_factor"].mean()), 6),
            "peak_load_factor": round(float(stop_events["load_factor"].max()), 6),
        }

    @staticmethod
    def _clip01(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce").fillna(0).clip(lower=0.0, upper=1.0)

    @staticmethod
    def _minmax_inverse(series: pd.Series) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce").fillna(0.0)
        min_v = float(values.min())
        max_v = float(values.max())

        if np.isclose(min_v, max_v):
            return pd.Series(np.ones(len(values)), index=values.index, dtype=float)

        normalized = (values - min_v) / (max_v - min_v)
        return 1.0 - normalized

    @staticmethod
    def _load_factor_score(
        value: float,
        target: float = DEFAULT_LOAD_FACTOR_TARGET,
        tolerance: float = DEFAULT_LOAD_FACTOR_TOLERANCE,
    ) -> float:
        if not np.isfinite(value):
            return 0.0
        deviation = abs(float(value) - target)
        score = max(0.0, 1.0 - (deviation / tolerance))
        return float(min(1.0, score))

    @staticmethod
    def _scenario_label(config: KpiRunConfig) -> str:
        model = config.forecast_model or "none"
        return f"{config.policy}__{model}__seed{config.seed}"


def run_kpi_evaluation(
    config: KpiRunConfig,
    network: HistoricalNetworkData | None = None,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    network = network or HistoricalNetworkData()

    events, stationsummary, linesummary = run_simulation(
        network=network,
        policy=config.policy,
        forecast_model=config.forecast_model,
        until=config.until,
        fleet_size=config.fleet_size,
        seed=config.seed,
        verbose=False,
    )

    evaluator = KpiEvaluator(weights=weights)
    return evaluator.evaluate_run(events, stationsummary, linesummary, config)


def compare_kpi_runs(
    configs: Iterable[KpiRunConfig],
    network: HistoricalNetworkData | None = None,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    configs = list(configs)
    if not configs:
        return pd.DataFrame()

    network = network or HistoricalNetworkData()
    evaluator = KpiEvaluator(weights=weights)
    frames = []

    for config in configs:
        events, stationsummary, linesummary = run_simulation(
            network=network,
            policy=config.policy,
            forecast_model=config.forecast_model,
            until=config.until,
            fleet_size=config.fleet_size,
            seed=config.seed,
            verbose=False,
        )
        frames.append(evaluator.evaluate_run(events, stationsummary, linesummary, config))

    combined = pd.concat(frames, ignore_index=True)
    return evaluator.add_decision_score(combined)


def save_kpi_results(
    frame: pd.DataFrame,
    output_dir: Path | str = "simulationresults",
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "kpi_evaluation.csv"
    parquet_path = output_path / "kpi_evaluation.parquet"

    frame.to_csv(csv_path, index=False)
    frame.to_parquet(parquet_path, index=False)
    return csv_path, parquet_path


def default_configs() -> list[KpiRunConfig]:
    return [
        KpiRunConfig(
            policy="static",
            forecast_model=None,
            seed=42,
            label="static_baseline",
        ),
        KpiRunConfig(
            policy="adaptiveforecast_exponentialsmoothing",
            forecast_model="exponentialsmoothing",
            seed=42,
            label="adaptive_exponentialsmoothing",
        ),
        KpiRunConfig(
            policy="adaptiveforecast_randomforest",
            forecast_model="randomforest",
            seed=42,
            label="adaptive_randomforest",
        ),
    ]


def main() -> None:
    configs = default_configs()
    evaluator = KpiEvaluator(weights=DEFAULT_WEIGHTS)
    result = evaluator.evaluate_and_store(configs=configs)

    csv_path, parquet_path = save_kpi_results(result)
    print(result.to_string(index=False))
    print(csv_path)
    print(parquet_path)


if __name__ == "__main__":
    main()