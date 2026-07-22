from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from kpi_definitions import (
    DB_KPI_MAPPING,
    KPI_DEFINITIONS,
    SCENARIO_ID_COLUMNS,
)


DEFAULT_OUTPUT_DIR = Path("simulationresults")
DEFAULT_FACT_TABLE_NAME = "fact_kpi_value"
DEFAULT_RUN_TABLE_NAME = "fact_kpi_run"


@dataclass(frozen=True)
class KpiRecord:
    scenario_label: str
    policy: str
    forecast_model: str
    seed: int
    until: int
    fleet_size: int
    peak_quantile: float
    kpi_code: str
    kpi_value: float
    kpi_unit: str
    kpi_direction: str
    kpi_weight: float
    source_level: str
    source_table: str


class KpiRepository:
    def __init__(
        self,
        output_dir: Path | str = DEFAULT_OUTPUT_DIR,
        fact_table_name: str = DEFAULT_FACT_TABLE_NAME,
        run_table_name: str = DEFAULT_RUN_TABLE_NAME,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fact_table_name = fact_table_name
        self.run_table_name = run_table_name

    def build_fact_kpi_value(self, evaluation_frame: pd.DataFrame) -> pd.DataFrame:
        if evaluation_frame.empty:
            return pd.DataFrame(
                columns=[
                    *SCENARIO_ID_COLUMNS,
                    "kpi_code",
                    "kpi_value",
                    "kpi_unit",
                    "kpi_direction",
                    "kpi_weight",
                    "source_level",
                    "source_table",
                ]
            )

        self._validate_evaluation_frame(evaluation_frame)

        records: list[dict[str, object]] = []

        for _, row in evaluation_frame.iterrows():
            scenario_values = {
                "scenario_label": row["scenario_label"],
                "policy": row["policy"],
                "forecast_model": row["forecast_model"],
                "seed": int(row["seed"]),
                "until": int(row["until"]),
                "fleet_size": int(row["fleet_size"]),
                "peak_quantile": float(row["peak_quantile"]),
            }

            for code, definition in KPI_DEFINITIONS.items():
                if code not in row.index:
                    continue
                value = row[code]
                if pd.isna(value):
                    continue

                records.append(
                    {
                        **scenario_values,
                        "kpi_code": code,
                        "kpi_value": float(value),
                        "kpi_unit": definition.unit,
                        "kpi_direction": definition.direction,
                        "kpi_weight": float(definition.weight),
                        "source_level": definition.source_level,
                        "source_table": definition.source_table,
                    }
                )

            if "decision_score" in row.index and not pd.isna(row["decision_score"]):
                records.append(
                    {
                        **scenario_values,
                        "kpi_code": "decision_score",
                        "kpi_value": float(row["decision_score"]),
                        "kpi_unit": "score",
                        "kpi_direction": "maximize",
                        "kpi_weight": 1.0,
                        "source_level": "run",
                        "source_table": "evaluation",
                    }
                )

            if "decision_rank" in row.index and not pd.isna(row["decision_rank"]):
                records.append(
                    {
                        **scenario_values,
                        "kpi_code": "decision_rank",
                        "kpi_value": float(row["decision_rank"]),
                        "kpi_unit": "rank",
                        "kpi_direction": "minimize",
                        "kpi_weight": 1.0,
                        "source_level": "run",
                        "source_table": "evaluation",
                    }
                )

        fact_frame = pd.DataFrame(records)
        if fact_frame.empty:
            return fact_frame

        fact_frame = fact_frame.sort_values(
            ["scenario_label", "kpi_code"]
        ).reset_index(drop=True)
        return fact_frame

    def build_fact_kpi_run(self, evaluation_frame: pd.DataFrame) -> pd.DataFrame:
        if evaluation_frame.empty:
            return pd.DataFrame(columns=list(SCENARIO_ID_COLUMNS))

        self._validate_evaluation_frame(evaluation_frame)

        run_columns = [column for column in SCENARIO_ID_COLUMNS if column in evaluation_frame.columns]
        optional_columns = [
            "generated_demand",
            "boarded_passengers",
            "left_waiting",
            "trips_started",
            "avg_wait_time",
            "mean_dispatch_delay",
            "peak_threshold",
            "peak_generated_demand",
            "peak_boarded_passengers",
            "peak_left_waiting",
            "avg_occupancy",
            "peak_occupancy",
            "peak_load_factor",
            "decision_score",
            "decision_rank",
        ]
        available_optional = [column for column in optional_columns if column in evaluation_frame.columns]

        frame = evaluation_frame[run_columns + available_optional].copy()
        frame = frame.sort_values(["scenario_label"]).reset_index(drop=True)
        return frame

    def save_fact_tables(
        self,
        evaluation_frame: pd.DataFrame,
    ) -> tuple[Path, Path, Path, Path]:
        fact_kpi_value = self.build_fact_kpi_value(evaluation_frame)
        fact_kpi_run = self.build_fact_kpi_run(evaluation_frame)

        fact_value_csv = self.output_dir / f"{self.fact_table_name}.csv"
        fact_value_parquet = self.output_dir / f"{self.fact_table_name}.parquet"
        fact_run_csv = self.output_dir / f"{self.run_table_name}.csv"
        fact_run_parquet = self.output_dir / f"{self.run_table_name}.parquet"

        fact_kpi_value.to_csv(fact_value_csv, index=False)
        fact_kpi_value.to_parquet(fact_value_parquet, index=False)
        fact_kpi_run.to_csv(fact_run_csv, index=False)
        fact_kpi_run.to_parquet(fact_run_parquet, index=False)

        return fact_value_csv, fact_value_parquet, fact_run_csv, fact_run_parquet

    def load_fact_kpi_value(self) -> pd.DataFrame:
        path = self.output_dir / f"{self.fact_table_name}.parquet"
        if path.exists():
            return pd.read_parquet(path)

        csv_path = self.output_dir / f"{self.fact_table_name}.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)

        return pd.DataFrame()

    def load_fact_kpi_run(self) -> pd.DataFrame:
        path = self.output_dir / f"{self.run_table_name}.parquet"
        if path.exists():
            return pd.read_parquet(path)

        csv_path = self.output_dir / f"{self.run_table_name}.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)

        return pd.DataFrame()

    def upsert_fact_kpi_value(self, new_frame: pd.DataFrame) -> pd.DataFrame:
        existing = self.load_fact_kpi_value()
        combined = pd.concat([existing, new_frame], ignore_index=True)

        if combined.empty:
            return combined

        dedupe_keys = [
            "scenario_label",
            "policy",
            "forecast_model",
            "seed",
            "until",
            "fleet_size",
            "peak_quantile",
            "kpi_code",
        ]
        combined = combined.drop_duplicates(subset=dedupe_keys, keep="last")
        combined = combined.sort_values(["scenario_label", "kpi_code"]).reset_index(drop=True)

        path_csv = self.output_dir / f"{self.fact_table_name}.csv"
        path_parquet = self.output_dir / f"{self.fact_table_name}.parquet"
        combined.to_csv(path_csv, index=False)
        combined.to_parquet(path_parquet, index=False)
        return combined

    def upsert_fact_kpi_run(self, new_frame: pd.DataFrame) -> pd.DataFrame:
        existing = self.load_fact_kpi_run()
        combined = pd.concat([existing, new_frame], ignore_index=True)

        if combined.empty:
            return combined

        dedupe_keys = [
            "scenario_label",
            "policy",
            "forecast_model",
            "seed",
            "until",
            "fleet_size",
            "peak_quantile",
        ]
        combined = combined.drop_duplicates(subset=dedupe_keys, keep="last")
        combined = combined.sort_values(["scenario_label"]).reset_index(drop=True)

        path_csv = self.output_dir / f"{self.run_table_name}.csv"
        path_parquet = self.output_dir / f"{self.run_table_name}.parquet"
        combined.to_csv(path_csv, index=False)
        combined.to_parquet(path_parquet, index=False)
        return combined

    def save_with_upsert(
        self,
        evaluation_frame: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        fact_kpi_value = self.build_fact_kpi_value(evaluation_frame)
        fact_kpi_run = self.build_fact_kpi_run(evaluation_frame)

        saved_value = self.upsert_fact_kpi_value(fact_kpi_value)
        saved_run = self.upsert_fact_kpi_run(fact_kpi_run)
        return saved_value, saved_run

    def build_db_mapping_table(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []

        for kpi_code, mapping in DB_KPI_MAPPING.items():
            definition = KPI_DEFINITIONS.get(kpi_code)

            rows.append(
                {
                    "kpi_code": kpi_code,
                    "kpi_name": definition.name if definition is not None else kpi_code,
                    "source_table": mapping.get("source_table"),
                    "value_column": mapping.get("value_column"),
                    "dimension_level": mapping.get("dimension_level"),
                    "aggregation": mapping.get("aggregation"),
                    "unit": definition.unit if definition is not None else None,
                    "direction": definition.direction if definition is not None else None,
                    "weight": definition.weight if definition is not None else None,
                }
            )

        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame

        return frame.sort_values(["kpi_code"]).reset_index(drop=True)

    def save_db_mapping_table(self) -> tuple[Path, Path]:
        frame = self.build_db_mapping_table()

        csv_path = self.output_dir / "dim_kpi_mapping.csv"
        parquet_path = self.output_dir / "dim_kpi_mapping.parquet"

        frame.to_csv(csv_path, index=False)
        frame.to_parquet(parquet_path, index=False)
        return csv_path, parquet_path

    @staticmethod
    def _validate_evaluation_frame(evaluation_frame: pd.DataFrame) -> None:
        missing = [column for column in SCENARIO_ID_COLUMNS if column not in evaluation_frame.columns]
        if missing:
            raise ValueError(
                "Die Evaluation-Datei enthaelt nicht alle benoetigten Szenario-Spalten: "
                f"{missing}"
            )


def build_kpi_record_list(evaluation_frame: pd.DataFrame) -> list[KpiRecord]:
    repository = KpiRepository()
    fact_frame = repository.build_fact_kpi_value(evaluation_frame)

    if fact_frame.empty:
        return []

    records: list[KpiRecord] = []
    for _, row in fact_frame.iterrows():
        records.append(
            KpiRecord(
                scenario_label=str(row["scenario_label"]),
                policy=str(row["policy"]),
                forecast_model=str(row["forecast_model"]),
                seed=int(row["seed"]),
                until=int(row["until"]),
                fleet_size=int(row["fleet_size"]),
                peak_quantile=float(row["peak_quantile"]),
                kpi_code=str(row["kpi_code"]),
                kpi_value=float(row["kpi_value"]),
                kpi_unit=str(row["kpi_unit"]),
                kpi_direction=str(row["kpi_direction"]),
                kpi_weight=float(row["kpi_weight"]),
                source_level=str(row["source_level"]),
                source_table=str(row["source_table"]),
            )
        )
    return records


def merge_evaluation_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    if "scenario_label" in combined.columns:
        combined = combined.sort_values(["scenario_label"]).reset_index(drop=True)
    return combined