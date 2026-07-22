from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


KpiDirection = Literal["maximize", "minimize", "target"]


DEFAULT_WEIGHTS: dict[str, float] = {
    "demand_coverage": 0.35,
    "peak_demand_coverage": 0.30,
    "unserved_demand_rate": 0.20,
    "avg_load_factor": 0.10,
    "avg_wait_time": 0.05,
}


DEFAULT_LOAD_FACTOR_TARGET: float = 0.75
DEFAULT_LOAD_FACTOR_TOLERANCE: float = 0.35
DEFAULT_PEAK_QUANTILE: float = 0.90


@dataclass(frozen=True)
class KpiDefinition:
    code: str
    name: str
    description: str
    unit: str
    direction: KpiDirection
    weight: float
    source_level: str
    source_table: str
    source_columns: tuple[str, ...]
    score_column: str
    nullable: bool = False
    target_value: float | None = None
    min_value: float | None = 0.0
    max_value: float | None = 1.0


KPI_DEFINITIONS: dict[str, KpiDefinition] = {
    "demand_coverage": KpiDefinition(
        code="demand_coverage",
        name="Demand Coverage",
        description=(
            "Anteil der insgesamt generierten Nachfrage, der im Simulationslauf "
            "tatsaechlich bedient bzw. befoerdert wurde."
        ),
        unit="ratio",
        direction="maximize",
        weight=DEFAULT_WEIGHTS["demand_coverage"],
        source_level="run",
        source_table="linesummary",
        source_columns=("generateddemand", "boardedpassengers"),
        score_column="score_demand_coverage",
        min_value=0.0,
        max_value=1.0,
    ),
    "peak_demand_coverage": KpiDefinition(
        code="peak_demand_coverage",
        name="Peak Demand Coverage",
        description=(
            "Anteil der bedienten Nachfrage innerhalb der als Peak klassifizierten "
            "Nachfrageereignisse."
        ),
        unit="ratio",
        direction="maximize",
        weight=DEFAULT_WEIGHTS["peak_demand_coverage"],
        source_level="run",
        source_table="events",
        source_columns=("event", "demand", "boarding", "line", "direction", "stop"),
        score_column="score_peak_demand_coverage",
        min_value=0.0,
        max_value=1.0,
    ),
    "unserved_demand_rate": KpiDefinition(
        code="unserved_demand_rate",
        name="Unserved Demand Rate",
        description=(
            "Anteil der generierten Nachfrage, der im Simulationslauf nicht bedient "
            "werden konnte."
        ),
        unit="ratio",
        direction="minimize",
        weight=DEFAULT_WEIGHTS["unserved_demand_rate"],
        source_level="run",
        source_table="linesummary",
        source_columns=("generateddemand", "boardedpassengers", "leftwaiting"),
        score_column="score_unserved_demand_rate",
        min_value=0.0,
        max_value=1.0,
    ),
    "avg_load_factor": KpiDefinition(
        code="avg_load_factor",
        name="Average Load Factor",
        description=(
            "Mittlere Auslastung der Fahrzeuge auf Basis von Occupancy relativ zur "
            "verfuegbaren Kapazitaet."
        ),
        unit="ratio",
        direction="target",
        weight=DEFAULT_WEIGHTS["avg_load_factor"],
        source_level="stop_event",
        source_table="events+linesummary",
        source_columns=("event", "occupancy", "capacity", "line"),
        score_column="score_avg_load_factor",
        target_value=DEFAULT_LOAD_FACTOR_TARGET,
        min_value=0.0,
        max_value=1.0,
    ),
    "avg_wait_time": KpiDefinition(
        code="avg_wait_time",
        name="Average Wait Time",
        description=(
            "Gewichtete durchschnittliche Wartezeit der befoerderten Fahrgaeste "
            "im Simulationslauf."
        ),
        unit="minutes",
        direction="minimize",
        weight=DEFAULT_WEIGHTS["avg_wait_time"],
        source_level="run",
        source_table="linesummary",
        source_columns=("avgwaittime", "boardedpassengers"),
        score_column="score_avg_wait_time",
        nullable=False,
        min_value=None,
        max_value=None,
    ),
}


DECISION_SCORE_COLUMNS: tuple[str, ...] = (
    "score_demand_coverage",
    "score_peak_demand_coverage",
    "score_unserved_demand_rate",
    "score_avg_load_factor",
    "score_avg_wait_time",
    "decision_score",
    "decision_rank",
)


LINESUMMARY_REQUIRED_COLUMNS: tuple[str, ...] = (
    "policy",
    "forecastmodel",
    "line",
    "direction",
    "capacity",
    "generateddemand",
    "boardedpassengers",
    "leftwaiting",
    "avgwaittime",
    "tripsstarted",
    "meandispatchdelay",
)


EVENTS_REQUIRED_COLUMNS: tuple[str, ...] = (
    "event",
    "line",
    "direction",
    "stop",
    "demand",
    "boarding",
    "deniedboarding",
    "occupancy",
)


SCENARIO_ID_COLUMNS: tuple[str, ...] = (
    "scenario_label",
    "policy",
    "forecast_model",
    "seed",
    "until",
    "fleet_size",
    "peak_quantile",
)


DB_KPI_MAPPING: dict[str, dict[str, object]] = {
    "demand_coverage": {
        "kpi_code": "demand_coverage",
        "source_table": "fact_kpi_value",
        "value_column": "kpi_value",
        "dimension_level": "run",
        "aggregation": "single_value",
    },
    "peak_demand_coverage": {
        "kpi_code": "peak_demand_coverage",
        "source_table": "fact_kpi_value",
        "value_column": "kpi_value",
        "dimension_level": "run",
        "aggregation": "single_value",
    },
    "unserved_demand_rate": {
        "kpi_code": "unserved_demand_rate",
        "source_table": "fact_kpi_value",
        "value_column": "kpi_value",
        "dimension_level": "run",
        "aggregation": "single_value",
    },
    "avg_load_factor": {
        "kpi_code": "avg_load_factor",
        "source_table": "fact_kpi_value",
        "value_column": "kpi_value",
        "dimension_level": "run",
        "aggregation": "single_value",
    },
    "avg_wait_time": {
        "kpi_code": "avg_wait_time",
        "source_table": "fact_kpi_value",
        "value_column": "kpi_value",
        "dimension_level": "run",
        "aggregation": "single_value",
    },
    "decision_score": {
        "kpi_code": "decision_score",
        "source_table": "fact_kpi_value",
        "value_column": "kpi_value",
        "dimension_level": "run",
        "aggregation": "single_value",
    },
    "decision_rank": {
        "kpi_code": "decision_rank",
        "source_table": "fact_kpi_value",
        "value_column": "kpi_value",
        "dimension_level": "run",
        "aggregation": "single_value",
    },
}


def get_kpi_definition(code: str) -> KpiDefinition:
    if code not in KPI_DEFINITIONS:
        raise KeyError(f"Unbekannter KPI-Code: {code}")
    return KPI_DEFINITIONS[code]


def get_default_weights() -> dict[str, float]:
    return dict(DEFAULT_WEIGHTS)


def list_kpi_codes() -> list[str]:
    return list(KPI_DEFINITIONS.keys())


def list_kpi_definitions() -> list[KpiDefinition]:
    return [KPI_DEFINITIONS[code] for code in KPI_DEFINITIONS]


def get_score_column_map() -> dict[str, str]:
    return {code: definition.score_column for code, definition in KPI_DEFINITIONS.items()}


def get_weighted_kpi_codes() -> list[str]:
    return [code for code, definition in KPI_DEFINITIONS.items() if definition.weight > 0]


def update_weights(new_weights: dict[str, float]) -> dict[str, KpiDefinition]:
    updated: dict[str, KpiDefinition] = {}
    for code, definition in KPI_DEFINITIONS.items():
        weight = float(new_weights.get(code, definition.weight))
        updated[code] = KpiDefinition(
            code=definition.code,
            name=definition.name,
            description=definition.description,
            unit=definition.unit,
            direction=definition.direction,
            weight=weight,
            source_level=definition.source_level,
            source_table=definition.source_table,
            source_columns=definition.source_columns,
            score_column=definition.score_column,
            nullable=definition.nullable,
            target_value=definition.target_value,
            min_value=definition.min_value,
            max_value=definition.max_value,
        )
    return updated


def validate_weight_set(weights: dict[str, float]) -> None:
    missing = set(KPI_DEFINITIONS) - set(weights)
    if missing:
        raise ValueError(f"Fehlende Gewichte fuer KPIs: {sorted(missing)}")

    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("Die Summe der Gewichte muss groesser als 0 sein.")