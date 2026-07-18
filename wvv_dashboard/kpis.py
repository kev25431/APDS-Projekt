from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from .service_policy import constrained_adaptive_runs, estimated_bus_hours


DEFAULT_BUS_HOURLY_COST_EUR = 230.0
COMFORT_UTILIZATION_LIMIT = 0.85
PRODUCTIVE_UTILIZATION_FLOOR = 0.35


@dataclass(frozen=True)
class KPIResult:
    label: str
    line: int | None
    selected_day: date
    hours: int
    demand: float
    wvv_runs: float
    adaptive_runs: float
    wvv_capacity: float
    adaptive_capacity: float
    wvv_bus_hours: float
    adaptive_bus_hours: float
    wvv_wait_minutes: float
    adaptive_wait_minutes: float
    wvv_cost: float
    adaptive_cost: float
    wvv_coverage: float
    adaptive_coverage: float
    wvv_peak_coverage: float
    adaptive_peak_coverage: float
    wvv_utilization: float
    adaptive_utilization: float
    wvv_overload: float
    adaptive_overload: float
    wvv_underload: float
    adaptive_underload: float
    wvv_unserved: float
    adaptive_unserved: float
    period: str = ""

    @property
    def cost_delta(self) -> float:
        return self.adaptive_cost - self.wvv_cost

    @property
    def run_delta(self) -> float:
        return self.adaptive_runs - self.wvv_runs

    @property
    def unserved_delta(self) -> float:
        return self.adaptive_unserved - self.wvv_unserved


def hourly_run_counts_from_table(table: pd.DataFrame) -> dict[int, int]:
    """Extract hourly departure counts from a WVV timetable matrix."""
    counts: dict[int, int] = {}
    if table.empty:
        return counts
    for column in table.columns:
        match = re.match(r"\d{2}\s+(\d{2}):\d{2}", str(column))
        if match:
            hour = int(match.group(1))
            counts[hour] = counts.get(hour, 0) + 1
    return counts


def attach_wvv_hourly_runs(schedule: pd.DataFrame, hourly_runs: dict[int, int]) -> pd.DataFrame:
    if schedule.empty or not hourly_runs:
        return schedule
    frame = schedule.copy()
    wvv_runs = frame["hour"].map(lambda value: hourly_runs.get(int(value), 0)).fillna(0).astype(float)
    if "baseline_runs" in frame:
        fallback_runs = pd.to_numeric(frame["baseline_runs"], errors="coerce").fillna(0).astype(float)
        frame["baseline_runs"] = wvv_runs.where(wvv_runs > 0, fallback_runs)
    else:
        frame["baseline_runs"] = wvv_runs
    return frame


def calculate_line_kpis(
    schedule: pd.DataFrame,
    *,
    line: int | None,
    label: str,
    selected_day: date,
    bus_hourly_cost: float = DEFAULT_BUS_HOURLY_COST_EUR,
    period: str = "",
) -> KPIResult:
    frame = _normalize_schedule(schedule)
    if frame.empty:
        return _empty_result(label, line, selected_day)

    demand = frame["kpi_demand"].clip(lower=0)
    avg_capacity = frame["avg_vehicle_capacity"].clip(lower=1)
    wvv_runs = frame["baseline_runs"].clip(lower=0)
    frame["recommended_runs"] = frame.apply(
        lambda row: constrained_adaptive_runs(
            demand=float(row.get("predicted_boardings", row.get("kpi_demand", 0.0))),
            baseline_runs=float(row.get("baseline_runs", 0.0)),
            avg_capacity=float(row.get("avg_vehicle_capacity", 90.0)),
            cost_per_bus_hour=bus_hourly_cost,
            default_cost_per_bus_hour=DEFAULT_BUS_HOURLY_COST_EUR,
            line=_row_line(row, line),
            hour=int(row.get("hour", 0)),
            allow_new_service=True,
        ),
        axis=1,
    )
    adaptive_runs = frame["recommended_runs"].clip(lower=0)
    wvv_capacity = wvv_runs * avg_capacity
    adaptive_capacity = adaptive_runs * avg_capacity
    wvv_bus_hours = frame.apply(lambda row: estimated_bus_hours(row["baseline_runs"], _row_line(row, line)), axis=1)
    adaptive_bus_hours = frame.apply(lambda row: estimated_bus_hours(row["recommended_runs"], _row_line(row, line)), axis=1)
    wvv_wait_minutes = _weighted_wait_minutes(demand, wvv_runs)
    adaptive_wait_minutes = _weighted_wait_minutes(demand, adaptive_runs)

    hours = frame["hour"].astype(int)
    wvv_side = _side_metrics(hours, demand, wvv_capacity)
    adaptive_side = _side_metrics(hours, demand, adaptive_capacity)
    return KPIResult(
        label=label,
        line=line,
        selected_day=selected_day,
        hours=int(len(frame) if period else frame["hour"].nunique()),
        demand=float(demand.sum()),
        wvv_runs=float(wvv_runs.sum()),
        adaptive_runs=float(adaptive_runs.sum()),
        wvv_capacity=float(wvv_capacity.sum()),
        adaptive_capacity=float(adaptive_capacity.sum()),
        wvv_bus_hours=float(wvv_bus_hours.sum()),
        adaptive_bus_hours=float(adaptive_bus_hours.sum()),
        wvv_wait_minutes=wvv_wait_minutes,
        adaptive_wait_minutes=adaptive_wait_minutes,
        wvv_cost=float(wvv_bus_hours.sum() * bus_hourly_cost),
        adaptive_cost=float(adaptive_bus_hours.sum() * bus_hourly_cost),
        wvv_coverage=wvv_side["coverage"],
        adaptive_coverage=adaptive_side["coverage"],
        wvv_peak_coverage=wvv_side["peak_coverage"],
        adaptive_peak_coverage=adaptive_side["peak_coverage"],
        wvv_utilization=wvv_side["utilization"],
        adaptive_utilization=adaptive_side["utilization"],
        wvv_overload=wvv_side["overload"],
        adaptive_overload=adaptive_side["overload"],
        wvv_underload=wvv_side["underload"],
        adaptive_underload=adaptive_side["underload"],
        wvv_unserved=wvv_side["unserved"],
        adaptive_unserved=adaptive_side["unserved"],
        period=period,
    )


def calculate_annual_kpis_from_raw(
    raw: pd.DataFrame,
    *,
    year: int,
    label: str,
    bus_hourly_cost: float = DEFAULT_BUS_HOURLY_COST_EUR,
) -> KPIResult:
    if raw.empty:
        return _empty_result(label, None, date(year, 1, 1), period=f"Jahr {year} | alle Linien")

    frame = raw.copy()
    if "date" not in frame or frame["date"].isna().all():
        frame["date"] = frame.get("report_date")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    frame = frame[pd.to_datetime(frame["date"].astype(str), errors="coerce").dt.year == int(year)].copy()
    if frame.empty:
        return _empty_result(label, None, date(year, 1, 1), period=f"Jahr {year} | alle Linien")

    for column in ["line", "hour", "passenger_boarding", "passenger_exiting"]:
        if column not in frame:
            frame[column] = 0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    if "journey" not in frame:
        frame["journey"] = "unknown"
    if "vehicle_type" not in frame:
        frame["vehicle_type"] = "Unknown"
    frame["vehicle_type"] = frame["vehicle_type"].fillna("Unknown").astype(str)
    frame["capacity"] = frame["vehicle_type"].map(_vehicle_capacity).fillna(_vehicle_capacity("Unknown"))

    hourly = (
        frame.groupby(["date", "line", "hour"], as_index=False)
        .agg(
            predicted_boardings=("passenger_boarding", "sum"),
            predicted_exiting=("passenger_exiting", "sum"),
            baseline_runs=("journey", "nunique"),
            avg_vehicle_capacity=("capacity", "mean"),
        )
        .sort_values(["date", "line", "hour"])
    )
    if hourly.empty:
        return _empty_result(label, None, date(year, 1, 1), period=f"Jahr {year} | alle Linien")

    hourly["predicted_demand"] = hourly["predicted_boardings"] + hourly["predicted_exiting"]
    hourly["recommended_runs"] = hourly.apply(
        lambda row: _annual_adaptive_runs(
            demand=float(row["predicted_boardings"]),
            baseline_runs=float(row["baseline_runs"]),
            avg_capacity=float(row["avg_vehicle_capacity"]),
            cost_per_bus_hour=bus_hourly_cost,
            line=row.get("line"),
            hour=row.get("hour"),
        ),
        axis=1,
    )
    return calculate_line_kpis(
        hourly,
        line=None,
        label=label,
        selected_day=date(year, 1, 1),
        bus_hourly_cost=bus_hourly_cost,
        period=f"Jahr {year} | alle Linien",
    )


def calculate_annual_kpis_from_parquet_dir(
    enriched_dir: Path,
    *,
    year: int,
    label: str,
    bus_hourly_cost: float = DEFAULT_BUS_HOURLY_COST_EUR,
) -> KPIResult:
    hourly_parts: list[pd.DataFrame] = []
    pattern = re.compile(r"_clean_context_2025\.parquet$")
    columns = [
        "date",
        "report_date",
        "line",
        "hour",
        "journey",
        "vehicle_type",
        "passenger_boarding",
        "passenger_exiting",
    ]
    if not enriched_dir.exists():
        return _empty_result(label, None, date(year, 1, 1), period=f"Jahr {year} | alle Linien")

    for path in sorted(enriched_dir.glob("*.parquet")):
        if not pattern.search(path.name):
            continue
        try:
            raw = pd.read_parquet(path, columns=columns)
        except Exception:
            try:
                raw = pd.read_parquet(path)
            except Exception:
                continue
        hourly = _hourly_from_raw(raw, year)
        if not hourly.empty:
            hourly_parts.append(hourly)

    if not hourly_parts:
        return _empty_result(label, None, date(year, 1, 1), period=f"Jahr {year} | alle Linien")

    hourly = (
        pd.concat(hourly_parts, ignore_index=True)
        .groupby(["date", "line", "hour"], as_index=False)
        .agg(
            predicted_boardings=("predicted_boardings", "sum"),
            predicted_exiting=("predicted_exiting", "sum"),
            baseline_runs=("baseline_runs", "sum"),
            avg_vehicle_capacity=("avg_vehicle_capacity", "mean"),
        )
    )
    return _calculate_annual_from_hourly(
        hourly,
        year=year,
        label=label,
        bus_hourly_cost=bus_hourly_cost,
    )


def calculate_annual_line_kpis_from_parquet_dir(
    enriched_dir: Path,
    *,
    year: int,
    bus_hourly_cost: float = DEFAULT_BUS_HOURLY_COST_EUR,
) -> list[KPIResult]:
    hourly_parts: list[pd.DataFrame] = []
    pattern = re.compile(r"_clean_context_2025\.parquet$")
    columns = [
        "date",
        "report_date",
        "line",
        "hour",
        "journey",
        "vehicle_type",
        "passenger_boarding",
        "passenger_exiting",
    ]
    if not enriched_dir.exists():
        return []

    for path in sorted(enriched_dir.glob("*.parquet")):
        if not pattern.search(path.name):
            continue
        try:
            raw = pd.read_parquet(path, columns=columns)
        except Exception:
            try:
                raw = pd.read_parquet(path)
            except Exception:
                continue
        hourly = _hourly_from_raw(raw, year)
        if not hourly.empty:
            hourly_parts.append(hourly)
    if not hourly_parts:
        return []

    hourly = (
        pd.concat(hourly_parts, ignore_index=True)
        .groupby(["date", "line", "hour"], as_index=False)
        .agg(
            predicted_boardings=("predicted_boardings", "sum"),
            predicted_exiting=("predicted_exiting", "sum"),
            baseline_runs=("baseline_runs", "sum"),
            avg_vehicle_capacity=("avg_vehicle_capacity", "mean"),
        )
    )
    results: list[KPIResult] = []
    for line, line_hourly in hourly.groupby("line", sort=True):
        result = _calculate_annual_from_hourly(
            line_hourly,
            year=year,
            label=f"Linie {int(line)} KPI-Jahresstandard",
            bus_hourly_cost=bus_hourly_cost,
            line=int(line),
            period=f"Jahr {year} | Linie {int(line)}",
        )
        if result.hours > 0:
            results.append(result)
    return results


def aggregate_kpis(results: list[KPIResult], *, label: str, selected_day: date, period: str = "") -> KPIResult:
    valid = [item for item in results if item.hours > 0]
    if not valid:
        return _empty_result(label, None, selected_day)

    demand = sum(item.demand for item in valid)
    wvv_capacity = sum(item.wvv_capacity for item in valid)
    adaptive_capacity = sum(item.adaptive_capacity for item in valid)
    wvv_bus_hours = sum(item.wvv_bus_hours for item in valid)
    adaptive_bus_hours = sum(item.adaptive_bus_hours for item in valid)
    wvv_wait_minutes = _result_weighted_average(valid, "wvv_wait_minutes")
    adaptive_wait_minutes = _result_weighted_average(valid, "adaptive_wait_minutes")
    wvv_overload = sum(item.wvv_overload for item in valid)
    adaptive_overload = sum(item.adaptive_overload for item in valid)
    wvv_underload = sum(item.wvv_underload for item in valid)
    adaptive_underload = sum(item.adaptive_underload for item in valid)
    wvv_unserved = sum(item.wvv_unserved for item in valid)
    adaptive_unserved = sum(item.adaptive_unserved for item in valid)

    return KPIResult(
        label=label,
        line=None,
        selected_day=selected_day,
        hours=sum(item.hours for item in valid),
        demand=demand,
        wvv_runs=sum(item.wvv_runs for item in valid),
        adaptive_runs=sum(item.adaptive_runs for item in valid),
        wvv_capacity=wvv_capacity,
        adaptive_capacity=adaptive_capacity,
        wvv_bus_hours=wvv_bus_hours,
        adaptive_bus_hours=adaptive_bus_hours,
        wvv_wait_minutes=wvv_wait_minutes,
        adaptive_wait_minutes=adaptive_wait_minutes,
        wvv_cost=sum(item.wvv_cost for item in valid),
        adaptive_cost=sum(item.adaptive_cost for item in valid),
        wvv_coverage=_safe_ratio(demand - wvv_unserved, demand),
        adaptive_coverage=_safe_ratio(demand - adaptive_unserved, demand),
        wvv_peak_coverage=min(item.wvv_peak_coverage for item in valid),
        adaptive_peak_coverage=min(item.adaptive_peak_coverage for item in valid),
        wvv_utilization=_safe_ratio(demand, wvv_capacity),
        adaptive_utilization=_safe_ratio(demand, adaptive_capacity),
        wvv_overload=wvv_overload,
        adaptive_overload=adaptive_overload,
        wvv_underload=wvv_underload,
        adaptive_underload=adaptive_underload,
        wvv_unserved=wvv_unserved,
        adaptive_unserved=adaptive_unserved,
        period=period,
    )


def format_kpi_report(result: KPIResult) -> str:
    if result.hours <= 0:
        return "Keine KPI-Daten berechnet.\n\nTrainiere zuerst ein Prediction-Modell oder waehle eine andere Linie."

    rows = [
        ("Demand Coverage", _percent(result.wvv_coverage), _percent(result.adaptive_coverage), _delta_percent(result.adaptive_coverage - result.wvv_coverage)),
        ("Peak Demand-Abdeckung", _percent(result.wvv_peak_coverage), _percent(result.adaptive_peak_coverage), _delta_percent(result.adaptive_peak_coverage - result.wvv_peak_coverage)),
        ("Auslastung", _percent(result.wvv_utilization), _percent(result.adaptive_utilization), _delta_percent(result.adaptive_utilization - result.wvv_utilization)),
        ("Überlastung", _number(result.wvv_overload), _number(result.adaptive_overload), _number(result.adaptive_overload - result.wvv_overload)),
        ("Unterauslastung", _number(result.wvv_underload), _number(result.adaptive_underload), _number(result.adaptive_underload - result.wvv_underload)),
        ("Nicht bediente Nachfrage", _number(result.wvv_unserved), _number(result.adaptive_unserved), _number(result.adaptive_unserved - result.wvv_unserved)),
        ("Ø Wartezeit", _minutes(result.wvv_wait_minutes), _minutes(result.adaptive_wait_minutes), _delta_minutes(result.adaptive_wait_minutes - result.wvv_wait_minutes)),
        ("Fahrten/Kurse", _number(result.wvv_runs), _number(result.adaptive_runs), _number(result.run_delta)),
        ("Busstunden geschätzt", _decimal(result.wvv_bus_hours), _decimal(result.adaptive_bus_hours), _decimal(result.adaptive_bus_hours - result.wvv_bus_hours)),
        ("Kosten", _euro(result.wvv_cost), _euro(result.adaptive_cost), _euro(result.cost_delta)),
    ]
    period_label = result.period or result.selected_day.isoformat()
    unit_label = "Stundenprofile" if result.period else "Stunden"
    width = max(len(row[0]) for row in rows)
    lines = [
        result.label,
        f"Zeitraum: {period_label} | {unit_label}: {result.hours} | Nachfrage: {_number(result.demand)}",
        f"Definition: Überlastung > {COMFORT_UTILIZATION_LIMIT:.0%} Komfortauslastung | "
        f"Unterauslastung < {PRODUCTIVE_UTILIZATION_FLOOR:.0%} produktive Mindestlast | "
        "nicht bedient > 100% Kapazität",
        "",
        f"{'KPI':<{width}}  {'WVV':>13}  {'Adaptiv':>13}  {'Delta':>13}",
        "-" * (width + 45),
    ]
    for name, wvv, adaptive, delta in rows:
        lines.append(f"{name:<{width}}  {wvv:>13}  {adaptive:>13}  {delta:>13}")
    return "\n".join(lines)


def kpi_chart_frame(result: KPIResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"KPI": "Demand Coverage", "WVV": result.wvv_coverage * 100, "Adaptiv": result.adaptive_coverage * 100},
            {"KPI": "Peak-Abdeckung", "WVV": result.wvv_peak_coverage * 100, "Adaptiv": result.adaptive_peak_coverage * 100},
            {"KPI": "Auslastung", "WVV": result.wvv_utilization * 100, "Adaptiv": result.adaptive_utilization * 100},
        ]
    )


def kpi_problem_frame(result: KPIResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"KPI": "Überlastung", "WVV": result.wvv_overload, "Adaptiv": result.adaptive_overload},
            {"KPI": "Unterauslastung", "WVV": result.wvv_underload, "Adaptiv": result.adaptive_underload},
            {"KPI": "Nicht bedient", "WVV": result.wvv_unserved, "Adaptiv": result.adaptive_unserved},
        ]
    )


def save_cost_setting(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'{{"bus_hourly_cost_eur": {float(value):.2f}}}\n', encoding="utf-8")


def load_cost_setting(path: Path, default: float = DEFAULT_BUS_HOURLY_COST_EUR) -> float:
    if not path.exists():
        return default
    try:
        raw = path.read_text(encoding="utf-8")
        match = re.search(r'"bus_hourly_cost_eur"\s*:\s*([0-9]+(?:[.,][0-9]+)?)', raw)
        if not match:
            return default
        return max(0.0, float(match.group(1).replace(",", ".")))
    except (OSError, ValueError):
        return default


def _normalize_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    if schedule.empty:
        return pd.DataFrame()
    frame = schedule.copy()
    required = ["hour", "baseline_runs", "recommended_runs", "avg_vehicle_capacity"]
    for column in required:
        if column not in frame:
            frame[column] = 0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    if "predicted_boardings" in frame:
        demand = pd.to_numeric(frame["predicted_boardings"], errors="coerce").fillna(0)
    elif "predicted_demand" in frame:
        demand = pd.to_numeric(frame["predicted_demand"], errors="coerce").fillna(0)
    else:
        demand = pd.Series(0.0, index=frame.index)
    if demand.sum() <= 0 and "predicted_demand" in frame:
        demand = pd.to_numeric(frame["predicted_demand"], errors="coerce").fillna(0)
    frame["kpi_demand"] = demand
    frame["hour"] = frame["hour"].astype(int)
    return frame


def _hourly_from_raw(raw: pd.DataFrame, year: int) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    frame = raw.copy()
    if "date" not in frame or frame["date"].isna().all():
        frame["date"] = frame.get("report_date")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    frame = frame[pd.to_datetime(frame["date"].astype(str), errors="coerce").dt.year == int(year)].copy()
    if frame.empty:
        return pd.DataFrame()

    for column in ["line", "hour", "passenger_boarding", "passenger_exiting"]:
        if column not in frame:
            frame[column] = 0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    if "journey" not in frame:
        frame["journey"] = "unknown"
    if "vehicle_type" not in frame:
        frame["vehicle_type"] = "Unknown"
    frame["vehicle_type"] = frame["vehicle_type"].fillna("Unknown").astype(str)
    frame["capacity"] = frame["vehicle_type"].map(_vehicle_capacity).fillna(_vehicle_capacity("Unknown"))
    return (
        frame.groupby(["date", "line", "hour"], as_index=False)
        .agg(
            predicted_boardings=("passenger_boarding", "sum"),
            predicted_exiting=("passenger_exiting", "sum"),
            baseline_runs=("journey", "nunique"),
            avg_vehicle_capacity=("capacity", "mean"),
        )
        .sort_values(["date", "line", "hour"])
    )


def _calculate_annual_from_hourly(
    hourly: pd.DataFrame,
    *,
    year: int,
    label: str,
    bus_hourly_cost: float,
    line: int | None = None,
    period: str | None = None,
) -> KPIResult:
    if hourly.empty:
        return _empty_result(label, None, date(year, 1, 1), period=f"Jahr {year} | alle Linien")
    hourly = hourly.copy()
    hourly["predicted_demand"] = hourly["predicted_boardings"] + hourly["predicted_exiting"]
    hourly["recommended_runs"] = hourly.apply(
        lambda row: _annual_adaptive_runs(
            demand=float(row["predicted_boardings"]),
            baseline_runs=float(row["baseline_runs"]),
            avg_capacity=float(row["avg_vehicle_capacity"]),
            cost_per_bus_hour=bus_hourly_cost,
            line=row.get("line", line),
            hour=row.get("hour"),
        ),
        axis=1,
    )
    return calculate_line_kpis(
        hourly,
        line=line,
        label=label,
        selected_day=date(year, 1, 1),
        bus_hourly_cost=bus_hourly_cost,
        period=period or f"Jahr {year} | alle Linien",
    )


def _side_metrics(hours: pd.Series, demand: pd.Series, capacity: pd.Series) -> dict[str, float]:
    demand_sum = float(demand.sum())
    capacity_sum = float(capacity.sum())
    hard_capacity = capacity.clip(lower=0)
    comfort_capacity = hard_capacity * COMFORT_UTILIZATION_LIMIT
    productive_floor = hard_capacity * PRODUCTIVE_UTILIZATION_FLOOR
    served = demand.combine(hard_capacity, min)
    unserved = (demand - hard_capacity).clip(lower=0)
    overload = (demand - comfort_capacity).clip(lower=0)
    underload = (productive_floor - demand).clip(lower=0)
    if demand_sum > 0:
        peak_index = demand.idxmax()
        peak_hour = int(hours.loc[peak_index])
        peak_mask = hours.map(lambda hour: _hour_distance(int(hour), peak_hour) <= 1)
        peak_demand = float(demand[peak_mask].sum())
        peak_capacity = float(hard_capacity[peak_mask].sum())
        peak_coverage = _safe_ratio(min(peak_demand, peak_capacity), peak_demand)
    else:
        peak_coverage = 1.0
    return {
        "coverage": _safe_ratio(float(served.sum()), demand_sum),
        "peak_coverage": peak_coverage,
        "utilization": _safe_ratio(demand_sum, capacity_sum),
        "overload": float(overload.sum()),
        "underload": float(underload.sum()),
        "unserved": float(unserved.sum()),
    }


def _weighted_wait_minutes(demand: pd.Series, runs: pd.Series) -> float:
    demand_clean = pd.to_numeric(demand, errors="coerce").fillna(0).clip(lower=0)
    runs_clean = pd.to_numeric(runs, errors="coerce").fillna(0).clip(lower=0)
    if float(demand_clean.sum()) <= 0:
        active_runs = runs_clean[runs_clean > 0]
        if active_runs.empty:
            return 0.0
        return float((60.0 / active_runs / 2.0).clip(upper=60.0).mean())
    wait = pd.Series(60.0, index=demand_clean.index, dtype=float)
    active = runs_clean > 0
    wait.loc[active] = (60.0 / runs_clean.loc[active] / 2.0).clip(upper=60.0)
    return float((wait * demand_clean).sum() / demand_clean.sum())


def _result_weighted_average(results: list[KPIResult], attr: str) -> float:
    demand_sum = sum(max(0.0, item.demand) for item in results)
    if demand_sum <= 0:
        return sum(float(getattr(item, attr)) for item in results) / max(len(results), 1)
    return sum(float(getattr(item, attr)) * max(0.0, item.demand) for item in results) / demand_sum


def _empty_result(label: str, line: int | None, selected_day: date, period: str = "") -> KPIResult:
    return KPIResult(
        label=label,
        line=line,
        selected_day=selected_day,
        hours=0,
        demand=0.0,
        wvv_runs=0.0,
        adaptive_runs=0.0,
        wvv_capacity=0.0,
        adaptive_capacity=0.0,
        wvv_bus_hours=0.0,
        adaptive_bus_hours=0.0,
        wvv_wait_minutes=0.0,
        adaptive_wait_minutes=0.0,
        wvv_cost=0.0,
        adaptive_cost=0.0,
        wvv_coverage=0.0,
        adaptive_coverage=0.0,
        wvv_peak_coverage=0.0,
        adaptive_peak_coverage=0.0,
        wvv_utilization=0.0,
        adaptive_utilization=0.0,
        wvv_overload=0.0,
        adaptive_overload=0.0,
        wvv_underload=0.0,
        adaptive_underload=0.0,
        wvv_unserved=0.0,
        adaptive_unserved=0.0,
        period=period,
    )


def _hour_distance(left: int, right: int) -> int:
    distance = abs((int(left) % 24) - (int(right) % 24))
    return min(distance, 24 - distance)


def _vehicle_capacity(vehicle_type: object) -> float:
    capacities = {
        "SKOM": 80.0,
        "GKOM": 140.0,
        "GTE": 180.0,
        "GTN": 170.0,
        "Unknown": 90.0,
    }
    return float(capacities.get(str(vehicle_type), capacities["Unknown"]))


def _row_line(row: pd.Series, fallback: int | None) -> int | float | str | None:
    value = row.get("line", fallback)
    if pd.isna(value):
        return fallback
    return value


def _annual_adaptive_runs(
    *,
    demand: float,
    baseline_runs: float,
    avg_capacity: float,
    cost_per_bus_hour: float,
    line: int | float | str | None = None,
    hour: int | float | None = None,
) -> int:
    return constrained_adaptive_runs(
        demand=demand,
        baseline_runs=baseline_runs,
        avg_capacity=avg_capacity,
        cost_per_bus_hour=cost_per_bus_hour,
        default_cost_per_bus_hour=DEFAULT_BUS_HOURLY_COST_EUR,
        line=line,
        hour=hour,
        allow_new_service=True,
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 1.0 if numerator <= 0 else 0.0
    value = numerator / denominator
    return float(min(max(value, 0.0), 9.99))


def _percent(value: float) -> str:
    if not math.isfinite(value):
        return "-"
    return f"{value * 100:,.1f}%".replace(",", "X").replace(".", ",").replace("X", ".")


def _delta_percent(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return sign + _percent(value)


def _number(value: float) -> str:
    if not math.isfinite(value):
        return "-"
    return f"{value:,.0f}".replace(",", ".")


def _decimal(value: float) -> str:
    if not math.isfinite(value):
        return "-"
    return f"{value:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _minutes(value: float) -> str:
    if not math.isfinite(value):
        return "-"
    return f"{_decimal(value)} min"


def _delta_minutes(value: float) -> str:
    if not math.isfinite(value):
        return "-"
    sign = "+" if value >= 0 else ""
    return sign + _minutes(value)


def _euro(value: float) -> str:
    return f"{_number(value)} EUR"
