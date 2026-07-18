from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq


YEAR = 2025
REQUIRED_SCHEMA_COLUMN = "vehicle_station"
PASSENGER_COLUMNS = [
    "passenger_boarding",
    "passenger_exiting",
    "passenger_change",
    "occupancy_departure",
    "vehicle_utilization",
    "passenger_boarding_measured",
    "passenger_exiting_measured",
    "passenger_kilometer",
]
NON_NEGATIVE_COLUMNS = [
    "quality_factor",
    "stop_sequence",
    "station_number",
    "cumsum_distance_plan",
    "cumsum_travel_time_actual",
    *PASSENGER_COLUMNS,
]
MODEL_READY_COLUMNS = [
    "date",
    "hour",
    "departure_minute_of_day",
    "weekday",
    "month",
    "day_of_year",
    "calendar_week",
    "is_weekend",
    "line",
    "line_route",
    "direction",
    "route",
    "day_type",
    "stop_sequence",
    "station_number",
    "station",
    "station_key",
    "vehicle_type",
    "passenger_boarding",
    "passenger_exiting",
    "passenger_change",
    "public_holiday",
    "nationwide",
    "school_holiday",
    "lecture_period_jmu",
    "lecture_period_thws",
    "event_day",
    "event_count",
    "concert_day",
    "concert_event_count",
    "total_event_duration_hours",
    "max_event_duration_hours",
    "event_hour",
    "concert_hour",
    "verkaufsoffener_sonntag",
]
MODEL_READY_STREET_COLUMNS = [
    "street_pedestrians_total",
    "street_pedestrians_mean",
    "street_pedestrians_towards_center",
    "street_pedestrians_away_center",
    "street_temperature_mean",
    "street_measurements",
]


@dataclass
class ProcessStats:
    source_file: str
    output_file: str
    rows_in: int
    rows_2025: int
    rows_out: int
    duplicate_rows_removed: int
    negative_values_fixed: int
    missing_values_filled: int
    model_ready_file: str = ""
    model_ready_rows: int = 0
    skipped: bool = False
    reason: str = ""


def parse_args() -> argparse.Namespace:
    app_dir = Path(__file__).resolve().parents[1]
    default_input = app_dir / "downloads" / "wvv-pjs-2026" / "full_api_data"
    default_additional = app_dir / "downloads" / "wvv-pjs-2026" / "Additional Data"
    default_output = app_dir / "downloads" / "wvv-pjs-2026" / "full_api_data_enriched_2025"
    default_model_ready = app_dir / "downloads" / "wvv-pjs-2026" / "model_ready_2025"

    parser = argparse.ArgumentParser(
        description=(
            "Bereinigt WVV-Parquet-Dateien fuer 2025, fuegt Kalender-/Kontextfeatures "
            "hinzu und schreibt neue Parquet-Dateien."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=default_input)
    parser.add_argument("--additional-dir", type=Path, default=default_additional)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--model-ready-dir", type=Path, default=default_model_ready)
    parser.add_argument("--no-model-ready", action="store_true", help="Keine reduzierte ML-Version schreiben.")
    parser.add_argument("--year", type=int, default=YEAR)
    parser.add_argument("--limit-files", type=int, default=0, help="Nur die ersten N Dateien verarbeiten; 0 = alle.")
    parser.add_argument("--include-streets", action="store_true", help="Stundenweise Innenstadt-Fussgaengerfeatures mergen.")
    parser.add_argument("--overwrite", action="store_true", help="Bereits erzeugte Dateien ueberschreiben.")
    return parser.parse_args()


def date_range_frame(year: int) -> pd.DataFrame:
    dates = pd.date_range(date(year, 1, 1), date(year, 12, 31), freq="D")
    frame = pd.DataFrame({"date": dates.date})
    frame["weekday"] = pd.to_datetime(frame["date"].astype(str)).dt.weekday
    frame["is_weekend"] = frame["weekday"].isin([5, 6]).astype("int8")
    frame["month"] = pd.to_datetime(frame["date"].astype(str)).dt.month.astype("int8")
    return frame


def read_csv_flexible(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.ParserError:
        return pd.read_csv(path, sep=";", **kwargs)


def ensure_date_column(frame: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return frame
    frame = frame.copy()
    frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.date
    return frame.dropna(subset=[column])


def build_daily_context(additional_dir: Path, year: int) -> pd.DataFrame:
    context = date_range_frame(year)

    public_holidays = ensure_date_column(read_csv_flexible(additional_dir / "bavarian_public_holidays_daily.csv"))
    context = merge_daily(context, public_holidays, ["public_holiday", "nationwide"])

    school_holidays = ensure_date_column(read_csv_flexible(additional_dir / "bavarian_school_holidays_daily.csv"))
    context = merge_daily(context, school_holidays, ["school_holiday"])

    lectures_daily = ensure_date_column(read_csv_flexible(additional_dir / "lectures_daily.csv"))
    context = merge_daily(context, lectures_daily, ["lecture_period_jmu"])

    lectures = read_csv_flexible(additional_dir / "lectures.csv")
    context = merge_lecture_ranges(context, lectures)

    events, hourly_events = build_event_context(additional_dir, year)
    context = context.merge(events, on="date", how="left")

    open_sundays = ensure_date_column(read_csv_flexible(additional_dir / "verkaufsoffene_sonntage.csv"))
    if not open_sundays.empty:
        open_sundays = open_sundays[open_sundays["date"].map(lambda value: value.year == year)].copy()
        open_sundays["verkaufsoffener_sonntag"] = 1
        open_sundays["verkaufsoffener_sonntag_name"] = open_sundays.get("name", "").fillna("").astype(str)
        context = context.merge(
            open_sundays[["date", "verkaufsoffener_sonntag", "verkaufsoffener_sonntag_name"]],
            on="date",
            how="left",
        )

    numeric_fill = {
        "public_holiday": 0,
        "nationwide": 0,
        "school_holiday": 0,
        "lecture_period_jmu": 0,
        "lecture_period_thws": 0,
        "event_day": 0,
        "event_count": 0,
        "concert_day": 0,
        "concert_event_count": 0,
        "total_event_duration_hours": 0.0,
        "max_event_duration_hours": 0.0,
        "verkaufsoffener_sonntag": 0,
    }
    for column, fill_value in numeric_fill.items():
        if column not in context.columns:
            context[column] = fill_value
        context[column] = context[column].fillna(fill_value)
    if "verkaufsoffener_sonntag_name" not in context.columns:
        context["verkaufsoffener_sonntag_name"] = ""
    context["verkaufsoffener_sonntag_name"] = context["verkaufsoffener_sonntag_name"].fillna("")

    int_columns = [
        "public_holiday",
        "nationwide",
        "school_holiday",
        "lecture_period_jmu",
        "lecture_period_thws",
        "event_day",
        "event_count",
        "concert_day",
        "concert_event_count",
        "verkaufsoffener_sonntag",
    ]
    for column in int_columns:
        context[column] = context[column].astype("int8" if context[column].max() <= 127 else "int16")

    return context, hourly_events


def merge_daily(base: pd.DataFrame, other: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if other.empty:
        for column in columns:
            if column not in base.columns:
                base[column] = 0
        return base
    available = ["date", *[column for column in columns if column in other.columns]]
    merged = base.merge(other[available], on="date", how="left")
    return merged


def merge_lecture_ranges(context: pd.DataFrame, lectures: pd.DataFrame) -> pd.DataFrame:
    if lectures.empty or not {"start", "end"}.issubset(lectures.columns):
        if "lecture_period_thws" not in context.columns:
            context["lecture_period_thws"] = 0
        return context

    frame = context.copy()
    frame["lecture_period_thws"] = 0
    if "lecture_period_jmu" not in frame.columns:
        frame["lecture_period_jmu"] = 0

    lectures = lectures.copy()
    lectures["start"] = pd.to_datetime(lectures["start"], errors="coerce").dt.date
    lectures["end"] = pd.to_datetime(lectures["end"], errors="coerce").dt.date
    lectures = lectures.dropna(subset=["start", "end"])
    for _, row in lectures.iterrows():
        mask = (frame["date"] >= row["start"]) & (frame["date"] <= row["end"])
        if int(row.get("jmu", 0)) == 1:
            frame.loc[mask, "lecture_period_jmu"] = 1
        if int(row.get("thws", 0)) == 1:
            frame.loc[mask, "lecture_period_thws"] = 1
    return frame


def build_event_context(additional_dir: Path, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = date_range_frame(year)[["date"]]
    events = read_csv_flexible(additional_dir / "events.csv")
    if events.empty or not {"start", "end"}.issubset(events.columns):
        empty_hourly = pd.DataFrame({"date": [], "hour": [], "event_hour": [], "concert_hour": []})
        for column in [
            "event_day",
            "event_count",
            "concert_day",
            "concert_event_count",
            "total_event_duration_hours",
            "max_event_duration_hours",
        ]:
            base[column] = 0
        return base, empty_hourly

    events = events.copy()
    events["start"] = pd.to_datetime(events["start"], errors="coerce")
    events["end"] = pd.to_datetime(events["end"], errors="coerce")
    events = events.dropna(subset=["start", "end"])
    events = events[(events["start"].dt.year <= year) & (events["end"].dt.year >= year)].copy()
    if events.empty:
        empty_hourly = pd.DataFrame({"date": [], "hour": [], "event_hour": [], "concert_hour": []})
        return build_event_context_from_empty(base), empty_hourly

    events["date"] = events["start"].dt.date
    events["duration_hours"] = ((events["end"] - events["start"]).dt.total_seconds() / 3600).clip(lower=0)
    events["concert"] = pd.to_numeric(events.get("concert", 0), errors="coerce").fillna(0).astype(int)
    daily = (
        events.groupby("date", as_index=False)
        .agg(
            event_count=("name", "count"),
            concert_event_count=("concert", "sum"),
            total_event_duration_hours=("duration_hours", "sum"),
            max_event_duration_hours=("duration_hours", "max"),
        )
    )
    daily["event_day"] = (daily["event_count"] > 0).astype("int8")
    daily["concert_day"] = (daily["concert_event_count"] > 0).astype("int8")
    daily = base.merge(daily, on="date", how="left")

    hourly_rows = []
    for _, row in events.iterrows():
        start = max(row["start"], pd.Timestamp(year=year, month=1, day=1))
        end = min(row["end"], pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, second=59))
        if start > end:
            continue
        for timestamp in pd.date_range(start.floor("h"), end.ceil("h"), freq="h"):
            if timestamp.year != year:
                continue
            hourly_rows.append(
                {
                    "date": timestamp.date(),
                    "hour": int(timestamp.hour),
                    "event_hour": 1,
                    "concert_hour": int(row["concert"] > 0),
                }
            )
    hourly = pd.DataFrame(hourly_rows)
    if hourly.empty:
        hourly = pd.DataFrame({"date": [], "hour": [], "event_hour": [], "concert_hour": []})
    else:
        hourly = (
            hourly.groupby(["date", "hour"], as_index=False)
            .agg(event_hour=("event_hour", "max"), concert_hour=("concert_hour", "max"))
        )
    return daily, hourly


def build_event_context_from_empty(base: pd.DataFrame) -> pd.DataFrame:
    frame = base.copy()
    for column in [
        "event_day",
        "event_count",
        "concert_day",
        "concert_event_count",
        "total_event_duration_hours",
        "max_event_duration_hours",
    ]:
        frame[column] = 0
    return frame


def build_street_hourly_context(additional_dir: Path, year: int) -> pd.DataFrame:
    path = additional_dir / "dataAllStreets.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        streets = pd.read_csv(path, sep=";")
    except Exception:
        return pd.DataFrame()
    if streets.empty or not {"date", "hour"}.issubset(streets.columns):
        return pd.DataFrame()

    streets["date"] = pd.to_datetime(streets["date"], errors="coerce").dt.date
    streets["hour"] = pd.to_numeric(streets["hour"], errors="coerce")
    streets = streets.dropna(subset=["date", "hour"])
    streets = streets[streets["date"].map(lambda value: value.year == year)].copy()
    if streets.empty:
        return pd.DataFrame()
    streets["hour"] = streets["hour"].astype(int)
    for column in [
        "pedestrians_count",
        "towards_citycenter_pedestrians_count",
        "awayfrom_citycenter_pedestrians_count",
        "temperature",
    ]:
        if column in streets.columns:
            streets[column] = pd.to_numeric(streets[column], errors="coerce")

    agg_map = {}
    if "pedestrians_count" in streets.columns:
        agg_map["street_pedestrians_total"] = ("pedestrians_count", "sum")
        agg_map["street_pedestrians_mean"] = ("pedestrians_count", "mean")
    if "towards_citycenter_pedestrians_count" in streets.columns:
        agg_map["street_pedestrians_towards_center"] = ("towards_citycenter_pedestrians_count", "sum")
    if "awayfrom_citycenter_pedestrians_count" in streets.columns:
        agg_map["street_pedestrians_away_center"] = ("awayfrom_citycenter_pedestrians_count", "sum")
    if "temperature" in streets.columns:
        agg_map["street_temperature_mean"] = ("temperature", "mean")
    if not agg_map:
        return pd.DataFrame()
    hourly = streets.groupby(["date", "hour"], as_index=False).agg(**agg_map)
    hourly["street_measurements"] = streets.groupby(["date", "hour"]).size().to_numpy()
    return hourly


def parquet_has_required_schema(path: Path) -> bool:
    try:
        return REQUIRED_SCHEMA_COLUMN in pq.ParquetFile(path).schema.names
    except Exception:
        return False


def filter_2025(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    frame = frame.copy()
    date_source = "main_date_day" if "main_date_day" in frame.columns else "report_date"
    frame["date"] = pd.to_datetime(frame[date_source], errors="coerce").dt.date
    return frame[frame["date"].map(lambda value: pd.notna(value) and value.year == year)].copy()


def add_hour_column(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    candidates = ["departure_plan_station", "departure_journey", "departure_plan_journey", "report_date"]
    parsed_source = None
    for column in candidates:
        if column not in frame.columns:
            continue
        parsed = pd.to_datetime(frame[column], errors="coerce")
        if parsed.notna().any():
            parsed_source = parsed
            frame["hour"] = parsed.dt.hour
            break
    if "hour" not in frame.columns:
        frame["hour"] = pd.NA
    frame["hour"] = pd.to_numeric(frame["hour"], errors="coerce").fillna(0).astype("int8")
    if parsed_source is not None:
        minute_of_day = parsed_source.dt.hour * 60 + parsed_source.dt.minute
        frame["departure_minute_of_day"] = pd.to_numeric(minute_of_day, errors="coerce")
    else:
        frame["departure_minute_of_day"] = pd.NA
    frame["departure_minute_of_day"] = (
        frame["departure_minute_of_day"]
        .fillna(frame["hour"].astype(int) * 60)
        .clip(lower=0, upper=1439)
        .round()
        .astype("int16")
    )
    return frame


def clean_and_impute(frame: pd.DataFrame) -> tuple[pd.DataFrame, int, int, int]:
    rows_before = len(frame)
    missing_before = int(frame.isna().sum().sum())

    if "stop_event_id" in frame.columns:
        frame = frame.drop_duplicates(subset=["stop_event_id"])
    else:
        frame = frame.drop_duplicates()
    duplicate_rows_removed = rows_before - len(frame)

    negative_values_fixed = 0
    for column in NON_NEGATIVE_COLUMNS:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        negative_mask = values < 0
        negative_values_fixed += int(negative_mask.sum())
        values = values.mask(negative_mask)
        frame[column] = values

    for column in ["passenger_boarding", "passenger_exiting"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).round().astype("int32")
    if {"passenger_boarding", "passenger_exiting"}.issubset(frame.columns):
        frame["passenger_change"] = frame["passenger_boarding"] - frame["passenger_exiting"]

    if "occupancy_departure" in frame.columns:
        frame["occupancy_departure"] = pd.to_numeric(frame["occupancy_departure"], errors="coerce")
        frame = impute_occupancy(frame)

    for column in ["passenger_boarding_measured", "passenger_exiting_measured"]:
        if column in frame.columns:
            source = "passenger_boarding" if "boarding" in column else "passenger_exiting"
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(frame.get(source, 0)).round().astype("int32")

    if "vehicle_utilization" in frame.columns:
        util = pd.to_numeric(frame["vehicle_utilization"], errors="coerce")
        frame["vehicle_utilization"] = util.mask(util < 0).clip(upper=2.0)
        frame["vehicle_utilization"] = frame["vehicle_utilization"].fillna(
            frame.groupby("line")["vehicle_utilization"].transform("median")
        ).fillna(frame["vehicle_utilization"].median()).fillna(0)

    numeric_impute = [
        "quality_factor",
        "stop_sequence",
        "station_number",
        "cumsum_distance_plan",
        "cumsum_travel_time_actual",
        "passenger_kilometer",
    ]
    for column in numeric_impute:
        if column not in frame.columns:
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if "line" in frame.columns:
            frame[column] = frame[column].fillna(frame.groupby("line")[column].transform("median"))
        frame[column] = frame[column].fillna(frame[column].median()).fillna(0)

    datetime_columns = [
        "report_date",
        "departure_plan_journey",
        "departure_journey",
        "departure_hour_station",
        "departure_plan_station",
        "arrival_plan_door",
        "departure_plan_door",
        "arrival_plan_stop",
        "departure_plan_stop",
    ]
    for column in datetime_columns:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    if {"departure_plan_station", "departure_plan_journey"}.issubset(frame.columns):
        frame["departure_plan_station"] = frame["departure_plan_station"].fillna(frame["departure_plan_journey"])
    if {"departure_journey", "departure_plan_journey"}.issubset(frame.columns):
        frame["departure_journey"] = frame["departure_journey"].fillna(frame["departure_plan_journey"])

    for column in frame.select_dtypes(include=["object", "string"]).columns:
        if column == "verkaufsoffener_sonntag_name":
            frame[column] = frame[column].fillna("")
        else:
            frame[column] = frame[column].fillna("Unknown")

    if "vehicle_station" in frame.columns:
        frame["vehicle_station"] = frame.groupby("journey")["vehicle_station"].ffill().bfill().fillna("Unknown")
    if "vehicle_type" in frame.columns:
        vehicle_type_by_vehicle = frame.dropna(subset=["vehicle_station", "vehicle_type"]).groupby("vehicle_station")["vehicle_type"]
        mode_by_vehicle = vehicle_type_by_vehicle.transform(lambda values: values.mode().iloc[0] if not values.mode().empty else "Unknown")
        frame["vehicle_type"] = frame["vehicle_type"].fillna(mode_by_vehicle).fillna("Unknown")

    missing_after = int(frame.isna().sum().sum())
    missing_values_filled = max(missing_before - missing_after, 0)
    return frame, duplicate_rows_removed, negative_values_fixed, missing_values_filled


def impute_occupancy(frame: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [column for column in ["unique_journey", "journey", "stop_sequence"] if column in frame.columns]
    frame = frame.sort_values(sort_columns) if sort_columns else frame
    if {"passenger_change", "journey"}.issubset(frame.columns):
        computed = frame.groupby("journey")["passenger_change"].cumsum().clip(lower=0)
        frame["occupancy_departure"] = frame["occupancy_departure"].fillna(computed)
    group_key = "unique_journey" if "unique_journey" in frame.columns else "journey"
    if group_key in frame.columns:
        frame["occupancy_departure"] = frame.groupby(group_key)["occupancy_departure"].ffill().bfill()
    frame["occupancy_departure"] = frame["occupancy_departure"].fillna(0).round().astype("int32")
    return frame


def add_context(
    frame: pd.DataFrame,
    daily_context: pd.DataFrame,
    hourly_events: pd.DataFrame,
    street_hourly: pd.DataFrame,
) -> pd.DataFrame:
    if "weekday" in frame.columns and "weekday_name" not in frame.columns:
        frame = frame.rename(columns={"weekday": "weekday_name"})
    frame = frame.merge(daily_context, on="date", how="left", suffixes=("", "_context"))
    if not hourly_events.empty:
        frame = frame.merge(hourly_events, on=["date", "hour"], how="left")
    else:
        frame["event_hour"] = 0
        frame["concert_hour"] = 0
    if not street_hourly.empty:
        frame = frame.merge(street_hourly, on=["date", "hour"], how="left")

    fill_zero_columns = [
        column
        for column in frame.columns
        if column.startswith(("event_", "concert_", "street_", "lecture_", "public_", "school_"))
        or column in ["nationwide", "verkaufsoffener_sonntag", "event_hour", "concert_hour"]
    ]
    for column in fill_zero_columns:
        if column == "verkaufsoffener_sonntag_name":
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    if "verkaufsoffener_sonntag_name" in frame.columns:
        frame["verkaufsoffener_sonntag_name"] = frame["verkaufsoffener_sonntag_name"].fillna("")
    return frame


def output_name(source: Path, output_dir: Path, year: int) -> Path:
    stem = re.sub(r"\.parquet$", "", source.name)
    return output_dir / f"{stem}_clean_context_{year}.parquet"


def model_ready_output_name(source: Path, model_ready_dir: Path, year: int) -> Path:
    stem = re.sub(r"\.parquet$", "", source.name)
    return model_ready_dir / f"{stem}_model_ready_{year}.parquet"


def normalize_key(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.strip().lower()
    text = (
        text.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
    )
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def build_model_ready_frame(enriched: pd.DataFrame, year: int) -> pd.DataFrame:
    frame = enriched.copy()
    date_values = pd.to_datetime(frame["date"].astype(str), errors="coerce")
    frame["day_of_year"] = date_values.dt.dayofyear.fillna(0).astype("int16")
    frame["calendar_week"] = date_values.dt.isocalendar().week.fillna(0).astype("int16")
    if "station" in frame.columns:
        frame["station_key"] = frame["station"].map(normalize_key)
    else:
        frame["station_key"] = "unknown"

    selected_columns = [
        column
        for column in [*MODEL_READY_COLUMNS, *MODEL_READY_STREET_COLUMNS]
        if column in frame.columns
    ]
    model = frame[selected_columns].copy()

    categorical_columns = [
        "station",
        "station_key",
        "direction",
        "route",
        "day_type",
        "vehicle_type",
    ]
    for column in categorical_columns:
        if column in model.columns:
            model[column] = model[column].fillna("Unknown").astype("string")

    numeric_columns = [column for column in model.columns if column not in categorical_columns and column != "date"]
    for column in numeric_columns:
        model[column] = pd.to_numeric(model[column], errors="coerce")
        if column in {
            "hour",
            "weekday",
            "month",
            "is_weekend",
            "public_holiday",
            "nationwide",
            "school_holiday",
            "lecture_period_jmu",
            "lecture_period_thws",
            "event_day",
            "event_count",
            "concert_day",
            "concert_event_count",
            "event_hour",
            "concert_hour",
            "verkaufsoffener_sonntag",
        }:
            model[column] = model[column].fillna(0).round().astype("int16")
        elif column in {"line", "line_route", "stop_sequence", "station_number", "departure_minute_of_day", "day_of_year", "calendar_week"}:
            model[column] = model[column].fillna(0).round().astype("int32")
        elif column in {"passenger_boarding", "passenger_exiting", "passenger_change"}:
            model[column] = model[column].fillna(0).round().astype("int32")
        else:
            model[column] = model[column].fillna(0).astype("float32")

    # Strong guarantee for downstream ML pipelines: no missing values in model-ready export.
    for column in model.columns:
        if column == "date":
            model[column] = pd.to_datetime(model[column].astype(str), errors="coerce").dt.date
            model[column] = model[column].fillna(date(year, 1, 1))
        elif str(model[column].dtype) == "string":
            model[column] = model[column].fillna("Unknown")
        else:
            model[column] = model[column].fillna(0)
    return model


def process_file(
    source: Path,
    output_dir: Path,
    model_ready_dir: Path | None,
    daily_context: pd.DataFrame,
    hourly_events: pd.DataFrame,
    street_hourly: pd.DataFrame,
    year: int,
    overwrite: bool,
) -> ProcessStats:
    destination = output_name(source, output_dir, year)
    model_destination = model_ready_output_name(source, model_ready_dir, year) if model_ready_dir is not None else None
    model_destination_ready = model_destination is None or model_destination.exists()
    if destination.exists() and model_destination_ready and not overwrite:
        return ProcessStats(source.name, destination.name, 0, 0, 0, 0, 0, 0, "", 0, True, "exists")
    if not parquet_has_required_schema(source):
        return ProcessStats(source.name, destination.name, 0, 0, 0, 0, 0, 0, "", 0, True, "empty_or_invalid_schema")

    frame = pd.read_parquet(source)
    rows_in = len(frame)
    frame = filter_2025(frame, year)
    rows_2025 = len(frame)
    if frame.empty:
        return ProcessStats(source.name, destination.name, rows_in, rows_2025, 0, 0, 0, 0, "", 0, True, "no_rows_for_year")

    frame = add_hour_column(frame)
    frame, duplicate_rows_removed, negative_values_fixed, missing_values_filled = clean_and_impute(frame)
    frame = add_context(frame, daily_context, hourly_events, street_hourly)

    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False, engine="pyarrow", compression="snappy")
    model_ready_file = ""
    model_ready_rows = 0
    if model_destination is not None:
        model = build_model_ready_frame(frame, year)
        model_destination.parent.mkdir(parents=True, exist_ok=True)
        model.to_parquet(model_destination, index=False, engine="pyarrow", compression="snappy")
        model_ready_file = model_destination.name
        model_ready_rows = len(model)
    return ProcessStats(
        source.name,
        destination.name,
        rows_in,
        rows_2025,
        len(frame),
        duplicate_rows_removed,
        negative_values_fixed,
        missing_values_filled,
        model_ready_file,
        model_ready_rows,
    )


def iter_candidate_files(input_dir: Path, year: int) -> Iterable[Path]:
    # Filename filter keeps runtime down; row-level filtering still guarantees only the requested year is exported.
    return sorted(input_dir.glob(f"*{year}*.parquet"))


def main() -> None:
    args = parse_args()
    daily_context, hourly_events = build_daily_context(args.additional_dir, args.year)
    street_hourly = build_street_hourly_context(args.additional_dir, args.year) if args.include_streets else pd.DataFrame()
    model_ready_dir = None if args.no_model_ready else args.model_ready_dir

    files = list(iter_candidate_files(args.input_dir, args.year))
    if args.limit_files > 0:
        files = files[: args.limit_files]
    if not files:
        raise SystemExit(f"Keine Parquet-Dateien fuer {args.year} in {args.input_dir} gefunden.")

    stats: list[ProcessStats] = []
    for index, source in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {source.name}")
        stat = process_file(
            source=source,
            output_dir=args.output_dir,
            model_ready_dir=model_ready_dir,
            daily_context=daily_context,
            hourly_events=hourly_events,
            street_hourly=street_hourly,
            year=args.year,
            overwrite=args.overwrite,
        )
        stats.append(stat)
        if stat.skipped:
            print(f"  -> uebersprungen: {stat.reason}")
        else:
            print(f"  -> {stat.rows_out:,} Zeilen -> {stat.output_file}".replace(",", "."))
            if stat.model_ready_file:
                print(f"  -> {stat.model_ready_rows:,} ML-Zeilen -> {stat.model_ready_file}".replace(",", "."))

    manifest = {
        "year": args.year,
        "input_dir": str(args.input_dir),
        "additional_dir": str(args.additional_dir),
        "output_dir": str(args.output_dir),
        "model_ready_dir": str(model_ready_dir) if model_ready_dir is not None else "",
        "include_streets": bool(args.include_streets),
        "files_total": len(files),
        "files_written": sum(not stat.skipped for stat in stats),
        "files_skipped": sum(stat.skipped for stat in stats),
        "rows_out_total": sum(stat.rows_out for stat in stats),
        "model_ready_files_written": sum(bool(stat.model_ready_file) for stat in stats),
        "model_ready_rows_total": sum(stat.model_ready_rows for stat in stats),
        "model_ready_columns": [*MODEL_READY_COLUMNS, *MODEL_READY_STREET_COLUMNS],
        "stats": [stat.__dict__ for stat in stats],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / f"manifest_{args.year}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if model_ready_dir is not None:
        model_ready_dir.mkdir(parents=True, exist_ok=True)
        (model_ready_dir / f"manifest_model_ready_{args.year}.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(f"\nFertig. Manifest: {manifest_path}")
    if model_ready_dir is not None:
        print(f"Model-Ready-Manifest: {model_ready_dir / f'manifest_model_ready_{args.year}.json'}")


if __name__ == "__main__":
    main()
