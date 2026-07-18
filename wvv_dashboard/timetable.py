from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from .config import ADDITIONAL_DATA_DIR, TIMETABLE_CLEAN_DIR
from .kpis import DEFAULT_BUS_HOURLY_COST_EUR
from .service_policy import constrained_adaptive_runs


SERVICE_LABELS = {
    "weekday": "Montag bis Freitag",
    "saturday": "Samstag",
    "sunday_holiday": "Sonntag / Feiertag",
    "unknown": "Unbekannter Gültigkeitstyp",
}


def station_key(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("strasse", "str").replace("str.", "str")
    return re.sub(r"[^a-z0-9]+", "", text)


def format_minutes(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    minutes = int(round(float(value))) % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


@dataclass
class TimetableComparison:
    line: int
    service_key: str
    service_label: str
    source_file: str
    route: pd.DataFrame
    wvv_table: pd.DataFrame
    adaptive_table: pd.DataFrame
    summary: dict[str, object] = field(default_factory=dict)
    has_prediction: bool = False


class TimetableRepository:
    def __init__(self, clean_dir: Path = TIMETABLE_CLEAN_DIR) -> None:
        self.clean_dir = clean_dir
        self._long: pd.DataFrame | None = None
        self._routes: pd.DataFrame | None = None
        self._holiday_dates: set[date] | None = None

    def available_lines(self) -> list[int]:
        long = self.long
        if long.empty or "line" not in long:
            return []
        return sorted(int(value) for value in long["line"].dropna().unique())

    def hourly_run_counts_for_line(
        self,
        line: int,
        selected_day: date,
        start_hour: int = 0,
        horizon_hours: int = 24,
    ) -> dict[int, int]:
        """Count all WVV departures for a line/service day across route tables."""
        long = self.long
        if long.empty:
            return {}
        service_key = self.service_key_for_date(selected_day)
        frame = long[
            (long["line"].astype("Int64") == int(line))
            & (long["service_key"].astype(str) == str(service_key))
        ].copy()
        if frame.empty:
            return {}

        plan_key = self._best_plan_key(frame)
        if plan_key:
            frame = frame[frame["source_file"].astype(str).map(self._plan_key) == plan_key].copy()
        if frame.empty:
            return {}

        first_times = (
            frame.groupby("trip_key", as_index=False)
            .agg(first_minute=("minutes", "min"), records=("station_key", "size"))
            .dropna(subset=["first_minute"])
        )
        hours = {(int(start_hour) + offset) % 24 for offset in range(max(1, int(horizon_hours)))}
        counts: dict[int, int] = {}
        for minute in first_times["first_minute"].tolist():
            hour = int(float(minute) // 60) % 24
            if hour in hours:
                counts[hour] = counts.get(hour, 0) + 1
        return counts

    @property
    def long(self) -> pd.DataFrame:
        if self._long is None:
            self._long = self._read_clean_file("fahrplan_long")
        return self._long

    @property
    def routes(self) -> pd.DataFrame:
        if self._routes is None:
            self._routes = self._read_clean_file("fahrplan_routes")
        return self._routes

    def _read_clean_file(self, stem: str) -> pd.DataFrame:
        parquet_path = self.clean_dir / f"{stem}.parquet"
        csv_path = self.clean_dir / f"{stem}.csv"
        if parquet_path.exists():
            frame = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            frame = pd.read_csv(csv_path)
        else:
            return pd.DataFrame()

        if "effective_date" in frame.columns:
            frame["effective_date"] = pd.to_datetime(frame["effective_date"], errors="coerce").dt.date
        if "line" in frame.columns:
            frame["line"] = pd.to_numeric(frame["line"], errors="coerce").astype("Int64")
        if "stop_sequence" in frame.columns:
            frame["stop_sequence"] = pd.to_numeric(frame["stop_sequence"], errors="coerce").fillna(0).astype(int)
        if "minutes" in frame.columns:
            frame["minutes"] = pd.to_numeric(frame["minutes"], errors="coerce")
        if "station_key" not in frame.columns and "station" in frame.columns:
            frame["station_key"] = frame["station"].map(station_key)
        return frame

    def service_key_for_date(self, selected_day: date) -> str:
        if selected_day in self._public_holiday_dates():
            return "sunday_holiday"
        if selected_day.weekday() < 5:
            return "weekday"
        if selected_day.weekday() == 5:
            return "saturday"
        return "sunday_holiday"

    def _public_holiday_dates(self) -> set[date]:
        if self._holiday_dates is not None:
            return self._holiday_dates
        path = ADDITIONAL_DATA_DIR / "bavarian_public_holidays_daily.csv"
        if not path.exists():
            self._holiday_dates = set()
            return self._holiday_dates
        try:
            frame = pd.read_csv(path, usecols=["date", "public_holiday"])
        except (OSError, ValueError, pd.errors.ParserError):
            self._holiday_dates = set()
            return self._holiday_dates
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
        holidays = frame[pd.to_numeric(frame["public_holiday"], errors="coerce").fillna(0).astype(int) > 0]
        self._holiday_dates = set(holidays["date"].dropna().tolist())
        return self._holiday_dates

    def build_comparison(
        self,
        line: int,
        selected_day: date,
        start_hour: int = 0,
        horizon_hours: int = 24,
        prediction_schedule: pd.DataFrame | None = None,
        max_trips: int = 160,
    ) -> TimetableComparison:
        long = self.long
        if long.empty:
            return self._empty_comparison(line, selected_day, "Keine bereinigten Fahrplandaten gefunden.")

        service_key = self.service_key_for_date(selected_day)
        selected = self._select_best_table(line, service_key, start_hour, horizon_hours)
        if selected.empty:
            return self._empty_comparison(line, selected_day, f"Keine Fahrplandaten für Linie {line} gefunden.")

        source_file = self._source_label(selected)
        table_no = int(selected["table_no"].iloc[0])
        selected_service = str(selected["service_key"].iloc[0])
        route = self._route_for(line, source_file, table_no, selected_service, selected)
        wvv_table, offsets, departures = self._build_wvv_table(selected, route, start_hour, horizon_hours, max_trips)
        adaptive_table, adaptive_departures = self._build_adaptive_table(
            line,
            route,
            offsets,
            departures,
            prediction_schedule,
            selected_day,
            start_hour,
            horizon_hours,
            max_trips,
        )
        summary = {
            "selected_day": selected_day.isoformat(),
            "start_hour": start_hour,
            "horizon_hours": horizon_hours,
            "wvv_runs": len(departures),
            "adaptive_runs": len(adaptive_departures),
            "note": self._summary_note(departures, adaptive_departures, prediction_schedule),
        }
        return TimetableComparison(
            line=int(line),
            service_key=selected_service,
            service_label=SERVICE_LABELS.get(selected_service, selected_service),
            source_file=source_file,
            route=route,
            wvv_table=wvv_table,
            adaptive_table=adaptive_table,
            summary=summary,
            has_prediction=prediction_schedule is not None and not prediction_schedule.empty,
        )

    def _empty_comparison(self, line: int, selected_day: date, message: str) -> TimetableComparison:
        return TimetableComparison(
            line=int(line),
            service_key=self.service_key_for_date(selected_day),
            service_label=SERVICE_LABELS.get(self.service_key_for_date(selected_day), "-"),
            source_file="-",
            route=pd.DataFrame(columns=["stop_sequence", "station", "station_key", "marker"]),
            wvv_table=pd.DataFrame({"Haltestelle": [message]}),
            adaptive_table=pd.DataFrame({"Haltestelle": [message]}),
            summary={"note": message, "wvv_runs": 0, "adaptive_runs": 0},
            has_prediction=False,
        )

    def _select_best_table(self, line: int, service_key: str, start_hour: int, horizon_hours: int) -> pd.DataFrame:
        frame = self.long[self.long["line"].astype("Int64") == int(line)].copy()
        if frame.empty:
            return frame

        service_candidates = [service_key, "unknown", "weekday", "saturday", "sunday_holiday"]
        service_candidates = list(dict.fromkeys(service_candidates))
        best_summary = pd.DataFrame()
        for key in service_candidates:
            subset = frame[frame["service_key"].astype(str) == key].copy()
            if subset.empty:
                continue
            summary = (
                subset.groupby(["source_file", "table_no", "service_key", "effective_date"], dropna=False)
                .agg(
                    stops=("station_key", "nunique"),
                    trips=("trip_key", "nunique"),
                    rows=("station_key", "size"),
                    min_minutes=("minutes", "min"),
                    max_minutes=("minutes", "max"),
                )
                .reset_index()
            )
            start_min = int(start_hour) * 60
            end_min = (int(start_hour) + max(1, int(horizon_hours))) * 60
            covering = summary[(summary["max_minutes"] >= start_min) & (summary["min_minutes"] <= end_min)]
            best_summary = covering if not covering.empty else summary
            break

        if best_summary.empty:
            return frame.iloc[0:0].copy()

        best_summary = best_summary.copy()
        best_summary["effective_sort"] = pd.to_datetime(best_summary["effective_date"], errors="coerce").fillna(pd.Timestamp("1900-01-01"))
        best = best_summary.sort_values(
            ["effective_sort", "stops", "trips", "rows"],
            ascending=[False, False, False, False],
        ).iloc[0]
        best_subset = frame[
            (frame["source_file"] == best["source_file"])
            & (frame["table_no"] == best["table_no"])
            & (frame["service_key"] == best["service_key"])
        ].copy()
        best_signature = self._route_signature(best_subset)
        best_plan = self._plan_key(str(best["source_file"]))
        same_plan = frame[
            (frame["service_key"] == best["service_key"])
            & (frame["source_file"].astype(str).map(self._plan_key) == best_plan)
        ].copy()
        matching_tables: list[pd.DataFrame] = []
        for (_source_file, _table_no), table in same_plan.groupby(["source_file", "table_no"], dropna=False):
            if self._route_signature(table) == best_signature:
                matching_tables.append(table)
        if matching_tables:
            return pd.concat(matching_tables, ignore_index=True)
        return best_subset

    def _plan_key(self, source_file: str) -> str:
        return re.sub(r"__(table|sheet)_[^/\\]+\.csv$", "", str(source_file))

    def _best_plan_key(self, frame: pd.DataFrame) -> str:
        if frame.empty:
            return ""
        summary = (
            frame.assign(plan_key=frame["source_file"].astype(str).map(self._plan_key))
            .groupby("plan_key", as_index=False)
            .agg(
                effective_date=("effective_date", "max"),
                trips=("trip_key", "nunique"),
                rows=("station_key", "size"),
                tables=("table_no", "nunique"),
            )
        )
        if summary.empty:
            return ""
        summary["effective_sort"] = pd.to_datetime(summary["effective_date"], errors="coerce").fillna(pd.Timestamp("1900-01-01"))
        best = summary.sort_values(
            ["effective_sort", "trips", "rows", "tables"],
            ascending=[False, False, False, False],
        ).iloc[0]
        return str(best["plan_key"])

    def _route_signature(self, table: pd.DataFrame) -> tuple[str, ...]:
        if table.empty:
            return tuple()
        route = (
            table.sort_values(["stop_sequence", "minutes"])
            .drop_duplicates("station_key")["station_key"]
            .astype(str)
            .tolist()
        )
        return tuple(route)

    def _source_label(self, selected: pd.DataFrame) -> str:
        if selected.empty or "source_file" not in selected:
            return "-"
        sources = selected[["source_file", "table_no"]].drop_duplicates()
        first = str(sources["source_file"].iloc[0])
        plan = self._plan_key(first)
        if len(sources) <= 1:
            return first
        return f"{plan} ({len(sources)} Tabellen zusammengeführt)"

    def _route_for(
        self,
        line: int,
        source_file: str,
        table_no: int,
        service_key: str,
        selected: pd.DataFrame,
    ) -> pd.DataFrame:
        routes = self.routes
        route = pd.DataFrame()
        if not routes.empty:
            route = routes[
                (routes["line"].astype("Int64") == int(line))
                & (routes["source_file"] == source_file)
                & (routes["table_no"] == int(table_no))
                & (routes["service_key"].astype(str) == str(service_key))
            ].copy()
        if route.empty:
            route = (
                selected.sort_values(["stop_sequence", "minutes"])
                .drop_duplicates("station_key")
                [["stop_sequence", "station", "station_key", "marker"]]
                .copy()
            )
        route = route.sort_values(["stop_sequence", "station"]).drop_duplicates("station_key")
        return route[["stop_sequence", "station", "station_key", "marker"]].reset_index(drop=True)

    def _build_wvv_table(
        self,
        selected: pd.DataFrame,
        route: pd.DataFrame,
        start_hour: int,
        horizon_hours: int,
        max_trips: int,
    ) -> tuple[pd.DataFrame, dict[str, float], list[dict[str, object]]]:
        if selected.empty or route.empty:
            return pd.DataFrame(), {}, []

        first_times = (
            selected.groupby("trip_key", as_index=False)
            .agg(first_minute=("minutes", "min"), records=("station_key", "size"))
            .dropna(subset=["first_minute"])
            .sort_values("first_minute")
        )
        first_times["first_minute_round"] = first_times["first_minute"].round().astype(int)
        first_times = (
            first_times.sort_values(["first_minute_round", "records"], ascending=[True, False])
            .drop_duplicates("first_minute_round", keep="first")
            .sort_values("first_minute")
        )
        start_min = int(start_hour) * 60
        end_min = (int(start_hour) + max(1, int(horizon_hours))) * 60
        window = first_times[(first_times["first_minute"] >= start_min) & (first_times["first_minute"] < end_min)]
        if window.empty:
            window = first_times.head(max_trips)
        else:
            window = window.head(max_trips)

        departures = [
            {
                "trip_key": str(row["trip_key"]),
                "minute": float(row["first_minute"]),
                "synthetic": False,
            }
            for _, row in window.iterrows()
        ]
        offsets = self._travel_offsets(selected, route, departures)
        table = self._matrix_from_trips(selected, route, departures)
        return table, offsets, departures

    def _travel_offsets(
        self,
        selected: pd.DataFrame,
        route: pd.DataFrame,
        departures: list[dict[str, object]],
    ) -> dict[str, float]:
        if selected.empty or route.empty:
            return {}
        trip_first = selected.groupby("trip_key")["minutes"].min().to_dict()
        working = selected.copy()
        working["first_minute"] = working["trip_key"].map(trip_first)
        working["offset"] = working["minutes"] - working["first_minute"]
        offsets = (
            working.groupby("station_key")["offset"]
            .median()
            .dropna()
            .to_dict()
        )
        result: dict[str, float] = {}
        last = 0.0
        for _, row in route.iterrows():
            key = str(row["station_key"])
            value = float(offsets.get(key, last))
            value = max(value, last)
            result[key] = value
            last = value
        return result

    def _matrix_from_trips(
        self,
        selected: pd.DataFrame,
        route: pd.DataFrame,
        departures: list[dict[str, object]],
    ) -> pd.DataFrame:
        base = route[["stop_sequence", "station", "station_key", "marker"]].copy()
        base["Haltestelle"] = base["station"].astype(str)
        base = base[["Haltestelle", "station_key", "marker"]]
        for index, departure in enumerate(departures, start=1):
            trip_key = str(departure["trip_key"])
            label = format_minutes(departure["minute"])
            lookup = (
                selected[selected["trip_key"].astype(str) == trip_key]
                .sort_values(["stop_sequence", "minutes"])
                .drop_duplicates("station_key")
                .set_index("station_key")["minutes"]
                .to_dict()
            )
            base[f"{index:02d} {label}"] = base["station_key"].map(lambda key: format_minutes(lookup.get(key)))
        return base.drop(columns=["station_key"])

    def _build_adaptive_table(
        self,
        line: int,
        route: pd.DataFrame,
        offsets: dict[str, float],
        wvv_departures: list[dict[str, object]],
        prediction_schedule: pd.DataFrame | None,
        selected_day: date,
        start_hour: int,
        horizon_hours: int,
        max_trips: int,
    ) -> tuple[pd.DataFrame, list[dict[str, object]]]:
        if route.empty:
            return pd.DataFrame(), []

        schedule_by_hour = self._schedule_rows_by_hour(prediction_schedule, selected_day, start_hour, horizon_hours)
        service_hours = self._service_hours_from_departures(wvv_departures)
        departures: list[dict[str, object]] = []
        for hour in range(int(start_hour), int(start_hour) + max(1, int(horizon_hours))):
            hour_mod = hour % 24
            hour_start = hour * 60
            existing = [item for item in wvv_departures if int(float(item["minute"]) // 60) == hour_mod]
            allow_new_service = self._hour_inside_service_span(hour_mod, service_hours) or bool(existing)
            target = self._efficient_target_runs(schedule_by_hour.get(hour_mod), len(existing), allow_new_service, line, hour_mod)
            departures.extend(self._adaptive_hour_departures(hour_start, existing, target))

        departures = sorted(departures, key=lambda item: float(item["minute"]))[:max_trips]
        table = route[["station", "station_key", "marker"]].copy()
        table["Haltestelle"] = table["station"].astype(str)
        table = table[["Haltestelle", "station_key", "marker"]]
        for index, departure in enumerate(departures, start=1):
            label = format_minutes(departure["minute"])
            if departure.get("synthetic"):
                label += "*"
            column = f"{index:02d} {label}"
            table[column] = table["station_key"].map(
                lambda key: format_minutes(float(departure["minute"]) + offsets.get(str(key), 0.0))
            )
        return table.drop(columns=["station_key"]), departures

    def _service_hours_from_departures(self, departures: list[dict[str, object]]) -> tuple[int, int] | None:
        if not departures:
            return None
        minutes = [int(float(item["minute"])) for item in departures]
        return min(minutes) // 60, max(minutes) // 60

    def _hour_inside_service_span(self, hour: int, service_hours: tuple[int, int] | None) -> bool:
        if service_hours is None:
            return True
        start_hour, end_hour = service_hours
        if start_hour <= end_hour:
            return start_hour <= int(hour) <= end_hour
        return int(hour) >= start_hour or int(hour) <= end_hour

    def _schedule_rows_by_hour(
        self,
        prediction_schedule: pd.DataFrame | None,
        selected_day: date,
        start_hour: int,
        horizon_hours: int,
    ) -> dict[int, pd.Series]:
        if prediction_schedule is None or prediction_schedule.empty:
            return {}
        schedule = prediction_schedule.copy()
        if "date" in schedule:
            schedule["date"] = pd.to_datetime(schedule["date"], errors="coerce").dt.date
            schedule = schedule[schedule["date"] == selected_day]
        if schedule.empty:
            return {}
        result: dict[int, pd.Series] = {}
        hours = {(int(start_hour) + offset) % 24 for offset in range(max(1, int(horizon_hours)))}
        for _, row in schedule.iterrows():
            hour = int(row.get("hour", -1))
            if hour in hours:
                result[hour] = row
        return result

    def _efficient_target_runs(
        self,
        schedule_row: pd.Series | None,
        existing_count: int,
        allow_new_service: bool = True,
        line: int | None = None,
        hour: int | None = None,
    ) -> int:
        if schedule_row is None:
            return max(0, int(existing_count))
        if not allow_new_service and int(existing_count) <= 0:
            return 0

        boardings = self._safe_float(schedule_row.get("predicted_boardings", math.nan), math.nan)
        if pd.isna(boardings):
            boardings = self._safe_float(schedule_row.get("predicted_demand", 0.0), 0.0) * 0.55
        return constrained_adaptive_runs(
            demand=boardings,
            baseline_runs=existing_count,
            avg_capacity=self._safe_float(schedule_row.get("avg_vehicle_capacity", 70.0), 70.0),
            cost_per_bus_hour=DEFAULT_BUS_HOURLY_COST_EUR,
            default_cost_per_bus_hour=DEFAULT_BUS_HOURLY_COST_EUR,
            line=line if line is not None else schedule_row.get("line"),
            hour=hour if hour is not None else schedule_row.get("hour"),
            allow_new_service=allow_new_service,
        )

    @staticmethod
    def _safe_float(value: object, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(number):
            return float(default)
        return number

    def _adaptive_hour_departures(
        self,
        hour_start: int,
        existing: list[dict[str, object]],
        target: int,
    ) -> list[dict[str, object]]:
        target = max(0, min(int(target), 12))
        if target == 0:
            return []
        existing = sorted(existing, key=lambda item: float(item["minute"]))
        if len(existing) == target:
            return [dict(item) for item in existing]
        if len(existing) > target:
            if target == 1:
                return [dict(existing[len(existing) // 2])]
            step = (len(existing) - 1) / max(target - 1, 1)
            return [dict(existing[int(round(index * step))]) for index in range(target)]

        generated = [dict(item) for item in existing]
        needed = target - len(generated)
        spacing = 60 / max(target, 1)
        candidate_minutes = [hour_start + (spacing / 2) + index * spacing for index in range(target)]
        existing_minutes = {int(round(float(item["minute"]))) for item in generated}
        for minute in candidate_minutes:
            rounded = int(round(minute))
            if rounded in existing_minutes:
                continue
            generated.append({"trip_key": f"adaptive:{rounded}", "minute": float(rounded), "synthetic": True})
            existing_minutes.add(rounded)
            needed -= 1
            if needed <= 0:
                break
        return sorted(generated, key=lambda item: float(item["minute"]))

    def _summary_note(
        self,
        wvv_departures: list[dict[str, object]],
        adaptive_departures: list[dict[str, object]],
        prediction_schedule: pd.DataFrame | None,
    ) -> str:
        if prediction_schedule is None or prediction_schedule.empty:
            return "Kein trainiertes Prognosemodell geladen: adaptiver Plan nutzt WVV-Bestand als Fallback."
        delta = len(adaptive_departures) - len(wvv_departures)
        if delta > 0:
            return f"Adaptive Planung würde {delta} zusätzliche Fahrten über den Betriebstag anbieten."
        if delta < 0:
            return f"Adaptive Planung würde {abs(delta)} Fahrten über den Betriebstag einsparen."
        return "Adaptive Planung hält die Fahrtenzahl, verteilt sie aber prognosebasiert über den Betriebstag."
