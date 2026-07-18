from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .config import CONTEXT_COLUMNS, ENRICHED_TRAINING_DIR
from .types import TrainingData
from .utils import normalize_station_name


class TrainingDataLoader:
    """Loads the training base for Prediction.

    Priority:
    1. Cleaned 2025 parquet archive with context features.
    2. Fallback to the repository range used by the dashboard.
    """

    def __init__(self, repo: Any, enriched_dir: Path = ENRICHED_TRAINING_DIR) -> None:
        self.repo = repo
        self.enriched_dir = enriched_dir

    def load(self, lines: list[int], start: date, end: date) -> TrainingData:
        enriched = self._load_enriched_2025(lines)
        if not enriched.empty:
            return TrainingData(
                raw=enriched,
                source="full_api_data_enriched_2025",
                date_span=(min(enriched["date"]), max(enriched["date"])),
            )
        return self._load_dashboard_repository(lines, start, end)

    def _load_dashboard_repository(self, lines: list[int], start: date, end: date) -> TrainingData:
        frames = [self.repo.load_line_range(line, start, end) for line in lines]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return TrainingData(pd.DataFrame(), "Keine Trainingsdaten", None)
        raw = pd.concat(frames, ignore_index=True).copy()
        raw["station_key"] = raw["station"].map(normalize_station_name)
        return TrainingData(raw, "App-Datenordner", (start, end))

    def _enriched_files_for_lines(self, lines: list[int]) -> list[Path]:
        if not self.enriched_dir.exists():
            return []
        wanted = {int(line) for line in lines}
        pattern = re.compile(r"data_2025-\d{2}-\d{2}_2025-\d{2}-\d{2}_line_(\d+)_clean_context_2025\.parquet$")
        paths: list[Path] = []
        for path in sorted(self.enriched_dir.glob("*.parquet")):
            match = pattern.match(path.name)
            if match and int(match.group(1)) in wanted:
                paths.append(path)
        return paths

    def _read_enriched_file(self, path: Path) -> pd.DataFrame:
        columns = [
            "stop_event_id",
            "report_date",
            "journey",
            "departure_plan_station",
            "stop_sequence",
            "line",
            "station_number",
            "station",
            "vehicle_type",
            "passenger_boarding",
            "passenger_exiting",
            "occupancy_departure",
            "vehicle_utilization",
            "date",
            "hour",
        ] + CONTEXT_COLUMNS
        try:
            frame = pd.read_parquet(path, columns=columns)
        except Exception:
            frame = pd.read_parquet(path)
        for column in columns:
            if column not in frame.columns:
                frame[column] = 0
        return frame[columns].copy()

    def _load_enriched_2025(self, lines: list[int]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for path in self._enriched_files_for_lines(lines):
            try:
                frames.append(self._read_enriched_file(path))
            except Exception:
                continue
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame()

        raw = pd.concat(frames, ignore_index=True).copy()
        if "date" not in raw or raw["date"].isna().all():
            raw["date"] = raw["report_date"]
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.date
        raw = raw.dropna(subset=["date", "line", "station"])
        raw = raw[pd.to_datetime(raw["date"].astype(str), errors="coerce").dt.year == 2025].copy()
        if raw.empty:
            return pd.DataFrame()

        if "hour" not in raw or raw["hour"].isna().all():
            raw["hour"] = pd.to_datetime(raw["departure_plan_station"], errors="coerce").dt.hour
        numeric_columns = [
            "line",
            "station_number",
            "hour",
            "passenger_boarding",
            "passenger_exiting",
            "occupancy_departure",
            "vehicle_utilization",
        ]
        for column in numeric_columns:
            raw[column] = pd.to_numeric(raw[column], errors="coerce").fillna(0)
        raw["line"] = raw["line"].astype(int)
        raw["station_number"] = raw["station_number"].astype(int)
        raw["hour"] = raw["hour"].clip(0, 23).astype(int)
        raw["journey"] = raw["journey"].fillna("unknown").astype(str)
        raw["station"] = raw["station"].fillna("Unbekannt").astype(str)
        raw["vehicle_type"] = raw["vehicle_type"].replace(0, "Unknown").fillna("Unknown").astype(str)
        if "departure_plan_station" not in raw or raw["departure_plan_station"].eq(0).all():
            raw["departure_plan_station"] = pd.to_numeric(raw["stop_sequence"], errors="coerce").fillna(0)
        raw["station_key"] = raw["station"].map(normalize_station_name)

        for column in CONTEXT_COLUMNS:
            raw[column] = pd.to_numeric(raw[column], errors="coerce").fillna(0)
        if "event_hours" in raw and "event_hour" in raw:
            raw["event_hours"] = raw["event_hours"].where(raw["event_hours"] > 0, raw["event_hour"])
        if "concert_hours" in raw and "concert_hour" in raw:
            raw["concert_hours"] = raw["concert_hours"].where(raw["concert_hours"] > 0, raw["concert_hour"])

        if "stop_event_id" in raw:
            raw = raw.drop_duplicates(subset=["stop_event_id"], keep="last")
        else:
            raw = raw.drop_duplicates(
                subset=["date", "line", "journey", "station_number", "station", "hour"],
                keep="last",
            )
        return raw.sort_values(["date", "line", "journey", "departure_plan_station", "station_number"])
