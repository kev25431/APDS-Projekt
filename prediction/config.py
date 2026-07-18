from __future__ import annotations

from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = APP_DIR / "prediction" / "models"
OUTPUT_DIR = APP_DIR / "prediction" / "outputs"
ENRICHED_TRAINING_DIR = APP_DIR / "downloads" / "wvv-pjs-2026" / "full_api_data_enriched_2025"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONTEXT_COLUMNS = [
    "lecture_period_jmu",
    "lecture_period_thws",
    "public_holiday",
    "nationwide",
    "school_holiday",
    "event_hours",
    "concert_hours",
    "event_count",
    "concert_event_count",
    "total_event_duration_hours",
    "max_event_duration_hours",
    "event_day",
    "concert_day",
    "event_hour",
    "concert_hour",
    "verkaufsoffener_sonntag",
]

VEHICLE_TYPE_CAPACITY_FALLBACK = {
    "SKOM": 80.0,
    "GKOM": 140.0,
    "GTE": 180.0,
    "GTN": 170.0,
    "Unknown": 90.0,
}

TARGET_UTILIZATION = 0.82
INITIAL_RANDOM_FOREST_TREES = 90
INCREMENTAL_RANDOM_FOREST_TREES = 30
