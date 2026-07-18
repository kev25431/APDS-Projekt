from __future__ import annotations

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "downloads" / "wvv-pjs-2026" / "full_api_data"
ADDITIONAL_DATA_DIR = APP_DIR / "downloads" / "wvv-pjs-2026" / "Additional Data"
EVENT_GEO_PATH = ADDITIONAL_DATA_DIR / "events_with_latitude_longitude.csv"
TIMETABLE_RAW_DIR = APP_DIR / "downloads" / "wvv-pjs-2026" / "Fahrplaene_neu"
TIMETABLE_CLEAN_DIR = APP_DIR / "downloads" / "wvv-pjs-2026" / "Fahrplaene_clean"
GEOJSON_PATH = APP_DIR / "downloads" / "export.geojson"
MPL_DIR = APP_DIR / ".mpl-cache"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
ROUTE_CACHE_PATH = APP_DIR / ".route-cache" / "osrm_routes.json"
BUS_ICON_PATH = APP_DIR / "wvv_dashboard" / "assets" / "bus_icon_small.png"
COST_SETTINGS_PATH = APP_DIR / "prediction" / "outputs" / "cost_settings.json"

PALETTE = {
    "page_bg": "#0f172a",
    "surface": "#172033",
    "surface_alt": "#22304a",
    "surface_soft": "#111827",
    "sidebar": "#0b1120",
    "sidebar_active": "#ef4444",
    "text": "#f8fafc",
    "muted": "#94a3b8",
    "accent": "#ef4444",
    "accent_dark": "#dc2626",
    "accent_soft": "#29364f",
    "border": "#334155",
    "danger": "#f97316",
    "chip": "#1e293b",
    "teal": "#14b8a6",
    "amber": "#f59e0b",
    "blue": "#38bdf8",
    "rose": "#fb7185",
}

WUERZBURG_CENTER = (49.7913, 9.9534)
WUERZBURG_ZOOM = 13
MAP_HOUR_PAUSE_MS = 1200
MAP_BUS_MINUTE_STEP = 0.5
MAP_TRANSITION_STEPS = int(60 / MAP_BUS_MINUTE_STEP)
MAP_BUS_FRAME_DELAY_MS = 110
MAP_STOP_EVENT_WINDOW_MINUTES = 3.0
MAP_BAR_REFERENCE_FLOW = 20.0
MAP_BAR_MAX_FLOW = 80.0

LINE_LABELS = {
    10: "Linie 10 | Hubland Volatil",
    20: "Linie 20 | Stabile Stadtachse",
    27: "Linie 27 | Pendlerfokus",
    33: "Linie 33 | Grundversorgung",
    55: "Linie 55 | Eventgetrieben",
    214: "Linie 214 | Uni Direkt",
}

GEOJSON_STATION_ALIASES = {
    "busbahnhof": "hauptbahnhofzob",
    "ehstaumuhle": "aumuhle",
}
