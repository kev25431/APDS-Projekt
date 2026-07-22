from __future__ import annotations

"""
Bereinigung und Strukturierung extrahierter Fahrplan-CSV-Dateien.

Das Skript transformiert tabellenförmige PDF/CSV-Extrakte in ein
maschinenlesbares Long-Format sowie eine reduzierte Routenbeschreibung.
Dabei werden Linien, Gültigkeitsdaten, Bedienzeiträume, Haltestellenfolgen und
Abfahrtszeiten aus teils unstrukturierten Tabellen rekonstruiert.

Projektkontext:
    Die erzeugten Fahrplandaten bilden die Grundlage für spätere
    Angebotsvergleiche, adaptive Fahrplanlogik und die Verknüpfung von
    Prognosen mit realen Linien- und Haltestellenstrukturen.
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from prediction import normalize_station_name # noqa: E402
from wvv_dashboard.config import TIMETABLE_CLEAN_DIR, TIMETABLE_RAW_DIR # noqa: E402

TIME_RE = re.compile(r"\b([0-2]?\d):([0-5]\d)\b")
TABLE_RE = re.compile(r"__table_(\d+)\.csv$", re.IGNORECASE)
EFFECTIVE_RE = re.compile(r"_ab_(\d{2})\.(\d{2})\.(\d{4})", re.IGNORECASE)
LINE_RE = re.compile(r"Linie[_\s-]*(\d+)", re.IGNORECASE)
SERVICE_PATTERNS = [
    ("weekday", re.compile(r"montag|freitag|montags|freitags|mo\.", re.IGNORECASE)),
    ("saturday", re.compile(r"samstag|samstags|sa\.", re.IGNORECASE)),
    ("sunday_holiday", re.compile(r"sonn|feiertag|feiertags|so\.", re.IGNORECASE)),
]


def clean_text(value: object) -> str:
    """
    Normalisiert Zellinhalte für robuste Mustererkennung.

    Zeilenumbrüche und Mehrfachleerzeichen werden entfernt, damit später
    reguläre Ausdrücke auf Fahrplanüberschriften, Haltestellen und Zeitangaben
    konsistent angewendet werden können.
    """
    text = "" if value is None else str(value)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_line(path: Path, root: Path) -> int | None:
    """
    Leitet die Liniennummer aus Dateiname oder Ordnerstruktur ab.

    Damit bleibt das Skript auch dann robust, wenn die Liniennummer nicht
    sauber im Dateinamen, sondern nur im übergeordneten Linienordner enthalten
    ist.
    """
    match = LINE_RE.search(path.name)
    if match:
        return int(match.group(1))
    folder = path.relative_to(root).parts[0]
    numbers = [int(value) for value in re.findall(r"\d+", folder)]
    return numbers[0] if numbers else None


def parse_effective_date(path: Path) -> str:
    """
    Extrahiert das Gültigkeitsdatum eines Fahrplanblatts aus dem Dateinamen.

    Das Datum wird als ISO-Format zurückgegeben, damit es später konsistent in
    tabellarischen Ausgaben und Vergleichen verwendet werden kann.
    """
    match = EFFECTIVE_RE.search(path.name)
    if not match:
        return ""
    day, month, year = match.groups()
    try:
        return datetime(int(year), int(month), int(day)).date().isoformat()
    except ValueError:
        return ""


def parse_table_number(path: Path) -> int:
    """
    Liest eine tabelleninterne Nummer aus dem Dateinamen aus.

    Diese Nummer hilft dabei, unterschiedliche Tabellen eines PDFs oder
    Fahrplanblatts im späteren Long-Format eindeutig zu unterscheiden.
    """
    match = TABLE_RE.search(path.name)
    return int(match.group(1)) if match else -1


def detect_service_period(frame: pd.DataFrame) -> tuple[str, str]:
    """
    Klassifiziert den Bedienzeitraum einer Tabelle heuristisch aus dem Kopfbereich.

    Unterschieden werden Werktag, Samstag sowie Sonn-/Feiertag. Die ermittelte
    Rohüberschrift wird zusätzlich gekürzt gespeichert, damit unklare oder
    fehlerhafte Klassifikationen später nachvollzogen werden können.
    """
    head_text = " ".join(clean_text(value) for value in frame.head(4).to_numpy().ravel())
    for key, pattern in SERVICE_PATTERNS:
        if pattern.search(head_text):
            return key, head_text[:120]
    return "unknown", head_text[:120]


def split_station_marker(raw_station: str, marker_cell: str) -> tuple[str, str]:
    """
    Trennt Haltestellenname und An-/Ab-Markierung.

    In den Extrakten kann der Marker entweder in einer separaten Zelle oder
    direkt am Namen hängen. Diese Funktion vereinheitlicht beide Fälle und
    isoliert ``ab`` bzw. ``an`` als eigenes Merkmal.
    """
    text = clean_text(raw_station)
    marker = clean_text(marker_cell).lower()
    if marker not in {"ab", "an"}:
        marker = ""
    match = re.search(r"\b(ab|an)\s*$", text, flags=re.IGNORECASE)
    if match:
        marker = match.group(1).lower()
        text = text[: match.start()].strip()
    text = re.sub(r"\s{2,}", " ", text)
    return text, marker


def extract_times(value: object) -> list[str]:
    """
    Extrahiert alle Uhrzeiten aus einer Fahrplanzelle.

    Mehrfachzeiten in derselben Zelle bleiben erhalten und werden in ihrer
    Reihenfolge ausgegeben. Das ist wichtig für Spalten, die mehrere Fahrten
    oder Varianten kodieren.
    """
    text = clean_text(value)
    return [f"{int(hour):02d}:{minute}" for hour, minute in TIME_RE.findall(text)]


def looks_like_station_row(row: pd.Series) -> bool:
    """
    Erkennt heuristisch, ob eine Tabellenzeile eine echte Haltestellenzeile ist.

    Ausgeschlossen werden leere Zeilen, reine Nummernzeilen und Hinweise wie
    „Verkehrshinweis“. Gleichzeitig muss mindestens eine Zeitangabe in den
    späteren Spalten vorhanden sein.
    """
    first = clean_text(row.iloc[0] if len(row) else "")
    if not first or first.lower().startswith("verkehrshinweis"):
        return False
    if first.isdigit():
        return False
    return any(extract_times(value) for value in row.iloc[2:])


def parse_table(path: Path, root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """
    Zerlegt eine Fahrplantabelle in Long-Records und Routeninformationen.

    Für jede erkannte Haltestellenzeile werden einerseits strukturierte
    Haltestellenmetadaten erzeugt und andererseits alle Zeitangaben pro
    Spalte/Fahrt als einzelne Long-Records geschrieben. Dadurch entstehen
    sowohl eine Linienroute als auch ein fahrtenbezogenes Zeitraster.
    """
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:
        return [], []
    if frame.empty or frame.shape[1] < 3:
        return [], []

    line = parse_line(path, root)
    if line is None:
        return [], []
    line_group = path.relative_to(root).parts[0]
    table_no = parse_table_number(path)
    effective_date = parse_effective_date(path)
    service_key, service_label = detect_service_period(frame)
    records: list[dict[str, object]] = []
    route_rows: list[dict[str, object]] = []

    station_sequence = 0
    for row_idx, row in frame.iterrows():
        if not looks_like_station_row(row):
            continue
        station, marker = split_station_marker(row.iloc[0], row.iloc[1] if len(row) > 1 else "")
        if not station:
            continue
        station_sequence += 1
        station_key = normalize_station_name(station)
        route_rows.append(
            {
                "line": line,
                "line_group": line_group,
                "source_file": path.name,
                "table_no": table_no,
                "effective_date": effective_date,
                "service_key": service_key,
                "service_label": service_label,
                "stop_sequence": station_sequence,
                "station": station,
                "station_key": station_key,
                "marker": marker,
            }
        )

        for col_idx in range(2, frame.shape[1]):
            times = extract_times(row.iloc[col_idx])
            for multi_idx, time_value in enumerate(times):
                hour = int(time_value[:2])
                minute = int(time_value[3:])
                records.append(
                    {
                        "line": line,
                        "line_group": line_group,
                        "source_file": path.name,
                        "table_no": table_no,
                        "effective_date": effective_date,
                        "service_key": service_key,
                        "service_label": service_label,
                        "stop_sequence": station_sequence,
                        "station": station,
                        "station_key": station_key,
                        "marker": marker,
                        "trip_column": col_idx,
                        "multi_index": multi_idx,
                        "trip_key": f"{path.stem}:c{col_idx}:m{multi_idx}",
                        "time": time_value,
                        "minutes": hour * 60 + minute,
                        "row_index": int(row_idx),
                    }
                )
    return records, route_rows


def clean_timetables(raw_dir: Path = TIMETABLE_RAW_DIR, clean_dir: Path = TIMETABLE_CLEAN_DIR) -> dict[str, object]:
    """
    Bereinigt alle verfügbaren Fahrplanextrakte eines Rohordners.

    Das Ergebnis umfasst ein fahrtenbezogenes Long-Format, eine deduplizierte
    Routenbeschreibung sowie eine JSON-Zusammenfassung mit Dateizahlen,
    Zeilenvolumina und Zielpfaden.
    """
    clean_dir.mkdir(parents=True, exist_ok=True)
    files = [path for path in sorted(raw_dir.rglob("*.csv")) if path.name.lower() != "conversion_log.csv"]
    all_records: list[dict[str, object]] = []
    all_routes: list[dict[str, object]] = []
    parsed_files = 0
    for path in files:
        records, routes = parse_table(path, raw_dir)
        if records:
            parsed_files += 1
            all_records.extend(records)
            all_routes.extend(routes)

    long_df = pd.DataFrame(all_records)
    route_df = pd.DataFrame(all_routes)
    if not long_df.empty:
        long_df = long_df.drop_duplicates(
            subset=["line", "source_file", "table_no", "station_key", "trip_key", "time"],
            keep="first",
        ).sort_values(["line", "service_key", "source_file", "table_no", "trip_column", "multi_index", "stop_sequence"])
    if not route_df.empty:
        route_df = route_df.drop_duplicates(
            subset=["line", "source_file", "table_no", "service_key", "station_key", "stop_sequence"],
            keep="first",
        ).sort_values(["line", "service_key", "source_file", "table_no", "stop_sequence"])

    long_csv = clean_dir / "fahrplan_long.csv"
    route_csv = clean_dir / "fahrplan_routes.csv"
    long_parquet = clean_dir / "fahrplan_long.parquet"
    route_parquet = clean_dir / "fahrplan_routes.parquet"
    summary_path = clean_dir / "fahrplan_clean_summary.json"

    long_df.to_csv(long_csv, index=False, encoding="utf-8")
    route_df.to_csv(route_csv, index=False, encoding="utf-8")
    try:
        long_df.to_parquet(long_parquet, index=False)
        route_df.to_parquet(route_parquet, index=False)
    except Exception:
        long_parquet = Path("")
        route_parquet = Path("")

    summary = {
        "raw_dir": str(raw_dir),
        "clean_dir": str(clean_dir),
        "input_csv_files": len(files),
        "parsed_files": parsed_files,
        "long_rows": int(len(long_df)),
        "route_rows": int(len(route_df)),
        "lines": sorted(int(value) for value in long_df["line"].dropna().unique()) if not long_df.empty else [],
        "outputs": {
            "long_csv": str(long_csv),
            "route_csv": str(route_csv),
            "long_parquet": str(long_parquet) if str(long_parquet) else "",
            "route_parquet": str(route_parquet) if str(route_parquet) else "",
        },
    }

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> int:
    """
    Startet die CLI-Ausführung für die Fahrplanbereinigung.

    Die JSON-Ausgabe auf stdout eignet sich zugleich als Kurzprotokoll für
    Skriptketten oder manuelle Batchläufe.
    """
    parser = argparse.ArgumentParser(description="Bereinigt WVV-Fahrplan-PDF-CSV-Extrakte in ein nutzbares Long-Format.")
    parser.add_argument("--raw-dir", type=Path, default=TIMETABLE_RAW_DIR)
    parser.add_argument("--clean-dir", type=Path, default=TIMETABLE_CLEAN_DIR)
    args = parser.parse_args()
    summary = clean_timetables(args.raw_dir, args.clean_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())