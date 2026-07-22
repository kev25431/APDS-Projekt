from __future__ import annotations


"""
Zentrales Dashboard-Modul für Analyse, Visualisierung und fahrplanbezogene Auswertung
von WVV-Transitdaten.

Das Modul bündelt die GUI-nahe Orchestrierung des Projekts: Es lädt Nachfrage-,
Haltestellen- und Ereignisdaten, bereitet diese für mehrere Analyseansichten auf und
verknüpft die Datenbasis mit Komponenten für Vorhersage, Fahrplanvergleich und KPI-
Berechnung. Der Fokus dieses Abschnitts liegt auf der Grundkonfiguration, der
Datenrepository-Schicht sowie der Initialisierung der Hauptanwendung.

Projektkontext:
    Dieses Modul bildet den Einstiegspunkt der interaktiven Analyseanwendung.
    Es verbindet Datenhaltung, Fachlogik und Visualisierung zu einem gemeinsamen
    Arbeitskontext für Nachfrageanalyse, Kartenansicht, Prediction, Fahrplanvergleich
    und KPI-Auswertung.
"""

import re
import json
import math
import threading
import traceback
import unicodedata
import tkinter as tk
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import tkinter.messagebox as mb


from .config import (
    ADDITIONAL_DATA_DIR,
    BUS_ICON_PATH,
    COST_SETTINGS_PATH,
    MAP_BAR_MAX_FLOW,
    MAP_BAR_REFERENCE_FLOW,
    DATA_DIR,
    EVENT_GEO_PATH,
    GEOJSON_PATH,
    GEOJSON_STATION_ALIASES,
    LINE_LABELS,
    MAP_BUS_FRAME_DELAY_MS,
    MAP_HOUR_PAUSE_MS,
    MAP_STOP_EVENT_WINDOW_MINUTES,
    MAP_TRANSITION_STEPS,
    PALETTE,
    ROUTE_CACHE_PATH,
    WUERZBURG_CENTER,
    WUERZBURG_ZOOM,
)
from .kpis import (
    DEFAULT_BUS_HOURLY_COST_EUR,
    KPIResult,
    aggregate_kpis,
    attach_wvv_hourly_runs,
    calculate_annual_line_kpis_from_parquet_dir,
    calculate_annual_kpis_from_parquet_dir,
    calculate_line_kpis,
    format_kpi_report,
    load_cost_setting,
    save_cost_setting,
)
from .timetable import TimetableComparison, TimetableRepository
from .widgets import AnimatedLineList, DatePicker, dark_date_entry_options
from prediction import DemandPredictionService, PredictionResult, normalize_station_name
from prediction.config import ENRICHED_TRAINING_DIR


import customtkinter as ctk
import pandas as pd
import tkintermapview
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tkcalendar import DateEntry


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


FLEX_X_OPTIONS = ["Stunde", "Datum", "Haltestelle", "Wochentag"]
FLEX_Y_OPTIONS = ["Einstiege", "Ausstiege", "Ein- und Ausstiege", "Fahrten", "Peak-Auslastung", "Ø Auslastung"]
WEEKDAY_LABELS = {
    0: "Mo",
    1: "Di",
    2: "Mi",
    3: "Do",
    4: "Fr",
    5: "Sa",
    6: "So",
}
DEFAULT_FILTER_START_DATE = date(2025, 1, 1)
DEFAULT_FILTER_END_DATE = date(2025, 12, 31)
DEFAULT_PREDICTION_DATE = date(2026, 1, 1)


def normalize_station_name(value: object) -> str:
    """
    Normalisiert Haltestellennamen für robuste Schlüsselvergleiche.

    Die Funktion bereitet Haltestellenbezeichnungen so auf, dass Daten aus
    unterschiedlichen Quellen trotz Schreibvarianten zusammengeführt werden können.
    Dabei werden unter anderem Groß-/Kleinschreibung vereinheitlicht, Umlaute bzw.
    Sonderzeichen reduziert, numerische Suffixe entfernt und nicht-alphanumerische
    Zeichen bereinigt. Das ist insbesondere für das Matching zwischen operativen
    Fahrgastdaten, GeoJSON-Haltestellen und Kartenlogik relevant.

    Parameter:
        value (object): Ursprünglicher Haltestellenname oder ein beliebiges Objekt,
            das in eine Zeichenkette umgewandelt werden kann.

    Rückgabewerte:
        str: Ein bereinigter, vergleichbarer Schlüssel für Haltestellen.

    Fehler/Sonderfälle:
        Bei None wird ein leerer String verarbeitet.
        Unterschiedliche Schreibweisen wie Zusätze mit Ziffern, Bindestrichen,
        Leerzeichen oder Varianten von „Straße/Str.“ werden harmonisiert.

    Projektkontext:
        Die Funktion ist eine zentrale Hilfsroutine für das Datenmatching im
        gesamten System, insbesondere für Kartenansichten, Routenabbildung,
        Geo-Referenzierung und aggregierte Haltestellenanalysen.
    """
    text = "" if value is None else str(value)
    text = re.sub(r"[_\s-]+\d+$", "", text.strip())
    text = text.replace("ß", "ss").replace("ẞ", "SS")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = text.replace("strasse", "str").replace("str.", "str")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text



@dataclass(frozen=True)
class LineFile:
    """
    Beschreibt eine physische Parquet-Datei für eine bestimmte Linie und einen
    Gültigkeitszeitraum.

    Die Klasse modelliert die Zuordnung zwischen einer Liniennummer, dem in der
    Datei enthaltenen Datumsintervall und dem Speicherort der Datei. Als
    unveränderliche Datenstruktur eignet sie sich für die Indexierung der
    verfügbaren Quelldateien innerhalb des Repositories.

    Parameter:
        line (int): Liniennummer, zu der die Datei gehört.
        start (date): Frühestes Datum, das in der Datei enthalten ist.
        end (date): Spätestes Datum, das in der Datei enthalten ist.
        path (Path): Dateipfad zur zugrunde liegenden Parquet-Datei.

    Rückgabewerte:
        LineFile: Eine unveränderliche Datenträgerstruktur für Dateimetadaten.

    Fehler/Sonderfälle:
        Die Klasse selbst validiert keine fachliche Konsistenz; sie setzt voraus,
        dass die Werte beim Erzeugen korrekt aus Dateinamen oder Metadaten
        abgeleitet wurden.

    Projektkontext:
        LineFile ist die Basis der Dateiverwaltung im TransitDataRepository und
        ermöglicht effizientes Auffinden relevanter Quelldateien für Linien- und
        Datumsabfragen.
    """
    line: int
    start: date
    end: date
    path: Path



class TransitDataRepository:
    """
    Stellt die zentrale Datenzugriffsschicht für Linien-, Haltestellen-, Kontext-
    und Ereignisdaten bereit.

    Die Klasse kapselt das Einlesen, Indexieren, Filtern und Aggregieren der
    Rohdaten. Sie abstrahiert die physische Dateistruktur in eine fachliche API,
    über die Dashboard-, Karten-, Prediction- und KPI-Komponenten die benötigten
    Daten in aufbereiteter Form abrufen können.

    Parameter:
        root (Path): Wurzelverzeichnis mit den linienbezogenen Parquet-Dateien.

    Rückgabewerte:
        TransitDataRepository: Initialisierte Repository-Instanz mit Dateindex,
        Cache und ergänzenden Kontextdaten.

    Fehler/Sonderfälle:
        Fehlende Zusatzdateien wie GeoJSON- oder Eventdateien führen nicht zu
        einem Abbruch, sondern zu leeren DataFrames. Dadurch bleibt die Anwendung
        grundsätzlich lauffähig, auch wenn einzelne Analysefunktionen weniger
        Kontextdaten nutzen können.

    Projektkontext:
        Das Repository bildet die Datenbasis des Gesamtsystems. Es liefert sowohl
        Rohdaten als auch verdichtete Sichten für Nachfrageanalyse, Karten-
        animation, Fahrplanvergleich und Vorhersagemodelle.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.line_files = self._index_files()
        self.available_lines = sorted(self.line_files)
        self.min_date, self.max_date = self._date_span()
        self._cache: dict[tuple[int, date, date], pd.DataFrame] = {}
        self.context_daily = self._load_context_daily()
        self.stop_geo = self._load_stop_geojson()
        self.event_geo = self._load_event_geodata()


    def _load_stop_geojson(self) -> pd.DataFrame:
        """
        Lädt und verdichtet GeoJSON-Haltestellendaten zu einer tabellarischen Form.

        Die Methode extrahiert aus einer GeoJSON-Datei ausschließlich Punktgeometrien
        mit gültigem Haltestellennamen und erzeugt daraus eine normalisierte Tabelle
        mit Haltestellenschlüssel, repräsentativem Namen und gemittelten Koordinaten.
        Mehrfachvorkommen derselben Haltestelle werden zusammengefasst, damit
        nachgelagerte Karten- und Matching-Funktionen auf konsistente Geo-Informationen
        zugreifen können.


        Rückgabewerte:
            pd.DataFrame: Tabelle mit den Spalten station_key, geo_name, lat, lon
            und geo_points.

        Fehler/Sonderfälle:
            Existiert die GeoJSON-Datei nicht oder enthält keine verwertbaren
            Punktobjekte, wird ein leerer DataFrame mit erwarteter Spaltenstruktur
            zurückgegeben. Nicht-Punkt-Geometrien werden ignoriert.

        Projektkontext:
            Die Methode stellt die geographische Grundlage für Kartenansichten,
            Haltestellen-Matching und räumliche Nachfragevisualisierung bereit.
        """
        if not GEOJSON_PATH.exists():
            return pd.DataFrame(columns=["station_key", "geo_name", "lat", "lon", "geo_points"])


        with GEOJSON_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)


        rows = []
        for feature in data.get("features", []):
            geometry = feature.get("geometry") or {}
            properties = feature.get("properties") or {}
            if geometry.get("type") != "Point":
                continue
            coords = geometry.get("coordinates") or []
            if len(coords) < 2 or not properties.get("name"):
                continue
            rows.append(
                {
                    "station_key": normalize_station_name(properties.get("name")),
                    "geo_name": properties.get("name"),
                    "lat": float(coords[1]),
                    "lon": float(coords[0]),
                }
            )


        if not rows:
            return pd.DataFrame(columns=["station_key", "geo_name", "lat", "lon", "geo_points"])


        raw = pd.DataFrame(rows).drop_duplicates(subset=["station_key", "lat", "lon"])
        return (
            raw.groupby("station_key", as_index=False)
            .agg(
                geo_name=("geo_name", "first"),
                lat=("lat", "mean"),
                lon=("lon", "mean"),
                geo_points=("geo_name", "count"),
            )
            .sort_values("geo_name")
        )


    def _load_context_daily(self) -> pd.DataFrame:
        """
        Lädt tagesbezogene Kontextmerkmale aus mehreren Zusatzquellen und führt sie
        zu einer gemeinsamen Tagesmatrix zusammen.

        Berücksichtigt werden unter anderem Vorlesungszeiten, gesetzliche Feiertage,
        Schulferien sowie Event- und Konzertinformationen. Die so erzeugte Tabelle
        erweitert die reinen Nachfrage- und Fahrplandaten um externe Einflussfaktoren,
        die für Interpretation, Visualisierung und spätere Modellierung relevant sind.

        

        Rückgabewerte:
            pd.DataFrame: Tagesbezogene Kontexttabelle mit Datumsspalte und
            zusätzlichen binären bzw. aggregierten Merkmalen.

        Fehler/Sonderfälle:
            Fehlende einzelne CSV-Dateien werden toleriert und führen nur dazu,
            dass der jeweilige Kontextblock entfällt. Existieren gar keine
            Kontextdateien, wird ein leerer DataFrame mit der Spalte date
            zurückgegeben. Fehlende Kontextspalten werden mit Standardwerten
            ergänzt.

        Projektkontext:
            Diese Daten erlauben es, Nachfrageverläufe fachlich einzuordnen, etwa
            durch den Einfluss von Vorlesungsbetrieb, Feiertagen oder Veranstaltungen
            auf das Fahrgastaufkommen.
        """
        frames: list[pd.DataFrame] = []


        lecture_path = ADDITIONAL_DATA_DIR / "lectures_daily.csv"
        if lecture_path.exists():
            lecture = pd.read_csv(lecture_path)
            lecture["date"] = pd.to_datetime(lecture["date"]).dt.date
            frames.append(lecture[["date", "lecture_period_jmu"]])


        holiday_path = ADDITIONAL_DATA_DIR / "bavarian_public_holidays_daily.csv"
        if holiday_path.exists():
            holiday = pd.read_csv(holiday_path)
            holiday["date"] = pd.to_datetime(holiday["date"]).dt.date
            frames.append(holiday[["date", "public_holiday", "nationwide"]])


        school_path = ADDITIONAL_DATA_DIR / "bavarian_school_holidays_daily.csv"
        if school_path.exists():
            school = pd.read_csv(school_path)
            school["date"] = pd.to_datetime(school["date"]).dt.date
            frames.append(school[["date", "school_holiday"]])


        event_path = ADDITIONAL_DATA_DIR / "events_daily.csv"
        if event_path.exists():
            event = pd.read_csv(event_path)
            event["date"] = pd.to_datetime(event["date"]).dt.date
            event_daily = event.groupby("date", as_index=False).agg(
                event_hours=("event", "sum"),
                concert_hours=("concert", "sum"),
            )
            frames.append(event_daily)


        event_detail_path = ADDITIONAL_DATA_DIR / "events.csv"
        if event_detail_path.exists():
            event_detail = pd.read_csv(event_detail_path)
            event_detail["start"] = pd.to_datetime(event_detail["start"])
            event_detail["end"] = pd.to_datetime(event_detail["end"])
            event_detail["date"] = event_detail["start"].dt.date
            event_detail["duration_hours"] = (
                (event_detail["end"] - event_detail["start"]).dt.total_seconds() / 3600.0
            ).clip(lower=0)
            detail_daily = event_detail.groupby("date", as_index=False).agg(
                event_count=("name", "count"),
                concert_event_count=("concert", "sum"),
                total_event_duration_hours=("duration_hours", "sum"),
                max_event_duration_hours=("duration_hours", "max"),
            )
            frames.append(detail_daily)


        if not frames:
            return pd.DataFrame(columns=["date"])


        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on="date", how="outer")


        fill_defaults = {
            "lecture_period_jmu": 0,
            "public_holiday": 0,
            "nationwide": 0,
            "school_holiday": 0,
            "event_hours": 0,
            "concert_hours": 0,
            "event_count": 0,
            "concert_event_count": 0,
            "total_event_duration_hours": 0.0,
            "max_event_duration_hours": 0.0,
        }
        for column, default in fill_defaults.items():
            if column not in merged.columns:
                merged[column] = default
        return merged.fillna(fill_defaults)


    def _load_event_geodata(self) -> pd.DataFrame:
        """
        Lädt georeferenzierte Eventdaten für die Kartenvisualisierung.

        Die Methode verarbeitet eine CSV-Datei mit Eventzeitpunkten und Koordinaten,
        prüft Mindestanforderungen an die Datenstruktur und erzeugt daraus eine
        bereinigte Ereignistabelle. Zusätzlich werden Dauer und eine stabile
        event_id berechnet, damit Events zeitabhängig auf der Karte angezeigt und
        eindeutig wiedererkannt werden können.

        

        Rückgabewerte:
            pd.DataFrame: Bereinigte Eventtabelle mit räumlichen und zeitlichen
            Informationen.

        Fehler/Sonderfälle:
            Fehlende Datei, Parserfehler oder fehlende Pflichtspalten führen zu
            einem leeren DataFrame. Ereignisse mit ungültigen Zeitstempeln,
            fehlenden Koordinaten oder negativer Dauer werden ausgeschlossen.

        Projektkontext:
            Die Event-Geodaten erweitern die Nachfragevisualisierung um externe
            Nachfrageimpulse, etwa Konzerte oder andere Veranstaltungen, die in
            der Kartenansicht zeitlich eingeblendet werden können.
        """
        columns = ["event_id", "name", "start", "end", "concert", "lat", "lon", "duration_hours"]
        if not EVENT_GEO_PATH.exists():
            return pd.DataFrame(columns=columns)


        try:
            raw = pd.read_csv(EVENT_GEO_PATH)
        except (OSError, pd.errors.ParserError):
            return pd.DataFrame(columns=columns)


        required = {"name", "start", "end", "latitude", "longitude"}
        if not required.issubset(raw.columns):
            return pd.DataFrame(columns=columns)


        events = raw.copy()
        events["start"] = pd.to_datetime(events["start"], errors="coerce")
        events["end"] = pd.to_datetime(events["end"], errors="coerce")
        events["lat"] = pd.to_numeric(events["latitude"], errors="coerce")
        events["lon"] = pd.to_numeric(events["longitude"], errors="coerce")
        if "concert" not in events:
            events["concert"] = 0
        events["concert"] = pd.to_numeric(events["concert"], errors="coerce").fillna(0).astype(int)
        events = events.dropna(subset=["name", "start", "end", "lat", "lon"])
        events = events[events["end"] >= events["start"]].copy()
        if events.empty:
            return pd.DataFrame(columns=columns)


        events["duration_hours"] = ((events["end"] - events["start"]).dt.total_seconds() / 3600.0).clip(lower=0)
        events["event_id"] = (
            events["name"].astype(str)
            + "|"
            + events["start"].dt.strftime("%Y-%m-%d %H:%M:%S")
            + "|"
            + events["lat"].round(6).astype(str)
            + "|"
            + events["lon"].round(6).astype(str)
        )
        return events[columns].drop_duplicates("event_id").sort_values(["start", "name"]).reset_index(drop=True)


    def _index_files(self) -> dict[int, list[LineFile]]:
        """
        Indiziert alle linienbezogenen Parquet-Dateien im Wurzelverzeichnis.

        Die Methode erkennt Dateinamen anhand eines festen Benennungsschemas,
        extrahiert daraus Zeitraum und Liniennummer und erzeugt für jede Linie eine
        chronologisch sortierte Liste von LineFile-Objekten. Damit kann das
        Repository spätere Datenabfragen effizient auf die relevanten Dateien
        einschränken.

        

        Rückgabewerte:
            dict[int, list[LineFile]]: Abbildung von Liniennummern auf die zugehörigen
            Dateiobjekte.

        Fehler/Sonderfälle:
            Dateien, die nicht dem erwarteten Namensmuster entsprechen, werden
            ignoriert. Bei fehlenden Parquet-Dateien wird ein leeres Mapping
            zurückgegeben.

        Projektkontext:
            Diese Indexstruktur ist die Grundlage für alle datums- und
            linienbasierten Ladevorgänge im Repository.
        """
        mapping: dict[int, list[LineFile]] = {}
        pattern = re.compile(r"data_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})_line_(\d+)\.parquet$")
        for path in sorted(self.root.glob("*.parquet")):
            match = pattern.match(path.name)
            if not match:
                continue
            start = datetime.strptime(match.group(1), "%Y-%m-%d").date()
            end = datetime.strptime(match.group(2), "%Y-%m-%d").date()
            line = int(match.group(3))
            mapping.setdefault(line, []).append(LineFile(line=line, start=start, end=end, path=path))
        for files in mapping.values():
            files.sort(key=lambda item: (item.start, item.end))
        return mapping


    def _date_span(self) -> tuple[date, date]:
        """
        Bestimmt den globalen Datumsbereich aller verfügbaren Liniendateien.

        Die Methode durchsucht den Dateindex und ermittelt frühestes und spätestes
        Datum über alle vorhandenen Dateien hinweg. Dieser Bereich dient unter
        anderem zur Initialisierung sinnvoller Standardfilter in der Oberfläche.

        

        Rückgabewerte:
            tuple[date, date]: Tupel aus Mindest- und Höchstdatum der verfügbaren
            Datenbasis.

        Fehler/Sonderfälle:
            Wenn keine Dateien indexiert wurden, wird als Fallback zweimal das
            aktuelle Tagesdatum zurückgegeben.

        Projektkontext:
            Der Datumsbereich hilft der Anwendung, nur fachlich sinnvolle
            Filterfenster anzubieten und den vorhandenen Datenbestand transparent
            zu machen.
        """
        spans = [item for files in self.line_files.values() for item in files]
        if not spans:
            today = date.today()
            return today, today
        return min(item.start for item in spans), max(item.end for item in spans)


    def label_for_line(self, line: int) -> str:
        """
        Liefert die darstellungsgeeignete Bezeichnung einer Linie.

        Die Methode übersetzt eine numerische Linienkennung in ein lesbares Label
        für GUI, Tabellen und Diagramme. Falls kein vordefiniertes Label vorliegt,
        wird eine generische Bezeichnung erzeugt.

        Parameter:
            line (int): Liniennummer.

        Rückgabewerte:
            str: Benutzerfreundliches Linienlabel.

        Fehler/Sonderfälle:
            Unbekannte Liniennummern führen nicht zu einem Fehler, sondern zu einem
            generischen Fallback im Format „Linie X“.

        Projektkontext:
            Einheitliche Linienlabels verbessern die Lesbarkeit über alle Ansichten
            des Dashboards hinweg.
        """
        return LINE_LABELS.get(line, f"Linie {line}")


    def load_line_range(self, line: int, start: date, end: date) -> pd.DataFrame:
        """
        Lädt Rohdaten einer Linie für einen gegebenen Datumsbereich.

        Die Methode durchsucht den Dateindex nach überlappenden Parquet-Dateien,
        liest nur die fachlich benötigten Spalten ein und filtert anschließend auf
        den gewünschten Zeitraum. Zusätzlich werden abgeleitete Felder wie Linie,
        Datum und Stunde erzeugt. Ergebnisse werden gecacht, um wiederholte
        Abfragen derselben Kombination effizient zu bedienen.

        Parameter:
            line (int): Gewünschte Liniennummer.
            start (date): Startdatum des Analysefensters.
            end (date): Enddatum des Analysefensters.

        Rückgabewerte:
            pd.DataFrame: Gefilterte Rohdaten der Linie im angegebenen Zeitraum.

        Fehler/Sonderfälle:
            Gibt es keine passenden Dateien oder keine Datensätze im Zeitraum, wird
            ein leerer DataFrame mit definierter Spaltenstruktur zurückgegeben.
            Bereits geladene Ergebnisse werden aus dem internen Cache kopiert
            zurückgegeben, um Seiteneffekte zu vermeiden.

        Projektkontext:
            Diese Methode ist der zentrale Zugriffspunkt auf operative Linien-
            Rohdaten und wird von nahezu allen Analysepfaden des Systems genutzt.
        """
        key = (line, start, end)
        if key in self._cache:
            return self._cache[key].copy()


        frames: list[pd.DataFrame] = []
        for file in self.line_files.get(line, []):
            if file.end < start or file.start > end:
                continue
            df = pd.read_parquet(
                file.path,
                columns=[
                    "report_date",
                    "departure_plan_station",
                    "station_number",
                    "station",
                    "journey",
                    "passenger_boarding",
                    "passenger_exiting",
                    "occupancy_departure",
                    "vehicle_utilization",
                ],
            )
            if df.empty:
                continue
            df["report_date"] = pd.to_datetime(df["report_date"])
            df["departure_plan_station"] = pd.to_datetime(df["departure_plan_station"])
            mask = df["report_date"].dt.date.between(start, end)
            if not mask.any():
                continue
            chunk = df.loc[mask].copy()
            chunk["line"] = line
            chunk["date"] = chunk["report_date"].dt.date
            chunk["hour"] = chunk["departure_plan_station"].dt.hour
            frames.append(chunk)


        if not frames:
            empty = pd.DataFrame(
                columns=[
                    "report_date",
                    "departure_plan_station",
                    "station",
                    "journey",
                    "passenger_boarding",
                    "passenger_exiting",
                    "occupancy_departure",
                    "vehicle_utilization",
                    "line",
                    "date",
                    "hour",
                ]
            )
            self._cache[key] = empty
            return empty.copy()


        result = pd.concat(frames, ignore_index=True).sort_values(["report_date", "departure_plan_station"])
        self._cache[key] = result
        return result.copy()


    def aggregate_selection(self, lines: list[int], start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Aggregiert mehrere Linien für Dashboard-Standardauswertungen.

        Aus den Rohdaten der ausgewählten Linien werden drei komplementäre Sichten
        erzeugt: Tagesaggregate, stündliche Nachfrage und haltestellenbezogene
        Summen. Zusätzlich werden Tageskontexte wie Feiertage oder Veranstaltungs-
        informationen eingebunden, damit die Interpretation der Kennzahlen fachlich
        angereichert wird.

        Parameter:
            lines (list[int]): Ausgewählte Linien.
            start (date): Startdatum der Auswertung.
            end (date): Enddatum der Auswertung.

        Rückgabewerte:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Tagesdaten, Stundendaten
            und Haltestellenaggregat.

        Fehler/Sonderfälle:
            Sind keine Daten für die Auswahl verfügbar, werden drei leere
            DataFrames zurückgegeben. Fehlende Kontextspalten werden ergänzt und
            mit Standardwerten gefüllt.

        Projektkontext:
            Diese Aggregation versorgt die klassische Dashboard-Ansicht mit den
            zentralen Kennzahlen und Grunddiagrammen.
        """
        line_frames = [self.load_line_range(line, start, end) for line in lines]
        line_frames = [frame for frame in line_frames if not frame.empty]
        if not line_frames:
            empty = pd.DataFrame()
            return empty, empty, empty


        raw = pd.concat(line_frames, ignore_index=True).sort_values(["line", "journey", "departure_plan_station"]).copy()
        raw["stop_sequence"] = raw.groupby(["line", "journey"]).cumcount() + 1
        daily = (
            raw.groupby(["date", "line"], as_index=False)
            .agg(
                boardings=("passenger_boarding", "sum"),
                journeys=("journey", "nunique"),
                peak_load=("occupancy_departure", "max"),
                avg_utilization=("vehicle_utilization", "mean"),
            )
            .sort_values(["date", "line"])
        )
        if not self.context_daily.empty:
            daily = daily.merge(self.context_daily, on="date", how="left")
        fill_defaults = {
            "lecture_period_jmu": 0,
            "public_holiday": 0,
            "nationwide": 0,
            "school_holiday": 0,
            "event_hours": 0,
            "concert_hours": 0,
        }
        for column, default in fill_defaults.items():
            if column not in daily.columns:
                daily[column] = default
        daily = daily.fillna(fill_defaults)
        hourly = (
            raw.groupby(["hour", "line"], as_index=False)
            .agg(boardings=("passenger_boarding", "sum"))
            .sort_values(["hour", "line"])
        )
        stations = (
            raw.groupby(["line", "station"], as_index=False)
            .agg(
                stop_sequence=("stop_sequence", "median"),
                boardings=("passenger_boarding", "sum"),
                exiting=("passenger_exiting", "sum"),
            )
            .sort_values(["line", "stop_sequence", "station"])
        )
        stations["display_sequence"] = stations.groupby("line").cumcount() + 1
        return daily, hourly, stations


    def aggregate_flexible_chart(
        self,
        lines: list[int],
        start: date,
        end: date,
        x_axis: str,
        y_metric: str,
        line_filter: int | None = None,
    ) -> pd.DataFrame:
        """
        Erzeugt eine flexible Aggregation für frei konfigurierbare Diagramme.

        Je nach gewählter X-Achse werden die Daten nach Stunde, Datum, Haltestelle
        oder Wochentag gruppiert. Die Y-Metrik bestimmt, welche fachliche Größe als
        Diagrammwert verwendet wird, etwa Einstiege, Ausstiege, Fahrtenzahl oder
        Auslastung. Für Haltestellen wird zusätzlich eine Reihenfolge im Linienverlauf
        approximiert, um sinnvolle Diagrammbeschriftungen zu erhalten.

        Parameter:
            lines (list[int]): Zu aggregierende Linien.
            start (date): Startdatum des Analysefensters.
            end (date): Enddatum des Analysefensters.
            x_axis (str): Gewählte Dimension der X-Achse.
            y_metric (str): Gewählte Kennzahl für die Y-Achse.
            line_filter (int | None): Optionale Einschränkung auf eine einzelne Linie.

        Rückgabewerte:
            pd.DataFrame: Diagrammgeeignete Aggregation mit X-Wert, X-Label,
            Kennzahlenwert sowie Ein- und Ausstiegen.

        Fehler/Sonderfälle:
            Fehlen Daten für die Auswahl oder für den optionalen Linienfilter, wird
            ein leerer DataFrame mit Standardspalten zurückgegeben. Unbekannte
            Metriken fallen auf Einstiege zurück.

        Projektkontext:
            Diese Methode versorgt die flexible Analyseansicht, in der fachliche
            Fragestellungen interaktiv entlang verschiedener Dimensionen untersucht
            werden können.
        """
        line_frames = [self.load_line_range(line, start, end) for line in lines]
        line_frames = [frame for frame in line_frames if not frame.empty]
        if not line_frames:
            return pd.DataFrame(columns=["x_value", "x_label", "value", "boardings", "exiting"])


        raw = pd.concat(line_frames, ignore_index=True)
        if line_filter is not None:
            raw = raw[raw["line"] == line_filter].copy()
            if raw.empty:
                return pd.DataFrame(columns=["x_value", "x_label", "value", "boardings", "exiting"])


        if x_axis == "Stunde":
            raw["x_value"] = raw["hour"].astype(int)
            raw["x_label"] = raw["x_value"].map(lambda hour: f"{hour:02d}:00")
            group_columns = ["x_value", "x_label"]
            sort_column = "x_value"
        elif x_axis == "Datum":
            raw["x_value"] = raw["date"]
            raw["x_label"] = raw["date"].map(lambda current: current.isoformat())
            group_columns = ["x_value", "x_label"]
            sort_column = "x_value"
        elif x_axis == "Haltestelle":
            raw = raw.sort_values(["line", "journey", "departure_plan_station"]).copy()
            raw["stop_sequence"] = raw.groupby(["line", "journey"]).cumcount() + 1
            raw["x_value"] = raw["line"].astype(str) + "-" + raw["station"].astype(str)
            group_columns = ["line", "station", "x_value"]
            sort_column = None
        else:
            raw["weekday_number"] = pd.to_datetime(raw["date"]).dt.weekday
            raw["x_value"] = raw["weekday_number"]
            raw["x_label"] = raw["weekday_number"].map(WEEKDAY_LABELS)
            group_columns = ["x_value", "x_label"]
            sort_column = "x_value"


        aggregations = {
            "boardings": ("passenger_boarding", "sum"),
            "exiting": ("passenger_exiting", "sum"),
            "journeys": ("journey", "nunique"),
            "peak_load": ("occupancy_departure", "max"),
            "avg_utilization": ("vehicle_utilization", "mean"),
        }
        if x_axis == "Haltestelle":
            aggregations["stop_sequence"] = ("stop_sequence", "median")


        grouped = (
            raw.groupby(group_columns, as_index=False)
            .agg(**aggregations)
        )
        grouped["saldo"] = grouped["boardings"] - grouped["exiting"]
        if x_axis == "Haltestelle":
            grouped["stop_sequence"] = grouped["stop_sequence"].round().astype(int)


        metric_map = {
            "Einstiege": "boardings",
            "Ausstiege": "exiting",
            "Ein- und Ausstiege": "boardings",
            "Fahrten": "journeys",
            "Peak-Auslastung": "peak_load",
            "Ø Auslastung": "avg_utilization",
        }
        value_column = metric_map.get(y_metric, "boardings")
        grouped["value"] = grouped[value_column].fillna(0)


        if x_axis == "Haltestelle":
            grouped = grouped.sort_values(["line", "stop_sequence", "station"])
            grouped["display_sequence"] = grouped.groupby("line").cumcount() + 1
            if len(lines) > 1:
                grouped["x_label"] = grouped.apply(
                    lambda row: f"L{int(row['line'])} {int(row['display_sequence'])}. {str(row['station'])[:18]}",
                    axis=1,
                )
            else:
                grouped["x_label"] = grouped.apply(
                    lambda row: f"{int(row['display_sequence'])}. {str(row['station'])[:22]}",
                    axis=1,
                )
        elif sort_column is not None:
            grouped = grouped.sort_values(sort_column)
        else:
            grouped = grouped.sort_values("value", ascending=False)
        if x_axis == "Haltestelle":
            grouped = grouped.head(28)
        return grouped[["x_value", "x_label", "value", "boardings", "exiting"]].reset_index(drop=True)


    def aggregate_station_map_data(self, lines: list[int], start: date, end: date) -> pd.DataFrame:
        """
        Aggregiert haltestellenbezogene Nachfrage für die Kartenansicht eines
        Datumsbereichs.

        Die Methode verdichtet Rohdaten zu stationären Nachfragekennzahlen und
        ergänzt, sofern verfügbar, Geodaten aus dem Haltestellenbestand. Zusätzlich
        werden bediente Linien als Textrepräsentation zusammengeführt, damit Marker
        in der Karte inhaltlich erklärt werden können.

        Parameter:
            lines (list[int]): Zu berücksichtigende Linien.
            start (date): Startdatum des Auswertungszeitraums.
            end (date): Enddatum des Auswertungszeitraums.

        Rückgabewerte:
            pd.DataFrame: Haltestellenaggregat mit Nachfrage- und Geoinformationen.

        Fehler/Sonderfälle:
            Ohne Rohdaten wird ein leerer DataFrame zurückgegeben. Ohne verfügbare
            Geodaten bleiben lat/lon leer, sodass die fachlichen Kennzahlen dennoch
            erhalten bleiben.

        Projektkontext:
            Diese Aggregation dient als Grundlage für statische oder kumulierte
            Kartenanalysen der Nachfrage.
        """
        line_frames = [self.load_line_range(line, start, end) for line in lines]
        line_frames = [frame for frame in line_frames if not frame.empty]
        if not line_frames:
            return pd.DataFrame()


        raw = pd.concat(line_frames, ignore_index=True)
        station_demand = (
            raw.groupby(["station_number", "station"], as_index=False)
            .agg(
                boardings=("passenger_boarding", "sum"),
                exiting=("passenger_exiting", "sum"),
                journeys=("journey", "nunique"),
                peak_load=("occupancy_departure", "max"),
                lines=("line", lambda values: ", ".join(str(int(v)) for v in sorted(set(values)))),
            )
            .sort_values("boardings", ascending=False)
        )
        station_demand["station_key"] = station_demand["station"].map(normalize_station_name)
        station_demand["station_key"] = station_demand["station_key"].replace(GEOJSON_STATION_ALIASES)
        if self.stop_geo.empty:
            station_demand["lat"] = pd.NA
            station_demand["lon"] = pd.NA
            station_demand["geo_name"] = ""
            return station_demand
        return station_demand.merge(self.stop_geo, on="station_key", how="left")


    def aggregate_station_hourly_map_data(self, lines: list[int], selected_day: date) -> pd.DataFrame:
        """
        Aggregiert stündliche Haltestellennachfrage für einen einzelnen Tag.

        Im Unterschied zur bereichsbezogenen Kartenaggregation bleibt hier die
        Zeitebene pro Stunde erhalten. Dadurch kann die Nachfrageentwicklung im
        Tagesverlauf kartographisch dargestellt oder animiert werden.

        Parameter:
            lines (list[int]): Zu berücksichtigende Linien.
            selected_day (date): Gewählter Kalendertag.

        Rückgabewerte:
            pd.DataFrame: Stündlich aggregierte Haltestellendaten inklusive
            Geoinformationen, sofern verfügbar.

        Fehler/Sonderfälle:
            Fehlen Roh- oder Geodaten, werden entsprechend leere oder nur teilweise
            georeferenzierte Ergebnisse zurückgegeben.

        Projektkontext:
            Die Methode unterstützt zeitaufgelöste Kartenvisualisierungen und ist
            eine Grundlage für animierte Nachfrageradare.
        """
        line_frames = [self.load_line_range(line, selected_day, selected_day) for line in lines]
        line_frames = [frame for frame in line_frames if not frame.empty]
        if not line_frames:
            return pd.DataFrame()


        raw = pd.concat(line_frames, ignore_index=True)
        hourly_demand = (
            raw.groupby(["station_number", "station", "hour"], as_index=False)
            .agg(
                boardings=("passenger_boarding", "sum"),
                exiting=("passenger_exiting", "sum"),
                journeys=("journey", "nunique"),
                peak_load=("occupancy_departure", "max"),
                lines=("line", lambda values: ", ".join(str(int(v)) for v in sorted(set(values)))),
            )
            .sort_values(["hour", "boardings"], ascending=[True, False])
        )
        hourly_demand["station_key"] = hourly_demand["station"].map(normalize_station_name)
        hourly_demand["station_key"] = hourly_demand["station_key"].replace(GEOJSON_STATION_ALIASES)
        if self.stop_geo.empty:
            hourly_demand["lat"] = pd.NA
            hourly_demand["lon"] = pd.NA
            hourly_demand["geo_name"] = ""
            return hourly_demand
        return hourly_demand.merge(self.stop_geo, on="station_key", how="left")


    def aggregate_station_stop_event_map_data(self, lines: list[int], selected_day: date) -> pd.DataFrame:
        """
        Bereitet minutengenaue Haltestellenereignisse für die kombinierte
        Kartenanimation auf.

        Die Methode verknüpft operative Stopdaten mit Geodaten und berechnet aus den
        geplanten Abfahrtszeiten eine minute_of_day-Kennzahl. Berücksichtigt werden
        nur Stopps mit tatsächlichem Ein- oder Aussteigeraufkommen, damit die
        Animation fachlich relevante Aktivität statt bloßer Fahrplanpunkte zeigt.

        Parameter:
            lines (list[int]): Zu berücksichtigende Linien.
            selected_day (date): Tag der Kartenanimation.

        Rückgabewerte:
            pd.DataFrame: Ereignisorientierte Stopptabelle mit räumlicher Lage und
            minutengenauer zeitlicher Position.

        Fehler/Sonderfälle:
            Fehlen Rohdaten oder Geodaten, wird ein leerer DataFrame zurückgegeben.
            Stopps ohne Fahrgastwechsel werden entfernt, um visuelles Rauschen in
            der Animation zu reduzieren.

        Projektkontext:
            Diese Methode versorgt die animierte Kartenansicht mit den zeitlich
            fein aufgelösten Haltestellenereignissen, die Nachfragebewegungen im
            Tagesverlauf sichtbar machen.
        """
        line_frames = [self.load_line_range(line, selected_day, selected_day) for line in lines]
        line_frames = [frame for frame in line_frames if not frame.empty]
        if not line_frames or self.stop_geo.empty:
            return pd.DataFrame()


        raw = pd.concat(line_frames, ignore_index=True).copy()
        raw["station_key"] = raw["station"].map(normalize_station_name).replace(GEOJSON_STATION_ALIASES)
        raw = raw.merge(self.stop_geo, on="station_key", how="left")
        raw = raw.dropna(subset=["lat", "lon"]).copy()
        if raw.empty:
            return pd.DataFrame()


        planned = pd.to_datetime(raw["departure_plan_station"])
        raw["minute_of_day"] = planned.dt.hour * 60 + planned.dt.minute + planned.dt.second / 60.0
        raw["boardings"] = raw["passenger_boarding"].fillna(0).astype(float)
        raw["exiting"] = raw["passenger_exiting"].fillna(0).astype(float)
        raw = raw[(raw["boardings"] > 0) | (raw["exiting"] > 0)].copy()
        if raw.empty:
            return pd.DataFrame()


        return (
            raw.groupby(
                [
                    "line",
                    "journey",
                    "station_number",
                    "station",
                    "station_key",
                    "lat",
                    "lon",
                    "minute_of_day",
                ],
                as_index=False,
            )
            .agg(
                boardings=("boardings", "sum"),
                exiting=("exiting", "sum"),
                departure_plan_station=("departure_plan_station", "min"),
            )
            .sort_values(["minute_of_day", "line", "journey", "station_number"])
        )


    def journey_options_for_day(self, line: int, selected_day: date) -> pd.DataFrame:
        """
        Ermittelt auswählbare Fahrten einer Linie für einen bestimmten Tag.

        Die Methode fasst Rohdaten je journey zusammen und erzeugt für die GUI eine
        lesbare Beschriftung mit Startzeit, Endzeit, Fahrtkennung und Stopanzahl.
        Damit kann eine konkrete Fahrt für Detailansichten oder Animationen
        ausgewählt werden.

        Parameter:
            line (int): Liniennummer.
            selected_day (date): Gewählter Kalendertag.

        Rückgabewerte:
            pd.DataFrame: Fahrtauswahl mit journey, Start-/Endzeit, Stopanzahl und
            darstellungsgeeignetem Label.

        Fehler/Sonderfälle:
            Sind für Linie und Tag keine Daten vorhanden, wird ein leerer DataFrame
            mit definierter Spaltenstruktur zurückgegeben.

        Projektkontext:
            Die Methode verbindet operative Fahrtdaten mit der Interaktionslogik der
            GUI, insbesondere für die Auswahl animierbarer Einzelrouten.
        """
        raw = self.load_line_range(line, selected_day, selected_day)
        if raw.empty:
            return pd.DataFrame(columns=["journey", "start_time", "end_time", "stop_count", "label"])


        grouped = (
            raw.sort_values("departure_plan_station")
            .groupby("journey", as_index=False)
            .agg(
                start_time=("departure_plan_station", "min"),
                end_time=("departure_plan_station", "max"),
                stop_count=("station", "count"),
            )
            .sort_values("start_time")
        )
        grouped["label"] = grouped.apply(
            lambda row: (
                f"{pd.Timestamp(row['start_time']).strftime('%H:%M')}-"
                f"{pd.Timestamp(row['end_time']).strftime('%H:%M')} | Fahrt {row['journey']} | "
                f"{int(row['stop_count'])} Stops"
            ),
            axis=1,
        )
        return grouped


    def journey_route_for_day(self, line: int, selected_day: date, journey: object) -> pd.DataFrame:
        """
        Liefert die georeferenzierte Haltestellenfolge einer konkreten Fahrt.

        Die Methode filtert die Rohdaten auf eine bestimmte Fahrt, ordnet die
        Haltestellen geographisch zu und berechnet die Position der Stopps im
        Tagesverlauf als minute_of_day. Das Ergebnis ist für Linienanimation,
        Routendarstellung und Detailvergleiche geeignet.

        Parameter:
            line (int): Liniennummer.
            selected_day (date): Gewählter Kalendertag.
            journey (object): Fahrtkennung, die mit den Rohdaten verglichen wird.

        Rückgabewerte:
            pd.DataFrame: Routentabelle mit Linien-, Fahrt-, Zeit- und
            Geoinformationen.

        Fehler/Sonderfälle:
            Fehlen Rohdaten, ist die Fahrt unbekannt oder lassen sich keine
            Haltestellen geographisch zuordnen, wird ein leerer DataFrame mit
            Standardspalten zurückgegeben.

        Projektkontext:
            Die Methode liefert die fachliche Grundlage für busbezogene
            Einzelanimations- und Routenansichten im Dashboard.
        """
        raw = self.load_line_range(line, selected_day, selected_day)
        if raw.empty:
            return pd.DataFrame(columns=["station", "departure_plan_station", "minute_of_day", "lat", "lon"])


        route = raw[raw["journey"].astype(str) == str(journey)].copy()
        if route.empty:
            return pd.DataFrame(columns=["station", "departure_plan_station", "minute_of_day", "lat", "lon"])


        route = route.sort_values("departure_plan_station").copy()
        route["station_key"] = route["station"].map(normalize_station_name).replace(GEOJSON_STATION_ALIASES)
        route = route.merge(self.stop_geo, on="station_key", how="left")
        route = route.dropna(subset=["lat", "lon"]).copy()
        if route.empty:
            return pd.DataFrame(columns=["station", "departure_plan_station", "minute_of_day", "lat", "lon"])


        route["minute_of_day"] = (
            pd.to_datetime(route["departure_plan_station"]).dt.hour * 60
            + pd.to_datetime(route["departure_plan_station"]).dt.minute
        )
        route["stop_sequence"] = range(1, len(route) + 1)
        return route[
            [
                "line",
                "journey",
                "stop_sequence",
                "station",
                "departure_plan_station",
                "minute_of_day",
                "lat",
                "lon",
            ]
        ].reset_index(drop=True)


    def fleet_schedule_for_day(self, lines: list[int], selected_day: date) -> pd.DataFrame:
        """
        Erstellt einen georeferenzierten Tagesfahrplan mehrerer Linien für die
        Flottenanimation.

        Im Gegensatz zur Einzelroutenabfrage werden hier alle Fahrten der gewählten
        Linien für einen Tag zusammengeführt. Die Methode berechnet eine zeitliche
        Position im Tagesverlauf und stellt die notwendigen Standortinformationen für
        parallele Fahrzeugdarstellungen auf der Karte bereit.

        Parameter:
            lines (list[int]): Ausgewählte Linien.
            selected_day (date): Gewählter Kalendertag.

        Rückgabewerte:
            pd.DataFrame: Georeferenzierter Tagesfahrplan mehrerer Fahrten.

        Fehler/Sonderfälle:
            Ohne Rohdaten oder Geodaten wird ein leerer DataFrame mit definierter
            Spaltenstruktur zurückgegeben.

        Projektkontext:
            Diese Methode ist die Basis für die Visualisierung simultan aktiver
            Fahrzeuge im Netz und unterstützt damit die Systemperspektive des
            Verkehrsangebots.
        """
        line_frames = [self.load_line_range(line, selected_day, selected_day) for line in lines]
        line_frames = [frame for frame in line_frames if not frame.empty]
        if not line_frames or self.stop_geo.empty:
            return pd.DataFrame(
                columns=[
                    "line",
                    "journey",
                    "station",
                    "departure_plan_station",
                    "minute_of_day",
                    "lat",
                    "lon",
                ]
            )


        raw = pd.concat(line_frames, ignore_index=True).copy()
        raw["station_key"] = raw["station"].map(normalize_station_name).replace(GEOJSON_STATION_ALIASES)
        raw = raw.merge(self.stop_geo, on="station_key", how="left")
        raw = raw.dropna(subset=["lat", "lon"]).copy()
        if raw.empty:
            return pd.DataFrame(
                columns=[
                    "line",
                    "journey",
                    "station",
                    "departure_plan_station",
                    "minute_of_day",
                    "lat",
                    "lon",
                ]
            )


        planned = pd.to_datetime(raw["departure_plan_station"])
        raw["minute_of_day"] = planned.dt.hour * 60 + planned.dt.minute + planned.dt.second / 60.0
        raw = raw.sort_values(["line", "journey", "departure_plan_station"])
        return raw[
            [
                "line",
                "journey",
                "station",
                "departure_plan_station",
                "minute_of_day",
                "lat",
                "lon",
            ]
        ].reset_index(drop=True)



class TransitDashboard(ctk.CTk):
    """
    Hauptfenster der Analyseanwendung für Nachfrage, Prediction, Fahrplanvergleich
    und KPI-Auswertung.

    Die Klasse orchestriert die grafische Benutzeroberfläche und verbindet die
    Datenbasis des Repositories mit spezialisierten Komponenten wie Prediction-
    Service, Fahrplanrepository und Kartenmodulen. Bereits im Konstruktor werden
    zentrale Zustände, Steuerelemente, Caches und Standardauswahlen vorbereitet,
    damit die verschiedenen Teilansichten konsistent zusammenarbeiten können.

    Parameter:
        repo (TransitDataRepository): Datenrepository mit Zugriff auf operative,
            geographische und kontextuelle Projektdaten.

    Rückgabewerte:
        TransitDashboard: Initialisierte GUI-Anwendung.

    Fehler/Sonderfälle:
        Einzelne Teilkomponenten dürfen initial leer sein, sofern zugehörige Daten
        oder Widgets erst später aufgebaut werden. Die Klasse arbeitet daher an
        vielen Stellen bewusst mit optionalen Attributen.

    Projektkontext:
        TransitDashboard ist die zentrale Benutzeroberfläche des Projekts und
        bündelt sämtliche fachlichen Perspektiven – von Nachfrageanalyse über
        Kartenanimation bis zur adaptiven Fahrplan- und KPI-Bewertung.
    """

    def __init__(self, repo: TransitDataRepository) -> None:
        """
        Initialisiert das Hauptfenster und den globalen Anwendungszustand.

        Der Konstruktor setzt Repository, Prediction-Service und Fahrplanrepository,
        definiert Standardlinien sowie Datumsbereiche und initialisiert alle für die
        verschiedenen Tabs benötigten Zustandsvariablen, Widgetspeicher und Caches.
        Anschließend wird die Hauptoberfläche aufgebaut und eine erste
        Dashboard-Aktualisierung ausgelöst.

        Parameter:
            repo (TransitDataRepository): Zentrale Datenquelle der Anwendung.

        Rückgabewerte:
            None: Die Methode erzeugt und initialisiert die GUI-Instanz.

        Fehler/Sonderfälle:
            Falls bevorzugte Standardlinien nicht vorhanden sind, wird auf die
            ersten verfügbaren Linien zurückgegriffen. Viele Widget-Referenzen sind
            zunächst None, bis die zugehörigen Oberflächenelemente aufgebaut wurden.

        Projektkontext:
            Die Initialisierung schafft die gemeinsame Laufzeitumgebung für alle
            interaktiven Analyse- und Entscheidungsunterstützungsfunktionen des
            Systems.
        """
        super().__init__()
        self.repo = repo
        self.prediction_service = DemandPredictionService(repo)
        self.timetable_repo = TimetableRepository()
        self.selected_lines = [line for line in [10, 20, 27] if line in repo.available_lines] or repo.available_lines[:3]


        start_default = DEFAULT_FILTER_START_DATE
        end_default = DEFAULT_FILTER_END_DATE


        self.title("WVV Adaptive Network Studio")
        self.geometry("1560x920")
        self.minsize(1380, 840)
        self.configure(fg_color=PALETTE["page_bg"])


        self.line_choice = ctk.StringVar(value="")
        self.map_line_choice = ctk.StringVar(value="")
        self.flex_x_choice = ctk.StringVar(value="Stunde")
        self.flex_y_choice = ctk.StringVar(value="Einstiege")
        self.flex_line_choice = ctk.StringVar(value="")
        self.flex_compare_choice = ctk.StringVar(value="Kein Vergleich")
        self.status_text = ctk.StringVar(value="Bereit für deine Analyse")
        self.show_lectures = ctk.BooleanVar(value=False)
        self.show_events = ctk.BooleanVar(value=False)
        self.show_holidays = ctk.BooleanVar(value=False)
        self.show_school_holidays = ctk.BooleanVar(value=False)
        self.metric_cards: dict[str, ctk.CTkLabel] = {}
        self.line_chip_frame: ctk.CTkFrame | None = None
        self.line_list: AnimatedLineList | None = None
        self.station_textbox: ctk.CTkTextbox | None = None
        self.insight_textbox: ctk.CTkTextbox | None = None
        self.daily_chart_insight: ctk.CTkLabel | None = None
        self.flex_chart_insight: ctk.CTkLabel | None = None
        self.flex_host: ctk.CTkFrame | None = None
        self.flex_line_menu: ctk.CTkOptionMenu | None = None
        self.flex_line_lookup: dict[str, int] = {}
        self.flex_compare_menu: ctk.CTkOptionMenu | None = None
        self.flex_compare_lookup: dict[str, int] = {}
        self.dashboard_update_after_id: str | None = None
        self.filter_shell: ctk.CTkFrame | None = None
        self.line_chips_panel: ctk.CTkFrame | None = None
        self.map_widget: tkintermapview.TkinterMapView | None = None
        self.map_status_label: ctk.CTkLabel | None = None
        self.map_day_entry: DateEntry | None = None
        self.map_line_menu: ctk.CTkOptionMenu | None = None
        self.map_line_lookup: dict[str, int | None] = {}
        self.map_hour_label: ctk.CTkLabel | None = None
        self.map_hour_slider: ctk.CTkSlider | None = None
        self.map_play_button: ctk.CTkButton | None = None
        self.map_speed_label: ctk.CTkLabel | None = None
        self.map_speed_slider: ctk.CTkSlider | None = None
        self.map_speed_factor = 1.0
        self.map_station_textbox: ctk.CTkTextbox | None = None
        self.prediction_day_entry: DateEntry | None = None
        self.prediction_hour_label: ctk.CTkLabel | None = None
        self.prediction_hour_slider: ctk.CTkSlider | None = None
        self.prediction_line_choice = ctk.StringVar(value="")
        self.prediction_mode_choice = ctk.StringVar(value="Nächste Stunden")
        self.prediction_horizon_choice = ctk.StringVar(value="4 Stunden")
        self.prediction_station_choice = ctk.StringVar(value="Alle Haltestellen")
        self.prediction_stop_count_choice = ctk.StringVar(value="5 Haltestellen")
        self.prediction_line_menu: ctk.CTkOptionMenu | None = None
        self.prediction_mode_menu: ctk.CTkOptionMenu | None = None
        self.prediction_horizon_box: ctk.CTkFrame | None = None
        self.prediction_station_box: ctk.CTkFrame | None = None
        self.prediction_stop_count_box: ctk.CTkFrame | None = None
        self.prediction_horizon_menu: ctk.CTkOptionMenu | None = None
        self.prediction_station_menu: ctk.CTkOptionMenu | None = None
        self.prediction_stop_count_menu: ctk.CTkOptionMenu | None = None
        self.prediction_line_lookup: dict[str, int] = {}
        self.prediction_station_lookup: dict[str, str] = {}
        self.prediction_station_rows: list[dict[str, object]] = []
        self.prediction_metric_labels: dict[str, ctk.CTkLabel] = {}
        self.prediction_chart_host: ctk.CTkFrame | None = None
        self.prediction_textbox: ctk.CTkTextbox | None = None
        self.prediction_status_label: ctk.CTkLabel | None = None
        self.prediction_train_button: ctk.CTkButton | None = None
        self.prediction_predict_button: ctk.CTkButton | None = None
        self.prediction_busy = False
        self.prediction_busy_after_id: str | None = None
        self.prediction_busy_message = ""
        self.prediction_busy_step = 0
        self.timetable_line_choice = ctk.StringVar(value="")
        self.timetable_horizon_choice = ctk.StringVar(value="4 Stunden")
        self.timetable_line_menu: ctk.CTkOptionMenu | None = None
        self.timetable_day_entry: DateEntry | None = None
        self.timetable_hour_label: ctk.CTkLabel | None = None
        self.timetable_hour_slider: ctk.CTkSlider | None = None
        self.timetable_status_label: ctk.CTkLabel | None = None
        self.timetable_route_canvas: tk.Canvas | None = None
        self.timetable_wvv_textbox: ctk.CTkTextbox | None = None
        self.timetable_adaptive_textbox: ctk.CTkTextbox | None = None
        self.timetable_calculate_button: ctk.CTkButton | None = None
        self.timetable_line_lookup: dict[str, int] = {}
        self.timetable_busy = False
        self.bus_hourly_cost = ctk.StringVar(value=self._format_cost_input(load_cost_setting(COST_SETTINGS_PATH)))
        self.kpi_cost_entry: ctk.CTkEntry | None = None
        self.kpi_annual_cache: dict[tuple[int, float], KPIResult] = {}
        self.kpi_ranking_cache: dict[tuple[int, float], list[KPIResult]] = {}
        self.kpi_day_entry: DateEntry | None = None
        self.kpi_line_choice = ctk.StringVar(value="")
        self.kpi_line_menu: ctk.CTkOptionMenu | None = None
        self.kpi_line_lookup: dict[str, int] = {}
        self.kpi_status_label: ctk.CTkLabel | None = None
        self.kpi_total_textbox: ctk.CTkTextbox | None = None
        self.kpi_line_textbox: ctk.CTkTextbox | None = None
        self.kpi_ranking_boxes: dict[str, ctk.CTkTextbox] = {}
        self.kpi_busy = False
        self.map_fleet_schedule = pd.DataFrame()
        self.map_fleet_trips: list[dict[str, object]] = []
        self.map_fleet_markers: dict[str, object] = {}
        self.map_fleet_icon: tk.PhotoImage | None = None
        self.map_fleet_route_cache = self._load_route_cache()
        self.map_fleet_route_failures: set[str] = set()
        self.map_fleet_cache_dirty = False
        self.map_markers = []
        self.map_marker_by_station = {}
        self.map_bar_icons: list[tk.PhotoImage] = []
        self.map_bar_icon_cache: dict[tuple[int, int], tk.PhotoImage] = {}
        self.map_event_markers: dict[str, object] = {}
        self.map_event_icon_cache: dict[tuple[bool, int], tk.PhotoImage] = {}
        self.map_animation_data = pd.DataFrame()
        self.map_current_hour = 8
        self.map_transition_from_hour = 8
        self.map_animation_running = False
        self.map_animation_after_id: str | None = None
        self.map_ignore_slider_update = False
        self.sidebar_buttons: dict[str, ctk.CTkButton] = {}
        self.section_targets: dict[str, ctk.CTkBaseClass] = {}


        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)


        self._build_sidebar()
        self._build_main_area(start_default, end_default)
        self._refresh_line_menu()
        self.update_dashboard()
        self.after(120, self._enter_startup_fullscreen)


    def _enter_startup_fullscreen(self) -> None:
        """
        Wechselt die Anwendung nach dem Start in einen maximierten Darstellungsmodus.

        Die Methode versucht zunächst den plattformspezifischen Zoom-/Maximierungs-
        zustand zu setzen. Falls dies nicht unterstützt wird, erfolgt ein Fallback
        auf einen Vollbildmodus. Damit wird unabhängig vom Betriebssystem eine
        möglichst große Arbeitsfläche für die datenintensive GUI angestrebt.

        

        Rückgabewerte:
            None: Die Methode verändert ausschließlich den Fensterzustand.

        Fehler/Sonderfälle:
            Nicht unterstützte Fensterattribute werden tolerant behandelt. Schlagen
            beide Varianten fehl, bleibt die bestehende Fenstergröße erhalten.

        Projektkontext:
            Die große Darstellungsfläche ist für Karten, Diagramme, Textboxen und
            Vergleichsansichten der Anwendung fachlich sinnvoll und verbessert die
            Nutzbarkeit im Analysekontext.
        """
        try:
            self.wm_state("zoomed")
        except Exception:
            try:
                self.attributes("-fullscreen", True)
            except Exception:
                pass


    def _build_sidebar(self) -> None:
        """
        Erstellt die linke Navigationsleiste der Anwendung.

        Die Sidebar bündelt den Wechsel zwischen den zentralen Arbeitsbereichen des
        Systems, darunter Dashboard, Kartenansicht, Prediction, Fahrplan und KPI-
        Vergleich. Zusätzlich wird eine Statusanzeige eingeblendet, die den Nutzer
        über den aktuellen Analysezustand informiert.

        

        Rückgabewerte:
            None: Die Methode erzeugt und platziert GUI-Elemente.

        Fehler/Sonderfälle:
            Es werden keine fachlichen Daten vorausgesetzt; die Sidebar kann daher
            bereits vor dem vollständigen Aufbau aller Hauptansichten erstellt
            werden.

        Projektkontext:
            Die Sidebar strukturiert die Anwendung in fachliche Teilperspektiven
            und ist damit das primäre Navigationsinstrument des Dashboards.
        """
        sidebar = ctk.CTkFrame(
            self,
            width=218,
            fg_color=PALETTE["sidebar"],
            corner_radius=28,
            border_width=1,
            border_color=PALETTE["border"],
        )
        sidebar.grid(row=0, column=0, sticky="ns", padx=(14, 10), pady=14)
        sidebar.grid_propagate(False)


        self._sidebar_button(sidebar, "Dashboard", command=lambda: self._show_view("dashboard"), active=True)
        self._sidebar_button(sidebar, "Kartenansicht", command=lambda: self._show_view("map"))
        self._sidebar_button(sidebar, "Prediction", command=lambda: self._show_view("prediction"))
        self._sidebar_button(sidebar, "Fahrplan", command=lambda: self._show_view("timetable"))
        self._sidebar_button(sidebar, "KPI Vergleich", command=lambda: self._show_view("kpi"))


        ctk.CTkLabel(
            sidebar,
            textvariable=self.status_text,
            text_color=PALETTE["muted"],
            wraplength=162,
            justify="left",
            font=ctk.CTkFont(family="Segoe UI", size=11),
        ).pack(side="bottom", anchor="w", padx=18, pady=(0, 18))

        def _sidebar_button(self, master, text: str, command, active: bool = False) -> None:
            """
            Erzeugt einen Navigationsbutton für die Sidebar und registriert ihn intern.

            Die Methode kapselt die einheitliche Gestaltung der Sidebar-Buttons, damit
            alle Navigationspunkte visuell konsistent aufgebaut werden. Zusätzlich wird
            der Button in einem internen Wörterbuch gespeichert, sodass spätere Zustands-
            wechsel wie Aktivierung, Deaktivierung oder visuelle Hervorhebung zentral
            gesteuert werden können.

        Parameter:
                master: Eltern-Widget, in das der Button eingefügt wird.
                text (str): Sichtbarer Text des Navigationsbuttons.
                command: Callback-Funktion, die beim Klick auf den Button ausgeführt wird.
                active (bool): Gibt an, ob der Button initial als aktive Ansicht
                    dargestellt werden soll.

            Rückgabewerte:
                None: Die Methode erzeugt und platziert ausschließlich ein GUI-Element.

            Fehler/Sonderfälle:
                Die Methode validiert den Callback nicht explizit und setzt voraus, dass
                ein aufrufbares Objekt übergeben wird. Der Button wird über den
                kleingeschriebenen Textschlüssel abgelegt; gleichlautende Beschriftungen
                würden einen vorhandenen Eintrag überschreiben.

            Projektkontext:
                Die Methode ist ein zentrales Hilfsmittel für die Navigation zwischen den
                fachlichen Hauptbereichen des Dashboards wie Dashboard, Kartenansicht,
                Prediction, Fahrplan und KPI-Vergleich.
            """
        button = ctk.CTkButton(
            master,
            text=text,
            anchor="w",
            height=38,
            corner_radius=14,
            fg_color=PALETTE["sidebar_active"] if active else PALETTE["accent_soft"],
            hover_color=PALETTE["accent_dark"] if active else PALETTE["surface_alt"],
            text_color="white" if active else PALETTE["text"],
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold" if active else "normal"),
            command=command,
        )
        button.pack(fill="x", padx=14, pady=5)
        self.sidebar_buttons[text.lower()] = button


    def _build_main_area(self, start_default: date, end_default: date) -> None:
        """
        Baut den zentralen Inhaltsbereich der Anwendung mit allen Hauptansichten auf.

        Die Methode erstellt den scrollbaren Hauptcontainer, den Filterbereich,
        die Dashboard-Ansicht sowie die Container für Karten-, Prediction-,
        Fahrplan- und KPI-Ansicht. Zusätzlich werden Kennzahlenkarten, Diagramm-
        bereiche, flexible Analysekomponenten und textuelle Ausgabefelder
        initialisiert. Damit bildet sie das strukturelle Rückgrat der gesamten
        Benutzeroberfläche.

        Parameter:
            start_default (date): Standardwert für das Startdatum im Filterbereich.
            end_default (date): Standardwert für das Enddatum im Filterbereich.

        Rückgabewerte:
            None: Die Methode erzeugt und arrangiert GUI-Strukturen.

        Fehler/Sonderfälle:
            Viele Unterelemente werden zunächst leer angelegt und erst durch spätere
            Aktualisierungsmethoden mit Daten befüllt. Nicht aktive Tabs werden direkt
            nach dem Aufbau ausgeblendet, bleiben aber als Container verfügbar.

        Projektkontext:
            Die Methode verknüpft die fachlichen Analyseperspektiven des Systems in
            einer gemeinsamen Oberfläche und sorgt dafür, dass Filter, Diagramme,
            Texte und Spezialansichten kohärent zusammenarbeiten.
        """
        main = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            corner_radius=0,
            scrollbar_button_color=PALETTE["accent"],
            scrollbar_button_hover_color=PALETTE["accent_dark"],
        )
        self.main_scroll = main
        main.grid(row=0, column=1, sticky="nsew", padx=(0, 18), pady=18)
        main.grid_columnconfigure(0, weight=1)


        shell = ctk.CTkFrame(
            main,
            fg_color=PALETTE["surface_soft"],
            corner_radius=32,
            border_width=2,
            border_color=PALETTE["border"],
        )
        shell.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        self.filter_shell = shell
        self.section_targets["dashboard"] = shell
        self.section_targets["kalender"] = shell
        shell.grid_columnconfigure(0, weight=1)


        accent_strip = ctk.CTkFrame(shell, fg_color=PALETTE["accent"], corner_radius=999, height=5)
        accent_strip.grid(row=0, column=0, sticky="ew", padx=24, pady=(18, 0))
        accent_strip.grid_propagate(False)


        control_grid = ctk.CTkFrame(shell, fg_color="transparent")
        control_grid.grid(row=1, column=0, sticky="ew", padx=24, pady=(16, 18))
        control_grid.grid_columnconfigure(0, weight=1)
        control_grid.grid_columnconfigure(1, weight=3)


        period_frame = ctk.CTkFrame(
            control_grid,
            fg_color=PALETTE["surface"],
            corner_radius=24,
            border_width=1,
            border_color=PALETTE["border"],
        )
        period_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        period_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            period_frame,
            text="Zeitraum",
            text_color=PALETTE["muted"],
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(14, 6))
        self.start_picker = DatePicker(period_frame, "Startdatum", start_default)
        self.start_picker.grid(row=1, column=0, padx=14, pady=(0, 8), sticky="ew")
        self.end_picker = DatePicker(period_frame, "Enddatum", end_default)
        self.end_picker.grid(row=2, column=0, padx=14, pady=(0, 14), sticky="ew")
        for picker in (self.start_picker, self.end_picker):
            picker.entry.bind("<<DateEntrySelected>>", self._schedule_dashboard_update, add="+")


        line_frame = ctk.CTkFrame(
            control_grid,
            fg_color=PALETTE["surface"],
            corner_radius=24,
            border_width=1,
            border_color=PALETTE["border"],
        )
        line_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=(0, 10))
        ctk.CTkLabel(
            line_frame,
            text="Linienauswahl  |  Klick fügt direkt hinzu",
            text_color=PALETTE["muted"],
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        ).pack(anchor="w", padx=16, pady=(14, 4))
        self.line_list = AnimatedLineList(
            line_frame,
            command=self._select_line_from_list,
            height=104,
        )
        self.line_list.pack(fill="both", expand=True, padx=16, pady=(0, 14))
        context_frame = ctk.CTkFrame(control_grid, fg_color=PALETTE["surface"], corner_radius=22, border_width=1, border_color=PALETTE["border"])
        context_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        context_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self._context_toggle(context_frame, 0, "Vorlesungen", self.show_lectures)
        self._context_toggle(context_frame, 1, "Events", self.show_events)
        self._context_toggle(context_frame, 2, "Feiertage", self.show_holidays)
        self._context_toggle(context_frame, 3, "Ferien", self.show_school_holidays)


        chips = ctk.CTkFrame(main, fg_color="transparent")
        chips.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.line_chips_panel = chips
        ctk.CTkLabel(
            chips,
            text="Aktive Linien",
            text_color=PALETTE["muted"],
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
        ).pack(anchor="w")
        self.line_chip_frame = ctk.CTkFrame(chips, fg_color="transparent")
        self.line_chip_frame.pack(fill="x", pady=(6, 0))


        content = ctk.CTkFrame(main, fg_color="transparent")
        content.grid(row=2, column=0, sticky="ew")
        content.grid_columnconfigure(0, weight=1)
        self.dashboard_frame = content


        map_tab = ctk.CTkFrame(main, fg_color="transparent")
        map_tab.grid(row=2, column=0, sticky="ew")
        map_tab.grid_columnconfigure(0, weight=1)
        self.map_frame = map_tab
        map_tab.grid_remove()


        prediction_tab = ctk.CTkFrame(main, fg_color="transparent")
        prediction_tab.grid(row=2, column=0, sticky="ew")
        prediction_tab.grid_columnconfigure(0, weight=1)
        self.prediction_frame = prediction_tab
        prediction_tab.grid_remove()


        timetable_tab = ctk.CTkFrame(main, fg_color="transparent")
        timetable_tab.grid(row=2, column=0, sticky="ew")
        timetable_tab.grid_columnconfigure(0, weight=1)
        self.timetable_frame = timetable_tab
        timetable_tab.grid_remove()


        kpi_tab = ctk.CTkFrame(main, fg_color="transparent")
        kpi_tab.grid(row=2, column=0, sticky="ew")
        kpi_tab.grid_columnconfigure(0, weight=1)
        self.kpi_frame = kpi_tab
        kpi_tab.grid_remove()


        metrics = ctk.CTkFrame(content, fg_color="transparent")
        metrics.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        metrics.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self._metric_card(metrics, 0, "Gesamt Einstiege", "0")
        self._metric_card(metrics, 1, "Tagesdurchschnitt", "0")
        self._metric_card(metrics, 2, "Peak Day", "0")
        self._metric_card(metrics, 3, "Aktive Fahrten", "0")


        chart_card = self._card(content, row=1, title="Daily Demand Stream", subtitle="Einstiege pro Tag und Linie")
        self.section_targets["nachfrage"] = chart_card
        self.chart_host = ctk.CTkFrame(chart_card, fg_color="transparent")
        self.chart_host.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.chart_host.configure(height=420)
        self.chart_host.pack_propagate(False)
        self.daily_chart_insight = self._inline_insight_label(chart_card)


        flex_card = self._card(content, row=2, title="Flexible Analyse", subtitle="Baue dir eine Ansicht aus Metrik, Achse und Linie")
        flex_controls = ctk.CTkFrame(flex_card, fg_color=PALETTE["surface_alt"], corner_radius=18)
        flex_controls.pack(fill="x", padx=12, pady=(0, 12))
        flex_controls.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self.flex_y_menu = self._builder_menu(
            flex_controls,
            column=0,
            label="Analysiere",
            variable=self.flex_y_choice,
            values=FLEX_Y_OPTIONS,
        )
        self.flex_x_menu = self._builder_menu(
            flex_controls,
            column=1,
            label="nach",
            variable=self.flex_x_choice,
            values=FLEX_X_OPTIONS,
        )
        self.flex_line_menu = self._builder_menu(
            flex_controls,
            column=2,
            label="für Linie",
            variable=self.flex_line_choice,
            values=["Keine Linie"],
        )
        self.flex_compare_menu = self._builder_menu(
            flex_controls,
            column=3,
            label="Vergleich",
            variable=self.flex_compare_choice,
            values=["Kein Vergleich"],
        )
        self.flex_host = ctk.CTkFrame(flex_card, fg_color="transparent")
        self.flex_host.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.flex_host.configure(height=340)
        self.flex_host.pack_propagate(False)
        self.flex_chart_insight = self._inline_insight_label(flex_card)


        lower = ctk.CTkFrame(content, fg_color="transparent")
        lower.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        self.section_targets["linienvergleich"] = lower
        lower.grid_columnconfigure((0, 1), weight=1)


        station_card = self._card(lower, row=0, title="Haltestellenfolge", subtitle="Ablauf nach Linie mit Ein- und Ausstiegen", column=0, padx=(0, 8))
        self.station_textbox = ctk.CTkTextbox(
            station_card,
            fg_color=PALETTE["surface_alt"],
            corner_radius=18,
            border_width=0,
            text_color=PALETTE["text"],
            font=ctk.CTkFont(family="Consolas", size=13),
        )
        self.station_textbox.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.station_textbox.configure(height=260)


        insight_card = self._card(lower, row=0, title="Smart Insights", subtitle="Adaptive Lesart für dein Network Design", column=1, padx=(8, 0))
        self.insight_textbox = ctk.CTkTextbox(
            insight_card,
            fg_color=PALETTE["surface_alt"],
            corner_radius=18,
            border_width=0,
            text_color=PALETTE["text"],
            font=ctk.CTkFont(family="Segoe UI", size=13),
        )
        self.insight_textbox.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.insight_textbox.configure(height=260)


        self._build_map_tab(map_tab)
        self._build_prediction_tab(prediction_tab)
        self._build_timetable_tab(timetable_tab)
        self._build_kpi_tab(kpi_tab)


    def _context_toggle(self, parent, column: int, title: str, variable: ctk.BooleanVar) -> None:
        """
        Erzeugt einen Kontextschalter für externe Einflussfaktoren im Dashboard.

        Die Methode baut eine kompakte Checkbox-Karte auf, mit der zusätzliche
        Kontextinformationen wie Vorlesungen, Events, Feiertage oder Ferien in die
        Visualisierung ein- oder ausgeblendet werden können. Jede Änderung löst
        direkt eine Aktualisierung des Dashboards aus.

        Parameter:
            parent: Eltern-Widget, in dem der Schalter platziert wird.
            column (int): Zielspalte im Grid-Layout des Eltern-Widgets.
            title (str): Sichtbare Bezeichnung des Kontextschalters.
            variable (ctk.BooleanVar): Gebundene Zustandsvariable für ein/aus.

        Rückgabewerte:
            None: Die Methode erzeugt und platziert GUI-Elemente.

        Fehler/Sonderfälle:
            Die Methode setzt voraus, dass parent ein grid-basiertes Layout verwendet.
            Die Zustandslogik selbst wird nicht hier, sondern in update_dashboard
            verarbeitet.

        Projektkontext:
            Kontextschalter sind wichtig, um Nachfrageverläufe nicht nur technisch,
            sondern auch fachlich anhand externer Einflussfaktoren interpretieren zu
            können.
        """
        card = ctk.CTkFrame(
            parent,
            fg_color=PALETTE["surface"],
            corner_radius=18,
            border_width=1,
            border_color=PALETTE["border"],
        )
        card.grid(row=0, column=column, sticky="ew", padx=6)
        ctk.CTkCheckBox(
            card,
            text=title,
            variable=variable,
            onvalue=True,
            offvalue=False,
            command=self.update_dashboard,
            text_color=PALETTE["text"],
            fg_color=PALETTE["accent"],
            hover_color=PALETTE["accent_dark"],
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
        ).pack(anchor="w", padx=14, pady=12)


    def _builder_menu(
        self,
        parent: ctk.CTkFrame,
        column: int,
        label: str,
        variable: ctk.StringVar,
        values: list[str],
    ) -> ctk.CTkOptionMenu:
        """
        Erzeugt ein standardisiertes Auswahlmenü für konfigurierbare Analysebausteine.

        Die Methode kapselt das wiederkehrende UI-Muster aus Beschriftung und
        Auswahlfeld, das insbesondere in der flexiblen Analyse verwendet wird.
        Änderungen an der Auswahl lösen unmittelbar eine Dashboard-Aktualisierung aus,
        sodass Interaktionen ohne zusätzlichen Bestätigungsschritt wirksam werden.

        Parameter:
            parent (ctk.CTkFrame): Eltern-Container des Menüs.
            column (int): Zielspalte innerhalb des Grid-Layouts.
            label (str): Beschriftung des Menüs.
            variable (ctk.StringVar): Gebundene String-Variable für die Auswahl.
            values (list[str]): Verfügbare Auswahloptionen.

        Rückgabewerte:
            ctk.CTkOptionMenu: Das erzeugte und bereits platzierte Auswahlmenü.

        Fehler/Sonderfälle:
            Ist die Werteliste fachlich noch nicht befüllt, kann das Menü zunächst
            Platzhalterwerte enthalten. Die eigentliche Interpretation der Auswahl
            erfolgt in nachgelagerter Dashboard-Logik.

        Projektkontext:
            Das Menübaustein-Muster unterstützt die interaktive, explorative Analyse
            des Systems und stellt eine konsistente Bedienlogik über mehrere Ansichten
            hinweg sicher.
        """
        frame = ctk.CTkFrame(
            parent,
            fg_color=PALETTE["surface"],
            corner_radius=18,
            border_width=1,
            border_color=PALETTE["border"],
        )
        frame.grid(row=0, column=column, sticky="ew", padx=8, pady=10)
        ctk.CTkLabel(
            frame,
            text=label,
            text_color=PALETTE["muted"],
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        ).pack(anchor="w", padx=12, pady=(10, 6))
        menu = ctk.CTkOptionMenu(
            frame,
            values=values,
            variable=variable,
            height=40,
            corner_radius=14,
            fg_color=PALETTE["surface_alt"],
            button_color=PALETTE["accent"],
            button_hover_color=PALETTE["accent_dark"],
            dropdown_fg_color=PALETTE["surface"],
            dropdown_text_color=PALETTE["text"],
            dropdown_hover_color=PALETTE["accent_soft"],
            text_color="white",
            command=lambda _choice: self.update_dashboard(),
        )
        menu.pack(fill="x", padx=12, pady=(0, 12))
        return menu


    def _build_prediction_tab(self, parent: ctk.CTkFrame) -> None:
        """
        Baut die Oberfläche für die kurzfristige Nachfrageprognose auf.

        Die Methode erstellt sämtliche Steuer- und Ausgabeelemente der Prediction-
        Ansicht: Auswahl von Tag, Stunde, Linie und Prognosemodus, Definition des
        Vorhersagehorizonts bzw. der Haltestellenanzahl, Aktionsbutton zur
        Berechnung, Kennzahlenkarten zur Modellgüte sowie Bereiche für Chart und
        operative Textausgabe. Die Ansicht dient dazu, kurzfristige Nachfrage-
        prognosen mit operativer Relevanz im Netzkontext nutzbar zu machen.

        Parameter:
            parent (ctk.CTkFrame): Container, in den der Prediction-Tab aufgebaut wird.

        Rückgabewerte:
            None: Die Methode erzeugt und platziert ausschließlich GUI-Komponenten.

        Fehler/Sonderfälle:
            Die Menüs werden initial teilweise mit Platzhalterwerten angelegt und erst
            durch spätere Refresh-Methoden mit konkreten Linien- bzw.
            Haltestellenoptionen befüllt. Vor der ersten Berechnung bleibt die
            Textausgabe in einem initialen Hinweiszustand.

        Projektkontext:
            Die Prediction-Ansicht verknüpft das Vorhersagemodell des Projekts mit
            einer operativen Benutzeroberfläche, damit kurzfristige Nachfrage- und
            Fahrplanentscheidungen direkt interpretiert werden können.
        """
        lab = self._card(
            parent,
            row=0,
            title="Kurzfristige Nachfrageprognose",
            subtitle="Operativer Forecast für die nächsten Stunden mit adaptiver Fahrplan-Empfehlung",
        )
        controls = ctk.CTkFrame(lab, fg_color=PALETTE["surface_alt"], corner_radius=22)
        controls.pack(fill="x", padx=14, pady=(0, 14))
        controls.configure(height=220)
        controls.pack_propagate(False)
        controls.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)
        controls.grid_rowconfigure(0, minsize=94)
        controls.grid_rowconfigure(1, minsize=82)


        day_box = ctk.CTkFrame(controls, fg_color="transparent")
        day_box.grid(row=0, column=0, sticky="ew", padx=12, pady=12)
        ctk.CTkLabel(
            day_box,
            text="Prediction-Tag",
            text_color=PALETTE["muted"],
            width=170,
            anchor="w",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        ).pack(anchor="w", pady=(0, 4))
        initial_day = DEFAULT_PREDICTION_DATE
        self.prediction_day_entry = DateEntry(
            day_box,
            date_pattern="yyyy-mm-dd",
            year=initial_day.year,
            month=initial_day.month,
            day=initial_day.day,
            **dark_date_entry_options(),
        )
        self.prediction_day_entry.pack(fill="x")


        hour_box = ctk.CTkFrame(controls, fg_color="transparent")
        hour_box.grid(row=0, column=1, sticky="ew", padx=12, pady=12)
        self.prediction_hour_label = ctk.CTkLabel(
            hour_box,
            text="08:00 Uhr",
            text_color=PALETTE["text"],
            width=150,
            anchor="w",
            font=ctk.CTkFont(family="Bahnschrift", size=20, weight="bold"),
        )
        self.prediction_hour_label.pack(anchor="w")
        self.prediction_hour_slider = ctk.CTkSlider(
            hour_box,
            from_=0,
            to=23,
            number_of_steps=23,
            progress_color=PALETTE["accent"],
            button_color=PALETTE["accent"],
            button_hover_color=PALETTE["accent_dark"],
            command=self._on_prediction_hour_slider,
        )
        self.prediction_hour_slider.pack(fill="x", pady=(8, 0))
        self.prediction_hour_slider.set(8)


        line_box = ctk.CTkFrame(controls, fg_color="transparent")
        line_box.grid(row=0, column=2, sticky="ew", padx=12, pady=12)
        ctk.CTkLabel(
            line_box,
            text="Linie",
            text_color=PALETTE["muted"],
            width=90,
            anchor="w",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        ).pack(anchor="w", pady=(0, 4))
        self.prediction_line_menu = ctk.CTkOptionMenu(
            line_box,
            variable=self.prediction_line_choice,
            values=["Keine Linie"],
            height=38,
            corner_radius=14,
            fg_color=PALETTE["surface"],
            button_color=PALETTE["accent"],
            button_hover_color=PALETTE["accent_dark"],
            dropdown_fg_color=PALETTE["surface"],
            dropdown_text_color=PALETTE["text"],
            dropdown_hover_color=PALETTE["accent_soft"],
            command=lambda _choice: self._refresh_prediction_station_menu(),
        )
        self.prediction_line_menu.pack(fill="x")


        mode_box = ctk.CTkFrame(controls, fg_color="transparent")
        mode_box.grid(row=0, column=3, sticky="ew", padx=12, pady=12)
        ctk.CTkLabel(
            mode_box,
            text="Modus",
            text_color=PALETTE["muted"],
            width=100,
            anchor="w",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        ).pack(anchor="w", pady=(0, 4))
        self.prediction_mode_menu = ctk.CTkOptionMenu(
            mode_box,
            variable=self.prediction_mode_choice,
            values=["Nächste Stunden", "Nächste Haltestellen"],
            height=38,
            corner_radius=14,
            fg_color=PALETTE["surface"],
            button_color=PALETTE["accent"],
            button_hover_color=PALETTE["accent_dark"],
            dropdown_fg_color=PALETTE["surface"],
            dropdown_text_color=PALETTE["text"],
            dropdown_hover_color=PALETTE["accent_soft"],
            command=lambda _choice: self._sync_prediction_mode_controls(),
        )
        self.prediction_mode_menu.pack(fill="x")


        horizon_box = ctk.CTkFrame(controls, fg_color="transparent")
        self.prediction_horizon_box = horizon_box
        horizon_box.grid(row=0, column=4, sticky="ew", padx=12, pady=12)
        ctk.CTkLabel(
            horizon_box,
            text="Horizont",
            text_color=PALETTE["muted"],
            width=100,
            anchor="w",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        ).pack(anchor="w", pady=(0, 4))
        self.prediction_horizon_menu = ctk.CTkOptionMenu(
            horizon_box,
            variable=self.prediction_horizon_choice,
            values=["2 Stunden", "4 Stunden", "6 Stunden", "8 Stunden"],
            height=38,
            corner_radius=14,
            fg_color=PALETTE["surface"],
            button_color=PALETTE["accent"],
            button_hover_color=PALETTE["accent_dark"],
            dropdown_fg_color=PALETTE["surface"],
            dropdown_text_color=PALETTE["text"],
            dropdown_hover_color=PALETTE["accent_soft"],
        )
        self.prediction_horizon_menu.pack(fill="x")


        station_box = ctk.CTkFrame(controls, fg_color="transparent")
        self.prediction_station_box = station_box
        station_box.grid(row=1, column=3, sticky="ew", padx=12, pady=(0, 12))
        ctk.CTkLabel(
            station_box,
            text="Start-Haltestelle",
            text_color=PALETTE["muted"],
            width=120,
            anchor="w",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        ).pack(anchor="w", pady=(0, 4))
        self.prediction_station_menu = ctk.CTkOptionMenu(
            station_box,
            variable=self.prediction_station_choice,
            values=["Alle Haltestellen"],
            height=38,
            corner_radius=14,
            fg_color=PALETTE["surface"],
            button_color=PALETTE["accent"],
            button_hover_color=PALETTE["accent_dark"],
            dropdown_fg_color=PALETTE["surface"],
            dropdown_text_color=PALETTE["text"],
            dropdown_hover_color=PALETTE["accent_soft"],
        )
        self.prediction_station_menu.pack(fill="x")


        stop_count_box = ctk.CTkFrame(controls, fg_color="transparent")
        self.prediction_stop_count_box = stop_count_box
        stop_count_box.grid(row=1, column=4, sticky="ew", padx=12, pady=(0, 12))
        ctk.CTkLabel(
            stop_count_box,
            text="Nächste",
            text_color=PALETTE["muted"],
            width=100,
            anchor="w",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        ).pack(anchor="w", pady=(0, 4))
        self.prediction_stop_count_menu = ctk.CTkOptionMenu(
            stop_count_box,
            variable=self.prediction_stop_count_choice,
            values=["3 Haltestellen", "5 Haltestellen", "8 Haltestellen", "10 Haltestellen"],
            height=38,
            corner_radius=14,
            fg_color=PALETTE["surface"],
            button_color=PALETTE["accent"],
            button_hover_color=PALETTE["accent_dark"],
            dropdown_fg_color=PALETTE["surface"],
            dropdown_text_color=PALETTE["text"],
            dropdown_hover_color=PALETTE["accent_soft"],
        )
        self.prediction_stop_count_menu.pack(fill="x")


        action_box = ctk.CTkFrame(controls, fg_color="transparent")
        action_box.grid(row=1, column=0, columnspan=3, sticky="ew", padx=12, pady=(0, 12))
        action_box.configure(height=52)
        action_box.grid_propagate(False)
        action_box.grid_columnconfigure(0, weight=0, minsize=220)
        action_box.grid_columnconfigure(1, weight=1)
        self.prediction_train_button = None
        self.prediction_predict_button = ctk.CTkButton(
            action_box,
            text="Forecast berechnen",
            width=220,
            height=48,
            corner_radius=14,
            fg_color=PALETTE["accent"],
            hover_color=PALETTE["accent_dark"],
            command=self._run_prediction,
        )
        self.prediction_predict_button.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.prediction_status_label = ctk.CTkLabel(
            action_box,
            text="Bereit für kurzfristige operative Prognosen.",
            fg_color=PALETTE["surface"],
            corner_radius=14,
            text_color=PALETTE["muted"],
            height=48,
            anchor="w",
            justify="left",
            font=ctk.CTkFont(family="Segoe UI", size=12),
        )
        self.prediction_status_label.grid(row=0, column=1, sticky="ew", ipadx=14, padx=(0, 0))


        metric_row = ctk.CTkFrame(lab, fg_color="transparent")
        metric_row.pack(fill="x", padx=14, pady=(0, 12))
        metric_row.grid_columnconfigure((0, 1, 2, 3), weight=1)
        for index, title in enumerate(["RF MAE", "Graph MAE", "Forecast MAE", "Gewichtung"]):
            card = ctk.CTkFrame(
                metric_row,
                fg_color=PALETTE["surface"],
                corner_radius=20,
                border_width=1,
                border_color=PALETTE["border"],
            )
            card.grid(row=0, column=index, sticky="ew", padx=6)
            ctk.CTkLabel(
                card,
                text=title,
                text_color=PALETTE["muted"],
                width=150,
                anchor="w",
                font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            ).pack(anchor="w", padx=14, pady=(12, 4))
            value = ctk.CTkLabel(
                card,
                text="-",
                text_color=PALETTE["text"],
                font=ctk.CTkFont(family="Bahnschrift", size=20, weight="bold"),
            )
            value.pack(anchor="w", padx=14, pady=(0, 12))
            self.prediction_metric_labels[title] = value


        body = ctk.CTkFrame(parent, fg_color="transparent")
        body.grid(row=1, column=0, sticky="ew")
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=1)


        chart_card = self._card(body, row=0, title="Kurzfrist-Forecast", subtitle="Erwarteter Demand je Stunde oder Linienabschnitt", column=0, padx=(0, 8))
        self.prediction_chart_host = ctk.CTkFrame(chart_card, fg_color="transparent")
        self.prediction_chart_host.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.prediction_chart_host.configure(height=420)
        self.prediction_chart_host.pack_propagate(False)


        ranking_card = self._card(body, row=0, title="Operative Ausgabe", subtitle="Adaptiver Fahrplan oder Demand der nächsten Haltestellen", column=1, padx=(8, 0))
        self.prediction_textbox = ctk.CTkTextbox(
            ranking_card,
            fg_color=PALETTE["surface_alt"],
            corner_radius=18,
            border_width=0,
            text_color=PALETTE["text"],
            font=ctk.CTkFont(family="Consolas", size=12),
        )
        self.prediction_textbox.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.prediction_textbox.insert("1.0", "Noch keine Kurzfrist-Prognose berechnet.\n\nWähle Linie, Modus und Startstunde.")
        self.prediction_textbox.configure(state="disabled")
        self._refresh_prediction_line_menu()
        self._sync_prediction_mode_controls()

    def _build_timetable_tab(self, parent: ctk.CTkFrame) -> None:
    """
    Baut die Oberfläche für den Fahrplanvergleich zwischen Bestands- und
    adaptivem Prognosefahrplan auf.

    Die Methode erstellt die komplette GUI-Struktur des Fahrplan-Tabs. Dazu
    gehören Steuerfelder für Linie und Tag, der Auslösemechanismus für die
    ganztägige Vergleichsberechnung, eine Statusanzeige, eine grafische
    Routendarstellung sowie zwei Textbereiche für den direkten Vergleich von
    WVV-Fahrplan und adaptiv erzeugtem Zielbild. Die Ansicht ist darauf
    ausgerichtet, operative Unterschiede zwischen aktuellem Angebot und
    nachfrageorientiertem Vorschlag sichtbar zu machen.

    Parameter:
        parent (ctk.CTkFrame): Container, in den die Fahrplanansicht eingebettet
            wird.

    Rückgabewerte:
        None: Die Methode erzeugt und platziert ausschließlich GUI-Komponenten.

    Fehler/Sonderfälle:
        Linienauswahl und Textausgaben werden zunächst mit Platzhalterzuständen
        initialisiert. Die tatsächlichen Vergleichsdaten werden erst durch
        nachgelagerte Berechnungsfunktionen geladen.

    Projektkontext:
        Der Fahrplanvergleich ist eine zentrale Brücke zwischen Prediction und
        operativer Angebotsplanung, da hier sichtbar wird, wie aus prognostizierter
        Nachfrage ein alternativer Fahrplanvorschlag abgeleitet werden kann.
    """
    header = self._card(
        parent,
        row=0,
        title="Fahrplanvergleich",
        subtitle="WVV-Fahrplan gegen adaptiven Prognose-Fahrplan",
    )
    ctk.CTkLabel(
        header,
        text="Oben die Route, unten der direkte Vergleich: WVV aktuell vs. adaptiver Vorschlag aus dem Demand-Modell.",
        text_color=PALETTE["muted"],
        anchor="w",
        justify="left",
        font=ctk.CTkFont(family="Segoe UI", size=13),
    ).pack(fill="x", padx=16, pady=(0, 12))


    controls = ctk.CTkFrame(header, fg_color=PALETTE["surface_alt"], corner_radius=22)
    controls.pack(fill="x", padx=14, pady=(0, 14))
    controls.grid_columnconfigure((0, 1, 2), weight=1)
    controls.grid_columnconfigure(3, weight=0)


    line_box = ctk.CTkFrame(controls, fg_color="transparent")
    line_box.grid(row=0, column=0, sticky="ew", padx=12, pady=12)
    ctk.CTkLabel(line_box, text="Linie", text_color=PALETTE["muted"], anchor="w", font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold")).pack(anchor="w", pady=(0, 4))
    self.timetable_line_menu = ctk.CTkOptionMenu(
        line_box,
        variable=self.timetable_line_choice,
        values=["Keine Fahrplandaten"],
        height=38,
        corner_radius=14,
        fg_color=PALETTE["surface"],
        button_color=PALETTE["accent"],
        button_hover_color=PALETTE["accent_dark"],
        dropdown_fg_color=PALETTE["surface"],
        dropdown_text_color=PALETTE["text"],
        dropdown_hover_color=PALETTE["accent_soft"],
        command=lambda _choice: self._calculate_timetable_comparison(),
    )
    self.timetable_line_menu.pack(fill="x")


    day_box = ctk.CTkFrame(controls, fg_color="transparent")
    day_box.grid(row=0, column=1, sticky="ew", padx=12, pady=12)
    ctk.CTkLabel(day_box, text="Tag", text_color=PALETTE["muted"], anchor="w", font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold")).pack(anchor="w", pady=(0, 4))
    self.timetable_day_entry = DateEntry(
        day_box,
        date_pattern="yyyy-mm-dd",
        year=DEFAULT_PREDICTION_DATE.year,
        month=DEFAULT_PREDICTION_DATE.month,
        day=DEFAULT_PREDICTION_DATE.day,
        **dark_date_entry_options(),
    )
    self.timetable_day_entry.pack(fill="x")
    self.timetable_day_entry.bind("<<DateEntrySelected>>", lambda _event: self._calculate_timetable_comparison(), add="+")


    day_scope_box = ctk.CTkFrame(controls, fg_color=PALETTE["surface"], corner_radius=16, border_width=1, border_color=PALETTE["border"])
    day_scope_box.grid(row=0, column=2, sticky="ew", padx=12, pady=12)
    ctk.CTkLabel(
        day_scope_box,
        text="Berechnung",
        text_color=PALETTE["muted"],
        anchor="w",
        font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
    ).pack(anchor="w", padx=14, pady=(10, 2))
    ctk.CTkLabel(
        day_scope_box,
        text="Ganzer Betriebstag",
        text_color=PALETTE["text"],
        anchor="w",
        font=ctk.CTkFont(family="Bahnschrift", size=18, weight="bold"),
    ).pack(anchor="w", padx=14, pady=(0, 10))


    self.timetable_calculate_button = ctk.CTkButton(
        controls,
        text="Ganztag berechnen",
        width=190,
        height=46,
        corner_radius=16,
        fg_color=PALETTE["teal"],
        hover_color="#0f766e",
        command=self._calculate_timetable_comparison,
    )
    self.timetable_calculate_button.grid(row=0, column=3, sticky="ew", padx=12, pady=12)


    self.timetable_status_label = ctk.CTkLabel(
        header,
        text="Fahrplandaten werden geladen...",
        fg_color=PALETTE["surface_alt"],
        corner_radius=16,
        text_color=PALETTE["muted"],
        anchor="w",
        justify="left",
        font=ctk.CTkFont(family="Segoe UI", size=13),
    )
    self.timetable_status_label.pack(fill="x", padx=14, pady=(0, 14), ipady=9)


    route_shell = ctk.CTkFrame(header, fg_color=PALETTE["surface_alt"], corner_radius=22)
    route_shell.pack(fill="x", padx=14, pady=(0, 14))
    self.timetable_route_canvas = tk.Canvas(route_shell, height=360, bg=PALETTE["surface_alt"], highlightthickness=0, bd=0)
    self.timetable_route_canvas.pack(fill="x", expand=True, padx=12, pady=12)


    tables = ctk.CTkFrame(parent, fg_color="transparent")
    tables.grid(row=1, column=0, sticky="ew")
    tables.grid_columnconfigure(0, weight=1)


    wvv_card = self._card(tables, row=0, title="WVV Fahrplan", subtitle="Bereinigter aktueller Fahrplan aus den PDF/CSV-Daten", column=0)
    self.timetable_wvv_textbox = ctk.CTkTextbox(
        wvv_card,
        fg_color=PALETTE["surface_alt"],
        corner_radius=18,
        border_width=0,
        text_color=PALETTE["text"],
        wrap="none",
        font=ctk.CTkFont(family="Consolas", size=13),
    )
    self.timetable_wvv_textbox.pack(fill="both", expand=True, padx=12, pady=(0, 12))
    self.timetable_wvv_textbox.configure(height=520)


    adaptive_card = self._card(tables, row=1, title="Adaptiver Fahrplan", subtitle="Aus Prognose-Fahrtenzahl abgeleiteter Vergleichsplan", column=0)
    self.timetable_adaptive_textbox = ctk.CTkTextbox(
        adaptive_card,
        fg_color=PALETTE["surface_alt"],
        corner_radius=18,
        border_width=0,
        text_color=PALETTE["text"],
        wrap="none",
        font=ctk.CTkFont(family="Consolas", size=13),
    )
    self.timetable_adaptive_textbox.pack(fill="both", expand=True, padx=12, pady=(0, 12))
    self.timetable_adaptive_textbox.configure(height=520)


    self._refresh_timetable_line_menu()

    def _build_kpi_tab(self, parent: ctk.CTkFrame) -> None:
        """
        Baut die Oberfläche für den KPI-Vergleich zwischen Bestands- und adaptivem
        Fahrplan auf.

        Die Methode erstellt alle Steuerelemente und Ausgabebereiche für die KPI-
        Analyse. Dazu zählen Datums- und Linienauswahl, Konfiguration der Kosten pro
        Busstunde, Start der Linienberechnung, Statusanzeige, Textboxen für Gesamt-
        und Linien-KPIs sowie ein dreispaltiges Ranking für Einsparpotenzial,
        Überlastung und Unterauslastung. Zusätzlich wird eine initiale Gesamt-
        berechnung mit Jahresstandarddaten verzögert angestoßen.

        Parameter:
            parent (ctk.CTkFrame): Container, in den die KPI-Ansicht eingebettet wird.

        Rückgabewerte:
            None: Die Methode erzeugt und positioniert GUI-Komponenten.

        Fehler/Sonderfälle:
            Textboxen und Rankings werden zunächst mit Hinweisen vorbelegt. Fehlen
            Fahrplandaten oder Linienoptionen, bleiben Auswahlmenüs auf
            Platzhalterwerten. Die eigentliche KPI-Berechnung erfolgt erst in
            separaten Folgemethoden.

        Projektkontext:
            Die KPI-Ansicht operationalisiert die fachliche Bewertung des adaptiven
            Fahrplankonzepts, indem Nachfrageabdeckung, Kostenwirkung sowie mögliche
            Über- und Unterauslastungen vergleichbar gemacht werden.
        """
        header = self._card(
            parent,
            row=0,
            title="KPI Vergleich",
            subtitle="WVV-Bestandsfahrplan gegen adaptiven Fahrplan mit Demand- und Kostenmetriken",
        )
        controls = ctk.CTkFrame(header, fg_color=PALETTE["surface_alt"], corner_radius=22)
        controls.pack(fill="x", padx=14, pady=(0, 14))
        controls.grid_columnconfigure((0, 1, 2), weight=1)
        controls.grid_columnconfigure(3, weight=0)


        day_box = ctk.CTkFrame(controls, fg_color="transparent")
        day_box.grid(row=0, column=0, sticky="ew", padx=12, pady=12)
        ctk.CTkLabel(
            day_box,
            text="Linien-KPI-Tag",
            text_color=PALETTE["muted"],
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        ).pack(anchor="w", pady=(0, 4))
        self.kpi_day_entry = DateEntry(
            day_box,
            date_pattern="yyyy-mm-dd",
            year=DEFAULT_PREDICTION_DATE.year,
            month=DEFAULT_PREDICTION_DATE.month,
            day=DEFAULT_PREDICTION_DATE.day,
            **dark_date_entry_options(),
        )
        self.kpi_day_entry.pack(fill="x")


        line_box = ctk.CTkFrame(controls, fg_color="transparent")
        line_box.grid(row=0, column=1, sticky="ew", padx=12, pady=12)
        ctk.CTkLabel(
            line_box,
            text="Linie für Detailvergleich",
            text_color=PALETTE["muted"],
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        ).pack(anchor="w", pady=(0, 4))
        self.kpi_line_menu = ctk.CTkOptionMenu(
            line_box,
            variable=self.kpi_line_choice,
            values=["Keine Linie"],
            height=38,
            corner_radius=14,
            fg_color=PALETTE["surface"],
            button_color=PALETTE["accent"],
            button_hover_color=PALETTE["accent_dark"],
            dropdown_fg_color=PALETTE["surface"],
            dropdown_text_color=PALETTE["text"],
            dropdown_hover_color=PALETTE["accent_soft"],
        )
        self.kpi_line_menu.pack(fill="x")


        cost_box = ctk.CTkFrame(controls, fg_color=PALETTE["surface"], corner_radius=16, border_width=1, border_color=PALETTE["border"])
        cost_box.grid(row=0, column=2, sticky="ew", padx=12, pady=12)
        ctk.CTkLabel(
            cost_box,
            text="Kosten pro Busstunde",
            text_color=PALETTE["muted"],
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        ).pack(anchor="w", padx=14, pady=(10, 6))
        cost_row = ctk.CTkFrame(cost_box, fg_color="transparent")
        cost_row.pack(fill="x", padx=14, pady=(0, 10))
        cost_row.grid_columnconfigure(0, weight=1)
        self.kpi_cost_entry = ctk.CTkEntry(
            cost_row,
            textvariable=self.bus_hourly_cost,
            height=36,
            corner_radius=12,
            fg_color=PALETTE["surface_alt"],
            border_color=PALETTE["border"],
            text_color=PALETTE["text"],
            font=ctk.CTkFont(family="Bahnschrift", size=16, weight="bold"),
        )
        self.kpi_cost_entry.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(
            cost_row,
            text="Speichern",
            width=92,
            height=36,
            corner_radius=12,
            fg_color=PALETTE["accent_soft"],
            hover_color=PALETTE["accent"],
            command=self._save_cost_settings,
        ).grid(row=0, column=1, sticky="ew", padx=(8, 0))


        ctk.CTkButton(
            controls,
            text="Linie berechnen",
            width=160,
            height=46,
            corner_radius=16,
            fg_color=PALETTE["teal"],
            hover_color="#0f766e",
            command=lambda: self._calculate_kpi_comparison(scope="line"),
        ).grid(row=0, column=3, sticky="ew", padx=12, pady=12)


        self.kpi_status_label = ctk.CTkLabel(
            header,
            text="Gesamt-KPIs laden automatisch als Jahresstandard 2025 über alle Linien. Linien-KPIs nutzt du für den gewählten Tag.",
            fg_color=PALETTE["surface_alt"],
            corner_radius=16,
            text_color=PALETTE["muted"],
            anchor="w",
            justify="left",
            font=ctk.CTkFont(family="Segoe UI", size=13),
        )
        self.kpi_status_label.pack(fill="x", padx=14, pady=(0, 14), ipady=9)


        body = ctk.CTkFrame(parent, fg_color="transparent")
        body.grid(row=1, column=0, sticky="ew")
        body.grid_columnconfigure((0, 1), weight=1)


        total_card = self._card(body, row=0, title="Gesamtfahrplan KPIs", subtitle="Jahresbenchmark 2025 über alle Linien", column=0, padx=(0, 8))
        self.kpi_total_textbox = self._kpi_textbox(total_card)
        self.kpi_total_textbox.insert("1.0", "Noch keine Jahres-Gesamt-KPIs berechnet.")
        self.kpi_total_textbox.configure(state="disabled")


        line_card = self._card(body, row=0, title="Linien-KPIs", subtitle="Detailvergleich für die ausgewählte Linie", column=1, padx=(8, 0))
        self.kpi_line_textbox = self._kpi_textbox(line_card)
        self.kpi_line_textbox.insert("1.0", "Noch keine Linien-KPIs berechnet.")
        self.kpi_line_textbox.configure(state="disabled")


        ranking_card = self._card(
            parent,
            row=2,
            title="Linienranking",
            subtitle="Top 10 nebeneinander: Einsparpotenzial, Überlastung und Unterauslastung",
        )
        ranking_grid = ctk.CTkFrame(ranking_card, fg_color="transparent")
        ranking_grid.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        ranking_grid.grid_columnconfigure((0, 1, 2), weight=1, uniform="kpi_ranking")
        ranking_grid.grid_rowconfigure(0, weight=1)
        self.kpi_ranking_boxes = {
            "savings": self._ranking_column(
                ranking_grid,
                column=0,
                title="Einsparpotenzial",
                subtitle="Welche Linien sparen am meisten Kosten?",
            ),
            "overload": self._ranking_column(
                ranking_grid,
                column=1,
                title="Überlastung",
                subtitle="Wo ist der WVV-Fahrplan rechnerisch zu knapp?",
            ),
            "underload": self._ranking_column(
                ranking_grid,
                column=2,
                title="Unterauslastung",
                subtitle="Wo entstehen die meisten leeren Kapazitäten?",
            ),
        }
        self._refresh_kpi_line_menu()
        self.after(450, lambda: self._calculate_kpi_comparison(scope="total"))


    def _kpi_textbox(self, parent: ctk.CTkFrame) -> ctk.CTkTextbox:
        """
        Erzeugt eine standardisierte Textbox für KPI-Berichte.

        Die Methode kapselt die wiederkehrende Gestaltung der Textbereiche, in denen
        KPI-Ausgaben formatiert dargestellt werden. Dadurch bleiben Darstellung,
        Höhe und typografische Anmutung in Gesamt- und Linienberichten konsistent.

        Parameter:
            parent (ctk.CTkFrame): Container, in den die Textbox eingefügt wird.

        Rückgabewerte:
            ctk.CTkTextbox: Konfigurierte Textbox für KPI-Ausgaben.

        Fehler/Sonderfälle:
            Die Methode setzt keine Inhalte und keinen Zustand voraus; beides kann
            nachgelagert gesetzt oder gesperrt werden.

        Projektkontext:
            Einheitlich formatierte Berichtsfelder sind wichtig, damit KPI-Ergebnisse
            in verschiedenen Vergleichsebenen konsistent gelesen und interpretiert
            werden können.
        """
        textbox = ctk.CTkTextbox(
            parent,
            fg_color=PALETTE["surface_alt"],
            corner_radius=18,
            border_width=0,
            text_color=PALETTE["text"],
            wrap="none",
            font=ctk.CTkFont(family="Consolas", size=13),
        )
        textbox.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        textbox.configure(height=360)
        return textbox


    def _ranking_column(
        self,
        parent: ctk.CTkFrame,
        column: int,
        title: str,
        subtitle: str,
    ) -> ctk.CTkTextbox:
        """
        Erstellt eine Ranking-Spalte für eine bestimmte KPI-Perspektive.

        Die Methode baut eine vollständige Spalte mit Titel, erläuterndem Untertitel
        und Textbox für Rankinginhalte auf. Sie wird verwendet, um verschiedene
        Bewertungsdimensionen wie Einsparpotenzial, Überlastung und Unterauslastung
        parallel darzustellen.

        Parameter:
            parent (ctk.CTkFrame): Eltern-Container des Rankingrasters.
            column (int): Zielspalte innerhalb des Grid-Layouts.
            title (str): Überschrift der Ranking-Spalte.
            subtitle (str): Erläuterung der fachlichen Bewertungsdimension.

        Rückgabewerte:
            ctk.CTkTextbox: Textbox, in die spätere Rankingausgaben geschrieben werden.

        Fehler/Sonderfälle:
            Die Spalte wird zunächst mit einem Platzhaltertext vorbelegt. Die
            tatsächliche Rankingberechnung erfolgt erst in nachgelagerter KPI-Logik.

        Projektkontext:
            Das nebeneinanderliegende Ranking macht verschiedene Optimierungs- und
            Problemdimensionen des Fahrplannetzes vergleichbar und unterstützt die
            Priorisierung von Maßnahmen.
        """
        shell = ctk.CTkFrame(
            parent,
            fg_color=PALETTE["surface_alt"],
            corner_radius=20,
            border_width=1,
            border_color=PALETTE["border"],
        )
        shell.grid(
            row=0,
            column=column,
            sticky="nsew",
            padx=(0 if column == 0 else 6, 0 if column == 2 else 6),
            pady=0,
        )
        ctk.CTkLabel(
            shell,
            text=title,
            text_color=PALETTE["text"],
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=14, pady=(12, 2))
        ctk.CTkLabel(
            shell,
            text=subtitle,
            text_color=PALETTE["muted"],
            font=ctk.CTkFont(family="Segoe UI", size=12),
            anchor="w",
            justify="left",
        ).pack(fill="x", padx=14, pady=(0, 8))
        textbox = ctk.CTkTextbox(
            shell,
            fg_color=PALETTE["surface"],
            corner_radius=16,
            border_width=0,
            text_color=PALETTE["text"],
            wrap="none",
            font=ctk.CTkFont(family="Consolas", size=12),
        )
        textbox.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        textbox.configure(height=340)
        textbox.insert("1.0", "Wird automatisch geladen.")
        textbox.configure(state="disabled")
        return textbox


    def _refresh_timetable_line_menu(self) -> None:
        """
        Aktualisiert die Linienauswahl für den Fahrplanvergleich.

        Die Methode liest die verfügbaren Fahrplanlinien aus dem Fahrplanrepository,
        erzeugt darstellungsgeeignete Labels und baut daraus eine Lookup-Struktur für
        die spätere Auswahlverarbeitung auf. Falls die aktuell gesetzte Auswahl nicht
        mehr gültig ist, wird eine bevorzugte Standardlinie oder ein Fallbackwert
        gesetzt.

        

        Rückgabewerte:
            None: Die Methode aktualisiert ausschließlich internen Zustand und GUI.

        Fehler/Sonderfälle:
            Wenn keine Fahrplandaten verfügbar sind, wird ein Platzhalterwert gesetzt.
            Fehlende Menü-Widgets werden toleriert, solange die Methode vor deren
            endgültiger Initialisierung aufgerufen wird.

        Projektkontext:
            Die Methode stellt sicher, dass der Fahrplanvergleich nur mit tatsächlich
            verfügbaren Linien arbeitet und die Bedienoberfläche konsistent zur
            Datenlage bleibt.
        """
        self.timetable_line_lookup = {}
        lines = self.timetable_repo.available_lines()
        values: list[str] = []
        for line in lines:
            label = self.repo.label_for_line(line)
            values.append(label)
            self.timetable_line_lookup[label] = line
        if not values:
            values = ["Keine Fahrplandaten"]
        if self.timetable_line_menu is not None:
            self.timetable_line_menu.configure(values=values)
        if self.timetable_line_choice.get() not in self.timetable_line_lookup:
            preferred = next((self.repo.label_for_line(line) for line in [10, 20, 27] if line in lines), values[0])
            self.timetable_line_choice.set(preferred)


    def _selected_timetable_line(self) -> int | None:
        """
        Liefert die aktuell ausgewählte Linie im Fahrplanvergleich als numerische ID.

        Die Methode übersetzt das im Auswahlmenü sichtbare Label über die interne
        Lookup-Struktur zurück in die fachlich verwendete Liniennummer.

        

        Rückgabewerte:
            int | None: Gewählte Liniennummer oder None, falls keine gültige Auswahl
            vorliegt.

        Fehler/Sonderfälle:
            Ungültige oder nicht mehr vorhandene Labelwerte führen zu None statt zu
            einem Fehler.

        Projektkontext:
            Die Methode bildet die Schnittstelle zwischen GUI-Auswahl und fachlicher
            Berechnungslogik des Fahrplanvergleichs.
        """
        return self.timetable_line_lookup.get(self.timetable_line_choice.get())


    def _format_cost_input(self, value: float) -> str:
        """
        Formatiert einen Kostenwert für die Anzeige im deutschsprachigen GUI-Feld.

        Der numerische Wert wird auf zwei Nachkommastellen gebracht und mit Komma als
        Dezimaltrennzeichen ausgegeben, damit die Eingabe- und Anzeigeform der
        Benutzeroberfläche dem lokalen Zahlenformat entspricht.

        Parameter:
            value (float): Zu formatierender Kostenwert.

        Rückgabewerte:
            str: Formatierter String im deutschen Dezimalformat.

        Fehler/Sonderfälle:
            Der übergebene Wert wird in float umgewandelt; ungewöhnliche numerische
            Eingaben werden dabei in das Standardformat überführt.

        Projektkontext:
            Eine konsistente Kostenformatierung ist wichtig, da Busstundenkosten als
            zentrale betriebswirtschaftliche Annahme in KPI- und Fahrplanbewertungen
            einfließen.
        """
        return f"{float(value):.2f}".replace(".", ",")


    def _current_bus_hourly_cost(self) -> float:
        """
        Liest und normalisiert den aktuell eingetragenen Wert für Busstundenkosten.

        Die Methode bereinigt Benutzereingaben um Währungssymbole und unterschiedliche
        Dezimaltrennzeichen, wandelt den Text in eine numerische Form um und setzt
        bei ungültiger Eingabe einen fachlich definierten Standardwert. Negative
        Kosten werden auf 0 begrenzt.

        

        Rückgabewerte:
            float: Verwendbarer Kostenwert pro Busstunde.

        Fehler/Sonderfälle:
            Ungültige Texteingaben führen nicht zu einem Abbruch, sondern zu einem
            Rückfall auf DEFAULT_BUS_HOURLY_COST_EUR. Werte kleiner 0 werden auf 0.0
            gekappt.

        Projektkontext:
            Diese Methode stellt sicher, dass KPI- und adaptive Fahrplanberechnungen
            stets mit einem robust interpretierbaren Kostenwert arbeiten können.
        """
        raw = self.bus_hourly_cost.get().strip().replace("€", "").replace("EUR", "").replace("eur", "")
        raw = raw.replace(".", "").replace(",", ".") if "," in raw else raw
        try:
            value = float(raw)
        except ValueError:
            value = DEFAULT_BUS_HOURLY_COST_EUR
        return max(0.0, value)


    def _save_cost_settings(self) -> None:
        """
        Speichert die aktuellen Busstundenkosten und invalidiert abhängige KPI-Caches.

        Die Methode übernimmt den aktuell im GUI-Feld stehenden Kostenwert,
        normalisiert ihn, persistiert ihn in der Konfigurationsdatei und leert
        vorhandene KPI-Caches, damit spätere Berechnungen mit den neuen Kosten
        durchgeführt werden. Zusätzlich werden Statusmeldungen aktualisiert und bei
        vorhandener Gesamtansicht die KPI-Berechnung neu angestoßen.

        

        Rückgabewerte:
            None: Die Methode verändert Anwendungszustand, Cache und Persistenz.

        Fehler/Sonderfälle:
            Auch bei ungünstig formatierter Eingabe wird ein stabiler Kostenwert über
            _current_bus_hourly_cost ermittelt. Ist keine KPI-Gesamttextbox vorhanden,
            wird lediglich der Status aktualisiert.

        Projektkontext:
            Die Kostenannahme ist eine zentrale Steuergröße für Wirtschaftlichkeits-
            und Vergleichsanalysen; Änderungen müssen daher systemweit konsistent
            wirksam werden.
        """
        value = self._current_bus_hourly_cost()
        self.bus_hourly_cost.set(self._format_cost_input(value))
        save_cost_setting(COST_SETTINGS_PATH, value)
        self.kpi_annual_cache.clear()
        self.kpi_ranking_cache.clear()
        message = f"Gespeichert: {value:.2f} EUR pro Busstunde. Neue Fahrplan- und KPI-Berechnungen nutzen diesen Wert."
        self.status_text.set(message)
        if self.kpi_status_label is not None:
            self.kpi_status_label.configure(text=message)
        if self.kpi_total_textbox is not None:
            self._calculate_kpi_comparison(scope="total")


    def _refresh_kpi_line_menu(self) -> None:
        """
        Aktualisiert die Linienauswahl für den KPI-Detailvergleich.

        Die Methode liest alle im Fahrplanrepository verfügbaren Linien ein, baut ein
        Label-zu-Linie-Mapping auf und aktualisiert das zugehörige Auswahlmenü.
        Falls die aktuelle Auswahl ungültig geworden ist, wird eine bevorzugte
        Standardlinie oder ein Platzhalterwert gesetzt.

        

        Rückgabewerte:
            None: Die Methode aktualisiert GUI-Elemente und internen Zustand.

        Fehler/Sonderfälle:
            Ohne verfügbare Fahrplandaten wird ein Platzhalterwert verwendet.
            Nicht initialisierte Menü-Widgets werden toleriert.

        Projektkontext:
            Diese Methode stellt sicher, dass KPI-Detailberechnungen nur auf
            tatsächlich auswählbaren Linien basieren.
        """
        self.kpi_line_lookup = {}
        lines = self.timetable_repo.available_lines()
        values: list[str] = []
        for line in lines:
            label = self.repo.label_for_line(line)
            values.append(label)
            self.kpi_line_lookup[label] = line
        if not values:
            values = ["Keine Fahrplandaten"]
        if self.kpi_line_menu is not None:
            self.kpi_line_menu.configure(values=values)
        if self.kpi_line_choice.get() not in self.kpi_line_lookup:
            preferred = next((self.repo.label_for_line(line) for line in [10, 20, 27] if line in lines), values[0])
            self.kpi_line_choice.set(preferred)


    def _selected_kpi_line(self) -> int | None:
        """
        Liefert die aktuell für den KPI-Detailvergleich ausgewählte Linie.

        Die Methode übersetzt den sichtbaren Auswahlwert des KPI-Menüs über die
        interne Lookup-Struktur in eine numerische Linienkennung.

        

        Rückgabewerte:
            int | None: Gewählte Liniennummer oder None bei ungültiger Auswahl.

        Fehler/Sonderfälle:
            Ungültige Menüwerte führen zu None statt zu einer Ausnahme.

        Projektkontext:
            Die Methode verbindet die GUI-Auswahl mit der fachlichen KPI-Berechnung
            pro Linie.
        """
        return self.kpi_line_lookup.get(self.kpi_line_choice.get())


    def _set_kpi_busy(self, busy: bool, message: str = "") -> None:
        """
        Setzt den Busy-Zustand der KPI-Ansicht und sperrt bzw. entsperrt Eingaben.

        Während laufender Hintergrundberechnungen werden relevante GUI-Elemente in
        einen deaktivierten Zustand versetzt, um konkurrierende Eingaben und
        inkonsistente Zustände zu vermeiden. Optional wird gleichzeitig eine
        Statusmeldung angezeigt.

        Parameter:
            busy (bool): Gibt an, ob die KPI-Ansicht als beschäftigt markiert werden
                soll.
            message (str): Optionale Statusmeldung für die Benutzeroberfläche.

        Rückgabewerte:
            None: Die Methode verändert ausschließlich GUI-Zustand.

        Fehler/Sonderfälle:
            Widgets können je nach Typ oder Initialisierungsstand unterschiedliche
            configure-Verhalten aufweisen; TclError wird deshalb tolerant behandelt.

        Projektkontext:
            Der Busy-Zustand ist wichtig, weil KPI-Berechnungen Hintergrundarbeit und
            potenziell längere Laufzeiten umfassen und die Oberfläche dennoch stabil
            bedienbar bleiben soll.
        """
        self.kpi_busy = busy
        if self.kpi_status_label is not None and message:
            self.kpi_status_label.configure(text=message)
        state = "disabled" if busy else "normal"
        for widget in (self.kpi_line_menu, self.kpi_day_entry, self.kpi_cost_entry):
            if widget is None:
                continue
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass


    def _calculate_kpi_comparison(self, scope: str = "line") -> None:
        """
        Startet die KPI-Berechnung für eine Linie oder für den Gesamtfahrplan.

        Die Methode validiert zunächst den aktuellen Zustand, bestimmt den
        Berechnungsumfang und startet anschließend einen Hintergrundthread für die
        eigentliche KPI-Ermittlung. Je nach scope werden entweder Linien-KPIs für
        einen ausgewählten Tag oder aggregierte Gesamt-KPIs inklusive Jahresranking
        berechnet. Ergebnisse und Fehlhinweise werden nach Abschluss wieder sicher in
        die GUI zurückgeführt.

        Parameter:
            scope (str): Berechnungsmodus, typischerweise "line" für eine einzelne
                Linie oder "total" für aggregierte Gesamt-KPIs.

        Rückgabewerte:
            None: Die Methode stößt asynchron eine Berechnung an und aktualisiert
            später die GUI.

        Fehler/Sonderfälle:
            Läuft bereits eine KPI-Berechnung, wird keine zweite gestartet. Fehlt bei
            scope="line" eine gültige Linienauswahl, wird ein Hinweistext ausgegeben.
            Tritt in der Hintergrundberechnung ein Fehler auf, werden Status- und
            Textausgaben entsprechend gesetzt, statt die Anwendung abbrechen zu
            lassen.

        Projektkontext:
            Diese Methode ist der operative Einstiegspunkt für die fachliche
            Bewertungslogik des Projekts und verbindet GUI, Cache, Prediction-Modell,
            Fahrplandaten und KPI-Aggregation.
        """
        if self.kpi_busy:
            if self.kpi_status_label is not None:
                self.kpi_status_label.configure(text="KPI-Berechnung laeuft bereits im Hintergrund...")
            return
        selected_day = self.kpi_day_entry.get_date() if self.kpi_day_entry is not None else DEFAULT_PREDICTION_DATE
        cost = self._current_bus_hourly_cost()
        if scope == "line":
            line = self._selected_kpi_line()
            if line is None:
                self._write_kpi_text(None, "Keine Linie für KPI-Vergleich ausgewählt.")
                return
            lines = [line]
            label = f"Linie {line}"
        else:
            lines = self.timetable_repo.available_lines()
            label = "Jahres-Gesamtfahrplan"
            if not lines:
                self._write_kpi_text("Keine Fahrplandaten für Gesamt-KPIs geladen.", None)
                return


        self._set_kpi_busy(True, f"{label}: KPI-Vergleich wird im Hintergrund berechnet...")


        def worker() -> tuple[str, KPIResult | None, list[KPIResult], list[str], str | None]:
            """
            Führt die eigentliche KPI-Berechnung im Hintergrund aus.

            Die Funktion kapselt die rechenintensive Logik, damit die GUI während der
            Berechnung responsiv bleibt. Für Gesamt-KPIs werden, sofern vorhanden,
            Jahresergebnisse und Rankings aus Caches verwendet; andernfalls werden sie
            neu erzeugt. Für Linien-KPIs werden Ergebnisse pro Linie gesammelt und bei
            Bedarf aggregiert.

            Parameter:
                Keine.

            Rückgabewerte:
                tuple[str, KPIResult | None, list[KPIResult], list[str], str | None]:
                Scope, Hauptergebnis, Linienergebnisse, Hinweisliste und optionaler
                Traceback-Text bei Fehlern.

            Fehler/Sonderfälle:
                Fachliche Teilfehler einzelner Linien werden als Hinweistexte
                gesammelt. Unerwartete Ausnahmen werden als Traceback zurückgegeben,
                damit die GUI später kontrolliert reagieren kann.

            Projektkontext:
                Die Trennung in eine Worker-Funktion ermöglicht nebenläufige KPI-
                Berechnungen, ohne die Interaktionsfähigkeit des Dashboards zu
                blockieren.
            """
            errors: list[str] = []
            results: list[KPIResult] = []
            try:
                if scope == "total":
                    cache_key = (2025, round(cost, 2))
                    total = self.kpi_annual_cache.get(cache_key)
                    rankings = self.kpi_ranking_cache.get(cache_key)
                    if total is None:
                        total = calculate_annual_kpis_from_parquet_dir(
                            ENRICHED_TRAINING_DIR,
                            year=2025,
                            label="Gesamtfahrplan KPI-Jahresstandard",
                            bus_hourly_cost=cost,
                        )
                        self.kpi_annual_cache[cache_key] = total
                    if rankings is None:
                        rankings = calculate_annual_line_kpis_from_parquet_dir(
                            ENRICHED_TRAINING_DIR,
                            year=2025,
                            bus_hourly_cost=cost,
                        )
                        self.kpi_ranking_cache[cache_key] = rankings
                    if total.hours <= 0:
                        errors.append("Keine enriched-2025 Jahresdaten für Gesamt-KPIs gefunden.")
                    return scope, total, rankings, errors, None
                for line_number in lines:
                    try:
                        result = self._build_kpi_result_for_line(line_number, selected_day, cost)
                        if result.hours > 0:
                            results.append(result)
                        else:
                            errors.append(f"Linie {line_number}: keine Schedule-KPIs.")
                    except Exception as exc:
                        errors.append(f"Linie {line_number}: {exc}")
                if scope == "line":
                    return scope, (results[0] if results else None), results, errors, None
                total = aggregate_kpis(results, label="Gesamtfahrplan KPI-Vergleich", selected_day=selected_day)
                return scope, total, results, errors, None
            except Exception:
                return scope, None, results, errors, traceback.format_exc()


        def finish(scope_name: str, result: KPIResult | None, line_results: list[KPIResult], errors: list[str], details: str | None) -> None:
            """
            Überführt das Berechnungsergebnis aus dem Hintergrund in die GUI.

            Die Funktion hebt den Busy-Zustand auf und entscheidet, ob ein
            Fehlerzustand oder ein erfolgreiches Ergebnis vorliegt. Je nach Fall
            werden Berichtstexte, Statusmeldungen und Rankings aktualisiert.

            Parameter:
                scope_name (str): Tatsächlich ausgeführter Berechnungsmodus.
                result (KPIResult | None): Hauptergebnis der KPI-Berechnung.
                line_results (list[KPIResult]): Gegebenenfalls berechnete
                    Linienergebnisse.
                errors (list[str]): Gesammelte Hinweise oder Teilfehler.
                details (str | None): Optionaler Detailtext, etwa ein Traceback.

            Rückgabewerte:
                None: Die Funktion aktualisiert ausschließlich die GUI.

            Fehler/Sonderfälle:
                Fehlt ein verwertbares Ergebnis oder liegen Detailfehler vor, wird
                ein Fehlertext in die passenden Ausgabefelder geschrieben.

            Projektkontext:
                Die Funktion ist die Brücke zwischen Hintergrundthread und
                Benutzeroberfläche und sorgt für eine kontrollierte Ergebnisdarstellung.
            """
            self._set_kpi_busy(False)
            if details or result is None:
                message = "KPI-Vergleich konnte nicht berechnet werden."
                if details:
                    message += "\n\n" + details
                if errors:
                    message += "\n\nHinweise:\n" + "\n".join(errors[:12])
                self._write_kpi_text(message if scope_name == "total" else None, message if scope_name == "line" else None)
                if self.kpi_status_label is not None:
                    self.kpi_status_label.configure(text="KPI-Berechnung fehlgeschlagen.")
                return
            self._render_kpi_result(scope_name, result, line_results, errors)


        def run_worker() -> None:
            """
            Führt die Worker-Berechnung aus und plant die GUI-Rückgabe im Hauptthread.

            Die Funktion dient als Thread-Ziel und stellt sicher, dass das Ergebnis
            der Hintergrundberechnung anschließend über self.after threadsicher in die
            Tkinter-Oberfläche zurückgespielt wird.

            Parameter:
                Keine.

            Rückgabewerte:
                None: Die Funktion steuert nur den Ausführungsablauf.

            Fehler/Sonderfälle:
                Fehlerbehandlung erfolgt in worker und finish; run_worker übernimmt
                lediglich die orchestrierende Verbindung zwischen beiden.

            Projektkontext:
                Diese Hilfsfunktion ist Teil des nebenläufigen Ausführungsmusters der
                KPI-Berechnung und schützt die GUI vor blockierenden Rechenläufen.
            """
            scope_name, result, line_results, errors, details = worker()
            self.after(0, lambda: finish(scope_name, result, line_results, errors, details))


        threading.Thread(target=run_worker, daemon=True).start()


    def _build_kpi_result_for_line(self, line: int, selected_day: date, cost: float) -> KPIResult:
        """
        Erzeugt das KPI-Ergebnis für eine einzelne Linie auf Basis adaptiver
        Prognosefahrten.

        Die Methode stellt sicher, dass für die gewünschte Linie ein passendes
        Vorhersagemodell verfügbar ist, erzeugt anschließend zwei zwölfstündige
        Kurzfrist-Prognosefenster für einen ganzen Betriebstag, führt die
        Teilfahrpläne zusammen und ergänzt sie um die WVV-Referenzfahrten pro Stunde.
        Auf dieser Basis werden dann die Linien-KPIs berechnet.

        Parameter:
            line (int): Zu bewertende Liniennummer.
            selected_day (date): Betrachteter Kalendertag.
            cost (float): Kostenannahme pro Busstunde.

        Rückgabewerte:
            KPIResult: KPI-Ergebnis für die angegebene Linie.

        Fehler/Sonderfälle:
            Ist kein gespeichertes Prediction-Modell verfügbar und kann auch nicht
            geladen werden, wird eine RuntimeError ausgelöst. Entstehen keine
            Prognosefahrten, wird dennoch ein formal gültiges KPI-Ergebnis auf Basis
            eines leeren Fahrplans zurückgegeben.

        Projektkontext:
            Die Methode verbindet Nachfrageprognose, WVV-Referenzfahrplan und
            betriebswirtschaftliche Bewertung auf Linienebene und ist damit ein
            Kernbaustein des adaptiven KPI-Vergleichs.
        """
        if not (self.prediction_service.trained and self.prediction_service.trained_lines == {int(line)}):
            if not self.prediction_service.load_for_lines([int(line)]):
                raise RuntimeError(
                    f"Kein gespeichertes Prediction-Modell für Linie {line} gefunden. "
                    "Bitte einmal im Prediction-Tab trainieren."
                )
        prediction_parts: list[pd.DataFrame] = []
        for start_hour in (0, 12):
            result = self.prediction_service.predict_short_term(
                int(line),
                selected_day,
                start_hour,
                horizon_hours=12,
                cost_per_bus_hour=cost,
            )
            if not result.schedule.empty:
                prediction_parts.append(result.schedule)
        if not prediction_parts:
            return calculate_line_kpis(
                pd.DataFrame(),
                line=int(line),
                label=f"Linie {line} KPI-Vergleich",
                selected_day=selected_day,
                bus_hourly_cost=cost,
            )
        schedule = pd.concat(prediction_parts, ignore_index=True)
        wvv_hourly_runs = self.timetable_repo.hourly_run_counts_for_line(
            int(line),
            selected_day,
            start_hour=0,
            horizon_hours=24,
        )
        schedule = attach_wvv_hourly_runs(schedule, wvv_hourly_runs)
        return calculate_line_kpis(
            schedule,
            line=int(line),
            label=f"Linie {line} KPI-Vergleich",
            selected_day=selected_day,
            bus_hourly_cost=cost,
        )


    def _render_kpi_result(
        self,
        scope: str,
        result: KPIResult,
        line_results: list[KPIResult],
        errors: list[str],
    ) -> None:
        """
        Bereitet ein KPI-Ergebnis für die Anzeige auf und aktualisiert die GUI.

        Die Methode formatiert den KPI-Bericht textuell, ergänzt bei Bedarf Hinweise
        und schreibt das Ergebnis je nach scope in die Gesamt- oder Linienansicht.
        Für Gesamtberechnungen wird zusätzlich das Ranking aktualisiert. Abschließend
        wird eine kompakte Statuszeile mit zentralen Vergleichskennzahlen erzeugt.

        Parameter:
            scope (str): Berechnungsmodus, typischerweise "total" oder "line".
            result (KPIResult): Hauptergebnis der Berechnung.
            line_results (list[KPIResult]): Linienergebnisse, insbesondere relevant
                für Rankings im Gesamtmodus.
            errors (list[str]): Gesammelte Hinweise oder Teilfehler.

        Rückgabewerte:
            None: Die Methode aktualisiert ausschließlich Anzeigeelemente.

        Fehler/Sonderfälle:
            Ist die Hinweisliste länger als zwölf Einträge, wird die Anzeige gekürzt
            und um einen Sammelhinweis ergänzt.

        Projektkontext:
            Die Methode transformiert numerische KPI-Ergebnisse in eine für
            Entwickler, Analysten und fachliche Stakeholder lesbare Form.
        """
        report = format_kpi_report(result)
        if errors:
            report += "\n\nHinweise:\n" + "\n".join(errors[:12])
            if len(errors) > 12:
                report += f"\n... {len(errors) - 12} weitere Hinweise"
        if scope == "total":
            self._write_kpi_text(report, None)
            self._write_kpi_ranking(line_results)
            status_scope = result.period or "Jahresstandard"
        else:
            self._write_kpi_text(None, report)
            status_scope = f"Linie {result.line}"
        if self.kpi_status_label is not None:
            self.kpi_status_label.configure(
                text=(
                    f"{status_scope}: WVV {result.wvv_runs:.0f} Kurse vs. adaptiv {result.adaptive_runs:.0f} Kurse | "
                    f"Coverage {result.adaptive_coverage * 100:.1f}% | Kosten-Delta {result.cost_delta:,.0f} EUR"
                ).replace(",", ".")
            )

    def _write_kpi_text(self, total_text: str | None, line_text: str | None) -> None:
        """
        Schreibt KPI-Berichtstexte in die vorgesehenen Ausgabefelder der GUI.

        Die Methode aktualisiert die Textboxen für Gesamt- und Linien-KPIs gezielt
        mit neuen Berichtsinhalten. Dabei werden nur diejenigen Felder verändert,
        für die sowohl ein Widget als auch ein Text vorhanden sind. Vor dem Schreiben
        wird das jeweilige Feld entsperrt und anschließend wieder schreibgeschützt.

        Parameter:
            total_text (str | None): Neuer Text für die Gesamt-KPI-Textbox oder None,
                wenn diese nicht aktualisiert werden soll.
            line_text (str | None): Neuer Text für die Linien-KPI-Textbox oder None,
                wenn diese nicht aktualisiert werden soll.

        Rückgabewerte:
            None: Die Methode aktualisiert ausschließlich GUI-Elemente.

        Fehler/Sonderfälle:
            Fehlt eine Textbox oder ist für sie kein Text übergeben worden, wird sie
            übersprungen. Dadurch können Gesamt- und Linienausgabe unabhängig
            voneinander aktualisiert werden.

        Projektkontext:
            Die Methode bildet die zentrale Schreibschnittstelle für KPI-Berichte und
            sorgt dafür, dass Analyseergebnisse kontrolliert und konsistent in der
            Oberfläche erscheinen.
        """
        targets = (
            (self.kpi_total_textbox, total_text),
            (self.kpi_line_textbox, line_text),
        )
        for textbox, text in targets:
            if textbox is None or text is None:
                continue
            textbox.configure(state="normal")
            textbox.delete("1.0", "end")
            textbox.insert("1.0", text)
            textbox.configure(state="disabled")


    def _write_kpi_ranking(self, line_results: list[KPIResult]) -> None:
        """
        Schreibt formatierte Linienrankings in die zugehörigen Ranking-Spalten.

        Die Methode erzeugt aus den übergebenen Linienergebnissen strukturierte
        Textdarstellungen für Einsparpotenzial, Überlastung und Unterauslastung und
        überträgt diese in die jeweils zugehörigen Textboxen. Vor dem Schreiben
        werden die Felder temporär entsperrt und danach wieder deaktiviert.

        Parameter:
            line_results (list[KPIResult]): Linienbezogene KPI-Ergebnisse, aus denen
                die Rankings abgeleitet werden.

        Rückgabewerte:
            None: Die Methode aktualisiert ausschließlich die Rankinganzeige.

        Fehler/Sonderfälle:
            Sind keine Ranking-Textboxen initialisiert, beendet sich die Methode ohne
            weitere Aktion. Fehlen für einzelne Rankingtypen Daten, wird ein
            Standardhinweis geschrieben.

        Projektkontext:
            Die Rankingdarstellung verdichtet Linienergebnisse zu priorisierbaren
            Managementinformationen und unterstützt die Identifikation besonders
            auffälliger Linien.
        """
        if not self.kpi_ranking_boxes:
            return
        rankings = self._format_line_ranking_tables(line_results)
        for key, textbox in self.kpi_ranking_boxes.items():
            textbox.configure(state="normal")
            textbox.delete("1.0", "end")
            textbox.insert("1.0", rankings.get(key, "Keine Daten."))
            textbox.configure(state="disabled")


    def _format_line_ranking_tables(self, line_results: list[KPIResult]) -> dict[str, str]:
        """
        Formatiert Linienergebnisse zu textbasierten Rankingtabellen.

        Die Methode erzeugt drei textuelle Ranglisten für unterschiedliche fachliche
        Perspektiven: Einsparpotenzial gegenüber dem WVV-Bestand, rechnerische
        Überlastung sowie Unterauslastung beziehungsweise Leerkapazitäten. Die
        Ausgabe ist so formatiert, dass sie in monospace-basierten Textboxen gut
        lesbar bleibt.

        Parameter:
            line_results (list[KPIResult]): Linienbezogene KPI-Ergebnisse, die
                ausgewertet und sortiert werden.

        Rückgabewerte:
            dict[str, str]: Mapping mit den Schlüsseln "savings", "overload" und
            "underload", jeweils mit einer fertig formatierten Texttabelle.

        Fehler/Sonderfälle:
            Liegen keine Linienergebnisse vor, wird für alle Rankingtypen ein
            einheitlicher Hinweistext zurückgegeben. Große Zahlen werden kompakt oder
            mit Tausendertrennzeichen aufbereitet, um die Darstellung in Textboxen zu
            stabilisieren.

        Projektkontext:
            Die Methode übersetzt numerische KPI-Ergebnisse in ein kompaktes,
            stakeholder-taugliches Vergleichsformat, das in der GUI ohne zusätzliche
            Tabellenkomponente auskommt.
        """
        if not line_results:
            return {
                "savings": "Keine Linienranking-Daten verfügbar.",
                "overload": "Keine Linienranking-Daten verfügbar.",
                "underload": "Keine Linienranking-Daten verfügbar.",
            }


        def fmt_number(value: float) -> str:
            """
            Formatiert eine Zahl mit Tausendertrennzeichen für Berichtsansichten.

            Die Funktion dient der besser lesbaren Darstellung größerer numerischer
            KPI-Werte in textbasierten Rankingtabellen.

            Parameter:
                value (float): Zu formatierender Zahlenwert.

            Rückgabewerte:
                str: Zahl ohne Nachkommastellen mit deutsch anmutender
                Tausendertrennung per Punkt.

            Fehler/Sonderfälle:
                Nachkommastellen werden bewusst unterdrückt, um die Texttabellen
                kompakt zu halten.

            Projektkontext:
                Einheitliche Zahlenformate erleichtern den Vergleich zwischen Linien
                und verbessern die Lesbarkeit in monospace-basierten Berichten.
            """
            return f"{value:,.0f}".replace(",", ".")


        def fmt_compact(value: float) -> str:
            """
            Verdichtet numerische Werte in eine kompakte Kurzschreibweise.

            Werte im Tausender- oder Millionenbereich werden zu einer kurzen Darstellung
            mit k- bzw. M-Suffix reduziert. Dadurch bleiben tabellarische
            Berichtsausgaben auch bei großen Werten platzsparend lesbar.

            Parameter:
                value (float): Zu formatierender Zahlenwert.

            Rückgabewerte:
                str: Kompakt formatierter Zahlenwert.

            Fehler/Sonderfälle:
                Negative Vorzeichen bleiben erhalten. Kleinere Werte werden ohne
                Suffix und ohne Nachkommastellen ausgegeben.

            Projektkontext:
                Die kompakte Darstellung ist besonders für Linienrankings relevant, da
                dort mehrere Kennzahlen in begrenzter Breite nebeneinander angezeigt
                werden.
            """
            sign = "-" if value < 0 else ""
            absolute = abs(float(value))
            if absolute >= 1_000_000:
                return f"{sign}{absolute / 1_000_000:.1f}M"
            if absolute >= 1_000:
                return f"{sign}{absolute / 1_000:.0f}k"
            return f"{value:.0f}"


        def fmt_eur_short(value: float) -> str:
            """
            Ergänzt einen kompakten Zahlenwert um die Währungseinheit EUR.

            Die Funktion baut auf der kompakten Zahlenformatierung auf und erzeugt
            eine kurze, tabellentaugliche Darstellung von Kosten- oder
            Einsparungswerten.

            Parameter:
                value (float): Zu formatierender Geldwert.

            Rückgabewerte:
                str: Kompakt formatierter Wert mit angehängter Währungseinheit.

            Fehler/Sonderfälle:
                Die Funktion übernimmt die Sonderfallbehandlung vollständig aus
                fmt_compact.

            Projektkontext:
                Kostenmetriken sind eine zentrale Perspektive im Ranking des
                adaptiven Fahrplankonzepts und müssen platzsparend darstellbar sein.
            """
            return f"{fmt_compact(value)} EUR"


        def line_name(item: KPIResult) -> str:
            """
            Erzeugt eine kompakte Linienbezeichnung für Rankingeinträge.

            Die Funktion extrahiert die Liniennummer aus einem KPI-Ergebnis und
            überführt sie in eine konsistente, menschenlesbare Kurzbezeichnung.

            Parameter:
                item (KPIResult): KPI-Ergebnis einer Linie.

            Rückgabewerte:
                str: Formatierte Linienbezeichnung.

            Fehler/Sonderfälle:
                Fehlt eine Liniennummer, wird ein Platzhalter ausgegeben.

            Projektkontext:
                Eine einheitliche Linienbenennung ist notwendig, damit die
                Ranglisten fachlich schnell erfassbar bleiben.
            """
            return f"Linie {int(item.line)}" if item.line is not None else "Linie ?"


        savings = sorted(line_results, key=lambda row: row.wvv_cost - row.adaptive_cost, reverse=True)
        overload = sorted(line_results, key=lambda row: row.wvv_overload, reverse=True)
        underload = sorted(line_results, key=lambda row: row.wvv_underload, reverse=True)


        savings_lines = [
            "Jahresstandard 2025",
            "Basis: aktueller Kostenwert",
            "",
            "Rang Linie   Sparen   Kurse   Cov.",
            "-" * 40,
        ]
        for rank, item in enumerate(savings[:10], start=1):
            saving = item.wvv_cost - item.adaptive_cost
            savings_lines.append(
                f"{rank:>2}.  {line_name(item):<7} {fmt_eur_short(saving):>9} "
                f"{fmt_compact(item.wvv_runs):>4}->{fmt_compact(item.adaptive_runs):<4} "
                f"{item.adaptive_coverage * 100:>6.1f}%"
            )


        overload_lines = [
            "WVV-Bestand",
            "Hohe Werte = zu wenig Kapazität",
            "",
            "Rang Linie   Überlast NichtBed Ausl.",
            "-" * 42,
        ]
        for rank, item in enumerate(overload[:10], start=1):
            overload_lines.append(
                f"{rank:>2}.  {line_name(item):<7} {fmt_number(item.wvv_overload):>8} "
                f"{fmt_number(item.wvv_unserved):>8} {item.wvv_utilization * 100:>5.1f}%"
            )


        underload_lines = [
            "WVV-Bestand",
            "Hohe Werte = viele Leerkapazitäten",
            "",
            "Rang Linie   Unterl. Kurse   Sparen",
            "-" * 40,
        ]
        for rank, item in enumerate(underload[:10], start=1):
            saving = item.wvv_cost - item.adaptive_cost
            underload_lines.append(
                f"{rank:>2}.  {line_name(item):<7} {fmt_number(item.wvv_underload):>8} "
                f"{fmt_compact(item.wvv_runs):>5}->{fmt_compact(item.adaptive_runs):<4} "
                f"{fmt_eur_short(saving):>8}"
            )
        return {
            "savings": "\n".join(savings_lines),
            "overload": "\n".join(overload_lines),
            "underload": "\n".join(underload_lines),
        }


    def _on_timetable_hour_slider(self, value: float) -> None:
        """
        Aktualisiert die Stundenanzeige des Fahrplan-Sliders.

        Die Methode rundet den Sliderwert auf eine volle Stunde und schreibt die
        formatierte Uhrzeit in das zugehörige Label.

        Parameter:
            value (float): Aktueller Wert des Sliders.

        Rückgabewerte:
            None: Die Methode aktualisiert ausschließlich ein GUI-Label.

        Fehler/Sonderfälle:
            Ist kein Stundenlabel vorhanden, wird keine Aktion ausgeführt.

        Projektkontext:
            Die Methode dient der unmittelbaren Rückmeldung in der GUI, wenn
            zeitbezogene Fahrplanansichten interaktiv gesteuert werden.
        """
        hour = int(round(float(value)))
        if self.timetable_hour_label is not None:
            self.timetable_hour_label.configure(text=f"{hour:02d}:00 Uhr")


    def _timetable_horizon_hours(self) -> int:
        """
        Extrahiert die gewünschte Betrachtungsdauer für Fahrplanvergleiche.

        Die Methode liest aus der textuellen Auswahl den ersten numerischen Wert aus
        und interpretiert ihn als Anzahl der Horizontstunden.

        

        Rückgabewerte:
            int: Anzahl der Stunden für den Vergleichshorizont.

        Fehler/Sonderfälle:
            Enthält die Auswahl keinen numerischen Wert, wird standardmäßig ein
            Vier-Stunden-Horizont verwendet.

        Projektkontext:
            Die Methode kapselt die Übersetzung von GUI-Texten in fachlich nutzbare
            Parameter für Fahrplanberechnungen.
        """
        match = re.search(r"\d+", self.timetable_horizon_choice.get())
        return int(match.group(0)) if match else 4


    def _set_timetable_busy(self, busy: bool, message: str = "") -> None:
        """
        Setzt den Busy-Zustand der Fahrplanansicht und sperrt Eingabeelemente.

        Während laufender Fahrplanberechnungen werden zentrale Steuerelemente
        deaktiviert, um konkurrierende Eingaben und inkonsistente Zustände zu
        vermeiden. Optional wird eine Statusnachricht in der GUI angezeigt.

        Parameter:
            busy (bool): Gibt an, ob die Fahrplanansicht als beschäftigt markiert
                werden soll.
            message (str): Optionale Statusmeldung für die Oberfläche.

        Rückgabewerte:
            None: Die Methode verändert ausschließlich GUI-Zustände.

        Fehler/Sonderfälle:
            Fehlende Widgets werden übersprungen. Unterschiedliches Verhalten bei
            Widget-Konfigurationen wird über TclError tolerant behandelt.

        Projektkontext:
            Der Busy-Zustand ist notwendig, damit langlaufende Hintergrundberechnungen
            die Benutzerführung nicht stören und die Oberfläche konsistent bleibt.
        """
        self.timetable_busy = busy
        if self.timetable_status_label is not None and message:
            self.timetable_status_label.configure(text=message)
        state = "disabled" if busy else "normal"
        for widget in (
            self.timetable_calculate_button,
            self.timetable_line_menu,
            self.timetable_day_entry,
        ):
            if widget is None:
                continue
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass


    def _calculate_timetable_comparison(self) -> None:
        """
        Startet die ganztägige Fahrplanvergleichsberechnung für die ausgewählte Linie.

        Die Methode validiert zunächst den aktuellen GUI-Zustand, bestimmt Linie und
        Betrachtungstag und stößt anschließend eine Hintergrundberechnung an. Dabei
        wird versucht, einen adaptiven Prognosefahrplan aus dem Prediction-Service zu
        erzeugen; falls dies nicht gelingt, wird ein fachlich sinnvoller Fallback auf
        Basis des WVV-Fahrplans verwendet. Nach Abschluss werden Ergebnis oder Fehler
        wieder threadsicher in die GUI übernommen.

        

        Rückgabewerte:
            None: Die Methode startet asynchron eine Berechnung und aktualisiert
            später die Oberfläche.

        Fehler/Sonderfälle:
            Läuft bereits eine Berechnung, wird kein zweiter Hintergrundlauf
            gestartet. Ist keine gültige Linie ausgewählt, werden Hinweistexte
            angezeigt. Prediction-Fehler führen nicht zum Abbruch des gesamten
            Fahrplanvergleichs, sondern zu einem dokumentierten Fallback.

        Projektkontext:
            Diese Methode ist der zentrale Einstiegspunkt für den operativen
            Vergleich zwischen bestehendem WVV-Fahrplan und adaptiv abgeleitetem
            Vorschlagsfahrplan.
        """
        if self.timetable_busy:
            if self.timetable_status_label is not None:
                self.timetable_status_label.configure(text="Fahrplanberechnung läuft bereits im Hintergrund...")
            return
        line = self._selected_timetable_line()
        if line is None:
            self._write_timetable_text("Keine Fahrplandaten geladen.", "Keine Fahrplandaten geladen.")
            return
        selected_day = self.timetable_day_entry.get_date() if self.timetable_day_entry is not None else DEFAULT_PREDICTION_DATE
        hour = 0
        horizon = 24
        self._set_timetable_busy(
            True,
            f"Ganztags-Fahrplan für Linie {line} wird im Hintergrund berechnet...",
        )
        self._write_timetable_text(
            "Ganztags-Fahrplan wird geladen.\n\nDie GUI bleibt bedienbar.",
            "Adaptiver Ganztags-Fahrplan wird geladen.\n\nPrediction wird falls vorhanden im Hintergrund genutzt.",
        )


        def worker() -> tuple[TimetableComparison | None, str, str | None]:
            """
            Führt die Fahrplanvergleichsberechnung im Hintergrund aus.

            Die Funktion erzeugt zunächst optional einen adaptiven Prognosefahrplan
            in zwei Tageshälften und übergibt diesen danach an das Fahrplanrepository,
            das den eigentlichen Vergleich mit dem WVV-Bestand erstellt.

            Parameter:
                Keine.

            Rückgabewerte:
                tuple[TimetableComparison | None, str, str | None]: Vergleichsobjekt,
                Prognosehinweis und optionaler Fehlertext.

            Fehler/Sonderfälle:
                Schlägt die Nutzung des Prediction-Service fehl, wird mit leerem
                Prognosefahrplan weitergerechnet und ein entsprechender Hinweistext
                gesetzt. Unerwartete Fehler der Vergleichsberechnung werden als
                Traceback zurückgegeben.

            Projektkontext:
                Die Funktion entkoppelt rechenintensive Vergleichslogik von der GUI
                und erlaubt einen robusten Umgang mit optional verfügbaren
                Vorhersagemodellen.
            """
            prediction_schedule = pd.DataFrame()
            prediction_note = "Kein Prognosemodell geladen. Adaptiver Fahrplan nutzt WVV-Fahrten als Fallback."
            try:
                if self.prediction_service.trained or self.prediction_service.load_for_lines([line]):
                    prediction_parts = []
                    messages = []
                    for start_hour in (0, 12):
                        result = self.prediction_service.predict_short_term(
                            line,
                            selected_day,
                            start_hour,
                            horizon_hours=12,
                            cost_per_bus_hour=self._current_bus_hourly_cost(),
                        )
                        if not result.schedule.empty:
                            prediction_parts.append(result.schedule)
                        if result.message:
                            messages.append(result.message)
                    if prediction_parts:
                        prediction_schedule = pd.concat(prediction_parts, ignore_index=True)
                    prediction_note = messages[-1] if messages else "Ganztags-Prediction berechnet."
            except Exception as exc:
                prediction_schedule = pd.DataFrame()
                prediction_note = f"Prediction konnte nicht genutzt werden: {exc}"


            try:
                comparison = self.timetable_repo.build_comparison(
                    line=line,
                    selected_day=selected_day,
                    start_hour=hour,
                    horizon_hours=horizon,
                    prediction_schedule=prediction_schedule,
                )
                return comparison, prediction_note, None
            except Exception:
                return None, prediction_note, traceback.format_exc()


        def finish(comparison: TimetableComparison | None, prediction_note: str, error: str | None) -> None:
            """
            Übernimmt das Ergebnis der Hintergrundberechnung in die GUI.

            Die Funktion hebt den Busy-Zustand auf und entscheidet, ob ein
            Fehlerzustand oder ein erfolgreich berechneter Fahrplanvergleich
            vorliegt. Entsprechend werden Statuslabel und Textausgaben gesetzt.

            Parameter:
                comparison (TimetableComparison | None): Berechneter
                    Fahrplanvergleich oder None bei Fehlschlag.
                prediction_note (str): Hinweis zur Nutzung oder Nichtnutzung des
                    Prediction-Service.
                error (str | None): Optionaler Fehlertext beziehungsweise Traceback.

            Rückgabewerte:
                None: Die Funktion aktualisiert nur GUI-Elemente.

            Fehler/Sonderfälle:
                Bei Fehlern wird nur die letzte Zeile des Fehlertexts für den
                kompakten Status verwendet, während der vollständige Text in den
                Berichtsfeldern erscheint.

            Projektkontext:
                Die Funktion bildet die threadsichere Rückführungslogik von der
                Hintergrundberechnung in die Benutzeroberfläche.
            """
            self._set_timetable_busy(False)
            if error or comparison is None:
                short = (error or "Unbekannter Fehler").splitlines()[-1]
                if self.timetable_status_label is not None:
                    self.timetable_status_label.configure(text=f"Fahrplan konnte nicht berechnet werden: {short}")
                self._write_timetable_text(
                    "Fahrplanfehler.\n\n" + (error or ""),
                    "Adaptiver Fahrplanfehler.\n\n" + (error or ""),
                )
                return
            self._render_timetable_comparison(comparison, prediction_note)


        def run_worker() -> None:
            """
            Führt die Worker-Funktion aus und plant die GUI-Aktualisierung im Hauptthread.

            Die Funktion dient als Thread-Ziel und sorgt dafür, dass das Ergebnis der
            Hintergrundberechnung anschließend per self.after sicher an die Tkinter-
            Oberfläche übergeben wird.

            Parameter:
                Keine.

            Rückgabewerte:
                None: Die Funktion koordiniert ausschließlich den Ausführungsablauf.

            Fehler/Sonderfälle:
                Fehlerbehandlung erfolgt in worker und finish; run_worker übernimmt
                nur die Orchestrierung.

            Projektkontext:
                Die Hilfsfunktion ist Teil des Threading-Musters für responsive
                Fahrplanberechnungen in der GUI.
            """
            comparison, prediction_note, error = worker()
            self.after(0, lambda: finish(comparison, prediction_note, error))


        threading.Thread(target=run_worker, daemon=True).start()


    def _render_timetable_comparison(self, comparison: TimetableComparison, prediction_note: str = "") -> None:
        """
        Rendert einen berechneten Fahrplanvergleich in Statusanzeige, Routenansicht
        und Texttabellen.

        Die Methode erzeugt zunächst eine zusammenfassende Statusmeldung mit Linie,
        Betriebstag, Fahrtenanzahl und optionalem Prediction-Hinweis. Anschließend
        werden die grafische Routendarstellung sowie die tabellarischen WVV- und
        adaptiven Fahrplandaten in die GUI geschrieben.

        Parameter:
            comparison (TimetableComparison): Vollständig berechneter Vergleich
                zwischen WVV- und adaptivem Fahrplan.
            prediction_note (str): Optionaler Zusatzhinweis zur Herkunft oder Qualität
                des adaptiven Fahrplans.

        Rückgabewerte:
            None: Die Methode aktualisiert ausschließlich GUI-Elemente.

        Fehler/Sonderfälle:
            Leere oder fehlende Notizen werden toleriert. Die Tabellen werden über
            separate Formatierungs- und Schreibmethoden robust in Textform dargestellt.

        Projektkontext:
            Diese Methode transformiert das fachliche Vergleichsobjekt in eine für
            Entwickler und Stakeholder interpretierbare Visualisierung und
            Berichtsausgabe.
        """
        note = str(comparison.summary.get("note", ""))
        status = (
            f"Linie {comparison.line} | {comparison.service_label} | "
            f"{comparison.summary.get('selected_day', '-')} | Ganzer Betriebstag | "
            f"WVV {comparison.summary.get('wvv_runs', 0)} Fahrten vs. adaptiv {comparison.summary.get('adaptive_runs', 0)} Fahrten. "
            f"{note}"
        )
        if prediction_note:
            status += f" | {prediction_note}"
        if self.timetable_status_label is not None:
            self.timetable_status_label.configure(text=status)


        self._draw_timetable_route(comparison)
        source = f"Quelle: {comparison.source_file}\nGültigkeit: {comparison.service_label}\n\n"
        self._write_timetable_text(
            source + self._format_timetable_table(comparison.wvv_table),
            source + self._format_timetable_table(comparison.adaptive_table) + "\n\n* = adaptiv neu getaktete Fahrt",
        )


    def _write_timetable_text(self, wvv_text: str, adaptive_text: str) -> None:
        """
        Schreibt die textuellen Fahrplantabellen in die beiden Vergleichs-Textboxen.

        Die Methode aktualisiert getrennt die Ausgabefelder für den WVV-Fahrplan und
        den adaptiven Fahrplan. Vor dem Schreiben werden die Textboxen entsperrt,
        geleert und anschließend wieder deaktiviert.

        Parameter:
            wvv_text (str): Inhalt für die Textbox des WVV-Fahrplans.
            adaptive_text (str): Inhalt für die Textbox des adaptiven Fahrplans.

        Rückgabewerte:
            None: Die Methode aktualisiert ausschließlich GUI-Ausgabefelder.

        Fehler/Sonderfälle:
            Nicht initialisierte Textboxen werden übersprungen.

        Projektkontext:
            Die Methode bildet die zentrale Ausgabeschnittstelle für den textuellen
            Fahrplanvergleich und sorgt für konsistente Darstellung beider Varianten.
        """
        for textbox, text in (
            (self.timetable_wvv_textbox, wvv_text),
            (self.timetable_adaptive_textbox, adaptive_text),
        ):
            if textbox is None:
                continue
            textbox.configure(state="normal")
            textbox.delete("1.0", "end")
            textbox.insert("1.0", text)
            textbox.configure(state="disabled")


    def _format_timetable_table(self, table: pd.DataFrame) -> str:
        """
        Formatiert eine Fahrplantabelle für die textbasierte GUI-Ausgabe.

        Die Methode bereitet ein DataFrame so auf, dass es in einer monospace-
        Textbox lesbar dargestellt werden kann. Optional vorhandene Markerspalten
        werden dabei fachlich verständlicher als Hinweis-Spalte benannt.

        Parameter:
            table (pd.DataFrame): Zu formatierende Fahrplantabelle.

        Rückgabewerte:
            str: Formatierte Tabellenrepräsentation oder ein Hinweistext bei leeren
            Daten.

        Fehler/Sonderfälle:
            Leere Tabellen führen zu einem expliziten Hinweistext. Fehlende Werte
            werden mit "-" ersetzt. Marker ohne Inhalt werden als Leerzeichen
            dargestellt, damit die Spaltenstruktur erhalten bleibt.

        Projektkontext:
            Die Methode übersetzt tabellarische Fahrplandaten in eine kompakte
            Darstellung, die ohne zusätzliche Tabellen-Widgets in der GUI nutzbar ist.
        """
        if table.empty:
            return "Keine Daten für diese Auswahl."
        display = table.copy()
        if "marker" in display.columns:
            display = display.rename(columns={"marker": "Hinweis"})
            display["Hinweis"] = display["Hinweis"].fillna("").replace("", " ")
        with pd.option_context("display.max_columns", None, "display.width", 220, "display.max_colwidth", 24):
            return display.fillna("-").to_string(index=False)


    def _draw_timetable_route(self, comparison: TimetableComparison) -> None:
        """
        Zeichnet die Route einer Linie als schematische Fahrplanübersicht auf ein Canvas.

        Die Methode erstellt eine abstrahierte Liniengrafik mit Start- und Zielhalt,
        Haltestellenpunkten sowie typischen Fahrzeiten zwischen aufeinanderfolgenden
        Stationen. Grundlage ist die im Vergleichsobjekt enthaltene Route und, sofern
        verfügbar, die erste WVV-Fahrt zur Ermittlung typischer Minutenabstände. Der
        Tkinter-Canvas eignet sich für solche frei positionierbaren grafischen
        Elemente wie Linien, Text und Marker. [web:89][web:93]

        Parameter:
            comparison (TimetableComparison): Vergleichsobjekt mit Routen- und
                Fahrplandaten.

        Rückgabewerte:
            None: Die Methode zeichnet ausschließlich auf das Canvas-Widget.

        Fehler/Sonderfälle:
            Ist kein Canvas vorhanden oder enthält die Route keine Stationen, wird
            keine reguläre Grafik erzeugt beziehungsweise ein Hinweistext angezeigt.
            Fahrzeitannotation zwischen Halten erfolgt nur, wenn in der ersten
            WVV-Fahrt auswertbare Uhrzeiten gefunden werden.

        Projektkontext:
            Die Routenvisualisierung ergänzt den tabellarischen Fahrplanvergleich um
            eine räumlich-sequenzielle Sicht auf die Linie und erleichtert die
            fachliche Interpretation der Haltestellenfolge.
        """
        canvas = self.timetable_route_canvas
        if canvas is None:
            return
        canvas.delete("all")
        canvas.update_idletasks()
        width = max(canvas.winfo_width(), 900)
        height = max(canvas.winfo_height(), 340)
        canvas.create_rectangle(0, 0, width, height, fill=PALETTE["surface_alt"], outline="")
        canvas.create_rectangle(14, 14, width - 14, 76, fill=PALETTE["surface"], outline=PALETTE["border"], width=1)
        canvas.create_text(36, 45, text="BUS", fill=PALETTE["accent"], font=("Segoe UI", 12, "bold"), anchor="w")


        route = comparison.route
        if route.empty:
            canvas.create_text(width / 2, height / 2, text="Keine Route gefunden", fill=PALETTE["muted"], font=("Segoe UI", 14, "bold"))
            return


        first_stop = str(route["station"].iloc[0])
        last_stop = str(route["station"].iloc[-1])
        title = self._ellipsize(f"Linie {comparison.line}  {first_stop} - {last_stop}", max(32, int(width / 18)))
        canvas.create_text(95, 45, text=title, fill=PALETTE["text"], font=("Bahnschrift", 20, "bold"), anchor="w")
        canvas.create_text(width - 32, 45, text=comparison.service_label, fill=PALETTE["muted"], font=("Segoe UI", 12, "bold"), anchor="e")


        count = len(route)
        x0, x1 = 54, width - 54
        y = 245
        step = (x1 - x0) / max(count - 1, 1)
        points = [(x0 + index * step, y) for index in range(count)]
        canvas.create_line(x0, y, x1, y, fill="#cbd5e1", width=3)


        first_trip_columns = [col for col in comparison.wvv_table.columns if re.match(r"\d{2} \d{2}:\d{2}", str(col))]
        first_trip = first_trip_columns[0] if first_trip_columns else None
        first_trip_minutes: list[int | None] = []
        if first_trip and "Haltestelle" in comparison.wvv_table:
            for _, row in comparison.wvv_table.iterrows():
                first_trip_minutes.append(self._parse_route_time(row.get(first_trip)))


        for index, (_, row) in enumerate(route.iterrows()):
            x, _y = points[index]
            canvas.create_oval(x - 8, y - 8, x + 8, y + 8, fill=PALETTE["surface_alt"], outline="#e2e8f0", width=2)
            station_label = self._ellipsize(str(row["station"]), 18)
            canvas.create_text(x - 8, y - 38, text=station_label, fill=PALETTE["text"], font=("Segoe UI", 10, "bold"), angle=55, anchor="sw")
            if index > 0 and index < len(first_trip_minutes):
                previous = first_trip_minutes[index - 1]
                current = first_trip_minutes[index]
                if previous is not None and current is not None:
                    mid_x = (points[index - 1][0] + x) / 2
                    canvas.create_text(mid_x, y + 34, text=str(max(0, current - previous)), fill=PALETTE["muted"], font=("Segoe UI", 10, "bold"))


        canvas.create_text(
            width / 2,
            height - 24,
            text="Zahl zwischen zwei Punkten = typische Fahrzeit in Minuten aus dem WVV-Fahrplan",
            fill=PALETTE["muted"],
            font=("Segoe UI", 10),
        )


    def _ellipsize(self, text: str, max_chars: int) -> str:
        """
        Kürzt einen Text auf eine maximale Zeichenlänge und ergänzt Auslassungspunkte.

        Die Methode dient dazu, längere Haltestellen- oder Routennamen für enge
        GUI-Bereiche lesbar zu halten, ohne das Layout zu sprengen.

        Parameter:
            text (str): Ursprünglicher Text.
            max_chars (int): Maximale Anzahl darzustellender Zeichen.

        Rückgabewerte:
            str: Unveränderter oder gekürzter Text mit Auslassungszeichen.

        Fehler/Sonderfälle:
            Bei sehr kleinen Grenzwerten wird sichergestellt, dass mindestens ein
            Zeichen vor dem Auslassungszeichen erhalten bleibt.

        Projektkontext:
            Kompakte Beschriftungen sind für die grafische Routendarstellung und
            andere dichte GUI-Bereiche essenziell.
        """
        if len(text) <= max_chars:
            return text
        return text[: max(max_chars - 1, 1)].rstrip() + "…"


    def _parse_route_time(self, value: object) -> int | None:
        """
        Wandelt eine Uhrzeit aus einer Fahrplantabelle in Minuten seit Tagesbeginn um.

        Die Methode extrahiert ein Uhrzeitmuster im Format HH:MM aus einem beliebigen
        Eingabewert und berechnet daraus die Gesamtminuten seit 00:00 Uhr.

        Parameter:
            value (object): Eingabewert, typischerweise ein Tabellenfeld mit
                Uhrzeitinformation.

        Rückgabewerte:
            int | None: Minuten seit Tagesbeginn oder None, falls keine gültige
            Uhrzeit erkannt wird.

        Fehler/Sonderfälle:
            Nicht passende, leere oder None-Werte führen zu None statt zu einer
            Ausnahme.

        Projektkontext:
            Die Methode wird verwendet, um aus Fahrplanzeiten Zwischenfahrzeiten in
            der grafischen Linienansicht ableiten zu können.
        """
        match = re.search(r"(\d{2}):(\d{2})", "" if value is None else str(value))
        if not match:
            return None
        return int(match.group(1)) * 60 + int(match.group(2))


    def _build_map_tab(self, parent: ctk.CTkFrame) -> None:
        """
        Baut die Kartenansicht für die zeitabhängige Nachfrage- und Flottenvisualisierung auf.

        Die Methode erstellt die komplette GUI des Map-Tabs mit Datums- und
        Linienauswahl, Stunden- und Geschwindigkeitsslider, Play-Steuerung,
        Legende, Kartenwidget sowie einer Live-Rangliste der Haltestellen. Die
        Ansicht kombiniert Nachfrageentwicklung pro Stunde mit einer animierten
        Darstellung der aktuell fahrenden Busse auf der Karte.

        Parameter:
            parent (ctk.CTkFrame): Container, in den die Kartenansicht eingebettet
                wird.

        Rückgabewerte:
            None: Die Methode erzeugt und positioniert GUI-Komponenten.

        Fehler/Sonderfälle:
            Initiale Zustände wie Fokus auf Würzburg, deaktiviertes Mausrad-Zoomen
            und erste Flottenaktualisierung werden direkt nach dem Aufbau gesetzt, um
            eine konsistente Startansicht sicherzustellen.

        Projektkontext:
            Die Kartenansicht ergänzt die tabellarische und KPI-basierte Analyse um
            eine räumlich-zeitliche Perspektive auf Nachfrage und Fahrzeugbewegungen.
        """
        map_card = self._card(parent, row=0, title="Karten-Radar", subtitle="Stündliche Nachfrageentwicklung je Haltestelle")
        toolbar = ctk.CTkFrame(map_card, fg_color="transparent")
        toolbar.pack(fill="x", padx=14, pady=(0, 10))
        self.map_status_label = ctk.CTkLabel(
            toolbar,
            text="Würzburg-Fokus aktiv. Wähle einen Tag und starte die Animation.",
            text_color=PALETTE["muted"],
            font=ctk.CTkFont(family="Segoe UI", size=13),
        )
        self.map_status_label.pack(side="left")
        ctk.CTkButton(
            toolbar,
            text="Zentrieren",
            width=120,
            height=36,
            corner_radius=12,
            fg_color=PALETTE["accent_soft"],
            hover_color=PALETTE["accent"],
            text_color=PALETTE["text"],
            command=self._focus_wuerzburg_map,
        ).pack(side="right")


        controls = ctk.CTkFrame(map_card, fg_color=PALETTE["surface_alt"], corner_radius=22)
        controls.pack(fill="x", padx=14, pady=(0, 12))
        controls.grid_columnconfigure(0, weight=0)
        controls.grid_columnconfigure(1, weight=0)
        controls.grid_columnconfigure(2, weight=1)
        controls.grid_columnconfigure(3, weight=0)
        controls.grid_columnconfigure(4, weight=0)


        day_box = ctk.CTkFrame(controls, fg_color="transparent")
        day_box.grid(row=0, column=0, sticky="w", padx=14, pady=12)
        ctk.CTkLabel(
            day_box,
            text="Animationstag",
            text_color=PALETTE["muted"],
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        ).pack(anchor="w", pady=(0, 4))
        initial_day = self.start_picker.get_date()
        self.map_day_entry = DateEntry(
            day_box,
            date_pattern="yyyy-mm-dd",
            year=initial_day.year,
            month=initial_day.month,
            day=initial_day.day,
            **dark_date_entry_options(),
        )
        self.map_day_entry.pack(fill="x")
        self.map_day_entry.bind("<<DateEntrySelected>>", self._on_map_day_selected)


        line_box = ctk.CTkFrame(controls, fg_color="transparent")
        line_box.grid(row=0, column=1, sticky="ew", padx=(4, 12), pady=12)
        ctk.CTkLabel(
            line_box,
            text="Karten-Linie",
            text_color=PALETTE["muted"],
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        ).pack(anchor="w", pady=(0, 4))
        self.map_line_menu = ctk.CTkOptionMenu(
            line_box,
            values=["Alle aktiven Linien"],
            variable=self.map_line_choice,
            width=260,
            height=36,
            corner_radius=14,
            fg_color=PALETTE["surface"],
            button_color=PALETTE["accent"],
            button_hover_color=PALETTE["accent_dark"],
            dropdown_fg_color=PALETTE["surface"],
            dropdown_text_color=PALETTE["text"],
            dropdown_hover_color=PALETTE["accent_soft"],
            text_color="white",
            command=self._on_map_line_selected,
        )
        self.map_line_menu.pack(fill="x")
        self._refresh_map_line_menu()
        self._style_dark_dropdown_border(self.map_line_menu)


        hour_box = ctk.CTkFrame(controls, fg_color="transparent")
        hour_box.grid(row=0, column=2, sticky="ew", padx=18, pady=12)
        self.map_hour_label = ctk.CTkLabel(
            hour_box,
            text=f"{self.map_current_hour:02d}:00 Uhr",
            text_color=PALETTE["text"],
            font=ctk.CTkFont(family="Bahnschrift", size=22, weight="bold"),
        )
        self.map_hour_label.pack(anchor="w")
        self.map_hour_slider = ctk.CTkSlider(
            hour_box,
            from_=0,
            to=23,
            number_of_steps=23 * 60,
            progress_color=PALETTE["accent"],
            button_color=PALETTE["accent"],
            button_hover_color=PALETTE["accent_dark"],
            command=self._on_map_hour_slider,
        )
        self.map_hour_slider.pack(fill="x", pady=(8, 0))
        self.map_hour_slider.set(self.map_current_hour)


        speed_box = ctk.CTkFrame(controls, fg_color="transparent")
        speed_box.grid(row=0, column=3, sticky="ew", padx=(4, 10), pady=12)
        speed_header = ctk.CTkFrame(speed_box, fg_color="transparent")
        speed_header.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(
            speed_header,
            text="Tempo",
            text_color=PALETTE["muted"],
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        ).pack(side="left")
        self.map_speed_label = ctk.CTkLabel(
            speed_header,
            text="1.0x",
            text_color=PALETTE["text"],
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        )
        self.map_speed_label.pack(side="right")
        self.map_speed_slider = ctk.CTkSlider(
            speed_box,
            from_=0.25,
            to=3.0,
            number_of_steps=11,
            width=170,
            progress_color=PALETTE["teal"],
            button_color=PALETTE["teal"],
            button_hover_color="#0f766e",
            command=self._on_map_speed_slider,
        )
        self.map_speed_slider.pack(fill="x", pady=(8, 0))
        self.map_speed_slider.set(self.map_speed_factor)


        self.map_play_button = ctk.CTkButton(
            controls,
            text="Play",
            width=110,
            height=42,
            corner_radius=16,
            fg_color=PALETTE["accent"],
            hover_color=PALETTE["accent_dark"],
            command=self._toggle_map_animation,
        )
        self.map_play_button.grid(row=0, column=4, sticky="e", padx=14, pady=12)


        fleet_note = ctk.CTkFrame(map_card, fg_color=PALETTE["surface_alt"], corner_radius=20)
        fleet_note.pack(fill="x", padx=14, pady=(0, 12))
        ctk.CTkLabel(
            fleet_note,
            text=(
                "Live-Buslayer: Beim Abspielen werden alle aktuell fahrenden Busse "
                "der aktiven Linien als Symbole auf echten Straßenrouten angezeigt."
            ),
            text_color=PALETTE["muted"],
            font=ctk.CTkFont(family="Segoe UI", size=13),
            anchor="w",
        ).pack(fill="x", padx=16, pady=12)


        legend = ctk.CTkFrame(map_card, fg_color="transparent")
        legend.pack(fill="x", padx=14, pady=(0, 10))
        for label, color in [
            ("↑ links: Einstiege", "#22d3ee"),
            ("↓ rechts: Ausstiege", "#34d399"),
            ("Event aktiv", "#fbbf24"),
            ("Konzert aktiv", "#f472b6"),
            ("Balkenhöhe = Personen pro Stunde", "#334155"),
        ]:
            item = ctk.CTkFrame(legend, fg_color="transparent")
            item.pack(side="left", padx=(0, 18))
            ctk.CTkLabel(item, text="", width=16, height=16, fg_color=color, corner_radius=8).pack(side="left", padx=(0, 6))
            ctk.CTkLabel(
                item,
                text=label,
                text_color=PALETTE["muted"],
                font=ctk.CTkFont(family="Segoe UI", size=12),
            ).pack(side="left")


        map_body = ctk.CTkFrame(map_card, fg_color="transparent")
        map_body.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        map_body.grid_columnconfigure(0, weight=4)
        map_body.grid_columnconfigure(1, weight=1)
        map_body.grid_rowconfigure(0, weight=1)


        self.map_widget = tkintermapview.TkinterMapView(map_body, height=680, corner_radius=0)
        self.map_widget.grid(row=0, column=0, sticky="nsew", padx=(0, 12))


        ranking_card = ctk.CTkFrame(
            map_body,
            fg_color=PALETTE["surface_alt"],
            corner_radius=22,
            border_width=1,
            border_color=PALETTE["border"],
        )
        ranking_card.grid(row=0, column=1, sticky="nsew")
        ctk.CTkLabel(
            ranking_card,
            text="Live-Nachfrage",
            text_color=PALETTE["text"],
            font=ctk.CTkFont(family="Bahnschrift", size=22, weight="bold"),
        ).pack(anchor="w", padx=14, pady=(14, 2))
        ctk.CTkLabel(
            ranking_card,
            text="Haltestellen sortiert nach Einstiegen",
            text_color=PALETTE["muted"],
            font=ctk.CTkFont(family="Segoe UI", size=12),
        ).pack(anchor="w", padx=14, pady=(0, 10))
        self.map_station_textbox = ctk.CTkTextbox(
            ranking_card,
            fg_color=PALETTE["surface"],
            corner_radius=16,
            border_width=0,
            text_color=PALETTE["text"],
            font=ctk.CTkFont(family="Consolas", size=12),
        )
        self.map_station_textbox.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self._disable_map_mouse_wheel()
        self._focus_wuerzburg_map()
        self._refresh_fleet_schedule()


    def _refresh_map_line_menu(self) -> None:
        """
        Aktualisiert die Linienauswahl für die Kartenansicht.

        Die Methode baut ein Lookup zwischen sichtbaren Linienlabels und numerischen
        Linienkennungen auf. Zusätzlich wird immer eine Option für alle aktiven
        Linien angeboten und bei Bedarf als Standardauswahl gesetzt.

        

        Rückgabewerte:
            None: Die Methode aktualisiert GUI und internen Lookup-Zustand.

        Fehler/Sonderfälle:
            Ist das Menü noch nicht initialisiert, wird lediglich die interne
            Lookup-Struktur vorbereitet.

        Projektkontext:
            Die Methode stellt sicher, dass Kartenanimation und Live-Flottenansicht
            auf einer konsistenten und fachlich sinnvollen Linienauswahl basieren.
        """
        self.map_line_lookup = {"Alle aktiven Linien": None}
        for line in self.repo.available_lines:
            self.map_line_lookup[self.repo.label_for_line(line)] = line


        values = list(self.map_line_lookup) or ["Alle aktiven Linien"]
        if self.map_line_menu is not None:
            self.map_line_menu.configure(values=values)
            self._style_dark_dropdown_border(self.map_line_menu)
        if self.map_line_choice.get() not in self.map_line_lookup:
            self.map_line_choice.set(values[0])


    def _style_dark_dropdown_border(self, menu: ctk.CTkOptionMenu | None) -> None:
        """
        Passt das interne Dropdown-Menü eines OptionMenus an das dunkle UI-Thema an.

        Die Methode greift auf das eingebettete Dropdown-Objekt zu und setzt Farben
        sowie Darstellungsattribute so, dass sich das Menü optisch in das restliche
        Dark-Theme der Anwendung einfügt.

        Parameter:
            menu (ctk.CTkOptionMenu | None): Zu stylendes Auswahlmenü.

        Rückgabewerte:
            None: Die Methode verändert ausschließlich Darstellungsattribute.

        Fehler/Sonderfälle:
            Existiert kein Menü oder kein internes Dropdown-Objekt, wird die Methode
            ohne Wirkung beendet. TclError wird tolerant behandelt, da interne
            Widget-Implementierungen je nach Plattform variieren können.

        Projektkontext:
            Einheitliches visuelles Styling trägt zur professionellen Wirkung der GUI
            bei und verbessert die Konsistenz komplexer Dashboard-Ansichten.
        """
        if menu is None:
            return
        dropdown = getattr(menu, "_dropdown_menu", None)
        if dropdown is None:
            return
        try:
            dropdown.configure(
                bg=PALETTE["surface"],
                fg=PALETTE["text"],
                activebackground=PALETTE["accent_soft"],
                activeforeground=PALETTE["text"],
                borderwidth=0,
                activeborderwidth=0,
                relief="flat",
                selectcolor=PALETTE["surface"],
            )
        except tk.TclError:
            pass


    def _selected_map_lines(self) -> list[int]:
        """
        Ermittelt die aktuell in der Kartenansicht zu berücksichtigenden Linien.

        Ist eine konkrete Linie im Kartenmenü gewählt, wird nur diese Linie
        zurückgegeben. Andernfalls werden die aktuell aktiven beziehungsweise
        ausgewählten Linien des Gesamtsystems verwendet.

        

        Rückgabewerte:
            list[int]: Liste der in der Kartenansicht relevanten Liniennummern.

        Fehler/Sonderfälle:
            Liegt keine explizite Auswahl vor, wird auf selected_lines oder
            ersatzweise auf alle verfügbaren Repository-Linien zurückgegriffen.

        Projektkontext:
            Die Methode synchronisiert Kartenfilter, Nachfrageanimation und
            Flottenvisualisierung mit dem globalen Linienkontext der Anwendung.
        """
        selected_line = self.map_line_lookup.get(self.map_line_choice.get())
        if selected_line is not None:
            return [selected_line]
        return list(self.selected_lines or self.repo.available_lines)


    def _on_map_line_selected(self, _choice: str | None = None) -> None:
        """
        Reagiert auf eine geänderte Linienauswahl in der Kartenansicht.

        Nach einer Änderung werden sowohl die Nachfrageanimationsdaten als auch der
        Fahrtenplan der Live-Flotte neu geladen, damit Karte und Nebenansichten
        konsistent bleiben.

        Parameter:
            _choice (str | None): Vom Auswahlmenü gelieferter Wert; wird fachlich
                nicht direkt benötigt.

        Rückgabewerte:
            None: Die Methode stößt Aktualisierungen des Kartenzustands an.

        Fehler/Sonderfälle:
            Der übergebene Auswahltext wird bewusst ignoriert, da die tatsächlich
            benötigten Linien über die interne Lookup-Struktur bestimmt werden.

        Projektkontext:
            Die Methode stellt die Reaktivität der Kartenansicht auf geänderte
            Filterbedingungen sicher.
        """
        self._refresh_map_animation_data()
        self._refresh_fleet_schedule()


    def _disable_map_mouse_wheel(self) -> None:
        """
        Deaktiviert Zoom-Interaktionen per Mausrad auf dem Karten-Canvas.

        Die Methode bindet die relevanten Mausereignisse so um, dass sie keine
        Standardaktion mehr auslösen. Dadurch wird unbeabsichtigtes Zoomen in der
        eingebetteten Kartenansicht verhindert.

        

        Rückgabewerte:
            None: Die Methode verändert Event-Bindings der Karte.

        Fehler/Sonderfälle:
            Ist kein Kartenwidget vorhanden, wird keine Aktion ausgeführt.

        Projektkontext:
            Die Einschränkung der Zoombedienung kann in Dashboard-Kontexten sinnvoll
            sein, um eine kontrolliertere und reproduzierbarere Nutzerführung zu
            gewährleisten.
        """
        if self.map_widget is None:
            return
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.map_widget.canvas.bind(sequence, lambda _event: "break")


    def _focus_wuerzburg_map(self) -> None:
        """
        Zentriert die Kartenansicht auf den vordefinierten Fokusraum Würzburg.

        Die Methode setzt sowohl Kartenposition als auch Zoomstufe auf die im Projekt
        definierten Standardwerte für den Untersuchungsraum.

        

        Rückgabewerte:
            None: Die Methode verändert ausschließlich den Kartenzustand.

        Fehler/Sonderfälle:
            Ist das Kartenwidget noch nicht vorhanden, wird die Methode ohne Wirkung
            beendet.

        Projektkontext:
            Der Fokus auf Würzburg verankert die Anwendung räumlich im vorgesehenen
            Analysegebiet und unterstützt eine konsistente Startperspektive.
        """
        if self.map_widget is None:
            return
        self.map_widget.set_position(*WUERZBURG_CENTER)
        self.map_widget.set_zoom(WUERZBURG_ZOOM)


    def _on_map_day_selected(self, _event=None) -> None:
        """
        Reagiert auf eine geänderte Tagesauswahl in der Kartenansicht.

        Nach Auswahl eines neuen Datums werden sowohl die Nachfrageanimation als
        auch die Flottenfahrten des gewählten Tages neu geladen.

        Parameter:
            _event: Optionales GUI-Ereignis des DateEntry-Widgets.

        Rückgabewerte:
            None: Die Methode stößt Aktualisierungen abhängiger Kartenkomponenten an.

        Fehler/Sonderfälle:
            Das Eventobjekt wird nicht ausgewertet und dient nur der Kompatibilität
            mit dem gebundenen GUI-Ereignis.

        Projektkontext:
            Die Methode verbindet die zeitliche Filterung der Kartenansicht mit den
            zugrunde liegenden Nachfrage- und Fahrtdaten.
        """
        self._refresh_map_animation_data()
        self._refresh_fleet_schedule()


    def _load_route_cache(self) -> dict[str, list[list[float]]]:
        """
        Lädt den persistenten Cache für vorberechnete Straßenrouten.

        Die Methode liest eine JSON-Datei mit zwischengespeicherten Routenverläufen
        ein und validiert grob deren Struktur, damit nur nutzbare Cacheeinträge
        zurückgegeben werden.

        

        Rückgabewerte:
            dict[str, list[list[float]]]: Mapping aus Cache-Schlüsseln auf gespeicherte
            Koordinatenfolgen.

        Fehler/Sonderfälle:
            Fehlt die Cache-Datei, ist sie nicht lesbar oder enthält kein gültiges
            JSON-Dictionary, wird ein leeres Mapping zurückgegeben.

        Projektkontext:
            Der Routen-Cache reduziert wiederholte externe oder rechenintensive
            Routenbestimmungen und verbessert damit die Reaktionszeit der
            Flottenvisualisierung.
        """
        if not ROUTE_CACHE_PATH.exists():
            return {}
        try:
            with ROUTE_CACHE_PATH.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(key): value for key, value in data.items() if isinstance(value, list)}


    def _save_route_cache(self) -> None:
        """
        Speichert den aktuellen Routen-Cache persistent auf dem Dateisystem.

        Die Methode schreibt den in-memory gehaltenen Cache nur dann zurück, wenn er
        seit dem letzten Speichern verändert wurde. Nach erfolgreichem Schreiben wird
        das Dirty-Flag zurückgesetzt.

        

        Rückgabewerte:
            None: Die Methode beeinflusst Persistenz und internen Cache-Zustand.

        Fehler/Sonderfälle:
            Ist der Cache unverändert, erfolgt kein Schreibzugriff. Dateisystemfehler
            werden tolerant behandelt, damit die Anwendung weiter nutzbar bleibt.

        Projektkontext:
            Persistenter Cache reduziert spätere Ladezeiten und verhindert unnötige
            Neuberechnung von Routen in der Kartenanimation.
        """
        if not self.map_fleet_cache_dirty:
            return
        try:
            ROUTE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with ROUTE_CACHE_PATH.open("w", encoding="utf-8") as handle:
                json.dump(self.map_fleet_route_cache, handle)
            self.map_fleet_cache_dirty = False
        except OSError:
            pass


    def _clear_fleet_markers(self) -> None:
        """
        Entfernt alle aktuell dargestellten Fahrzeugmarker von der Karte.

        Die Methode löscht sämtliche Markerobjekte der Live-Flottenanzeige und
        initialisiert die interne Markerstruktur neu.

        

        Rückgabewerte:
            None: Die Methode verändert ausschließlich den Zustand der Kartenmarker.

        Fehler/Sonderfälle:
            Fehler beim Löschen einzelner Marker werden tolerant behandelt, damit ein
            inkonsistenter Markerzustand die weitere Kartenlogik nicht blockiert.

        Projektkontext:
            Das vollständige Zurücksetzen der Marker ist notwendig, bevor neue
            Flottendaten geladen oder andere Filter angewendet werden.
        """
        for marker in self.map_fleet_markers.values():
            try:
                marker.delete()
            except Exception:
                pass
        self.map_fleet_markers = {}


    def _refresh_fleet_schedule(self) -> None:
        """
        Aktualisiert den Fahrtenplan der Live-Flottenanzeige für die Kartenansicht.

        Die Methode lädt für den gewählten Tag und die aktuell relevanten Linien alle
        verfügbaren Busfahrten, bereitet sie zu zusammenhängenden Trip-Strukturen auf
        und synchronisiert anschließend die Live-Anzeige mit der aktuellen Kartenzeit.
        Zusätzlich wird die Statusanzeige über Anzahl oder Fehlen gematchter Fahrten
        informiert.

        

        Rückgabewerte:
            None: Die Methode aktualisiert Kartenzustand, interne Trip-Strukturen und
            Statusanzeige.

        Fehler/Sonderfälle:
            Fahrten mit weniger als zwei geographisch auswertbaren Punkten werden
            verworfen. Leere Fahrplandaten führen nicht zu einem Fehler, sondern zu
            einer entsprechenden Statusmeldung.

        Projektkontext:
            Die Methode ist zentral für die animierte Fahrzeugdarstellung auf der
            Karte und verbindet Linienfilter, Tageswahl und operative Fahrtenlage.
        """
        self._clear_fleet_markers()
        self.map_fleet_route_failures = set()
        selected_day = self.map_day_entry.get_date() if self.map_day_entry is not None else self.start_picker.get_date()
        self.map_fleet_schedule = self.repo.fleet_schedule_for_day(self._selected_map_lines(), selected_day)
        self.map_fleet_trips = []


        if not self.map_fleet_schedule.empty:
            for (line, journey), trip in self.map_fleet_schedule.groupby(["line", "journey"], sort=False):
                trip = (
                    trip.sort_values("minute_of_day")
                    .drop_duplicates(subset=["minute_of_day", "lat", "lon"], keep="first")
                    .reset_index(drop=True)
                )
                if len(trip) < 2:
                    continue
                stops = [
                    {
                        "minute": float(row["minute_of_day"]),
                        "lat": float(row["lat"]),
                        "lon": float(row["lon"]),
                        "station": str(row["station"]),
                    }
                    for _, row in trip.iterrows()
                ]
                self.map_fleet_trips.append(
                    {
                        "id": f"{int(line)}-{journey}",
                        "line": int(line),
                        "journey": str(journey),
                        "start": stops[0]["minute"],
                        "end": stops[-1]["minute"],
                        "stops": stops,
                    }
                )


        self._sync_fleet_to_minute(float(self.map_current_hour * 60), allow_route_fetch=False)
        if self.map_status_label is not None:
            if self.map_fleet_trips:
                self.map_status_label.configure(
                    text=(
                        f"{selected_day.isoformat()} | {len(self.map_fleet_trips)} Fahrten geladen. "
                        "Play oder Slider zeigt die aktuell fahrenden Busse."
                    )
                )
            else:
                self.map_status_label.configure(text=f"{selected_day.isoformat()} | Keine gematchten Busfahrten gefunden.")

    def _route_leg_key(self, current: dict[str, object], nxt: dict[str, object]) -> str:
        """
        Erzeugt einen stabilen Cache-Schlüssel für einen Streckenabschnitt zwischen zwei Halten.

        Die Methode verdichtet die geographischen Koordinaten eines Start- und
        Zielpunkts in eine normierte Zeichenkette. Durch die feste Rundung werden
        identische Streckenabschnitte zuverlässig wiedererkannt, auch wenn die
        Quelldaten geringfügige numerische Schwankungen enthalten.

        Parameter:
            current (dict[str, object]): Startpunkt des Abschnitts mit mindestens
                Latitude- und Longitude-Werten.
            nxt (dict[str, object]): Endpunkt des Abschnitts mit mindestens
                Latitude- und Longitude-Werten.

        Rückgabewerte:
            str: Eindeutiger Schlüssel zur Identifikation eines Streckenabschnitts.

        Fehler/Sonderfälle:
            Die Methode setzt voraus, dass in beiden Dictionaries die Schlüssel
            "lat" und "lon" vorhanden und in float umwandelbar sind.

        Projektkontext:
            Der Schlüssel dient als Grundlage für das Routing-Cache der
            Flottenvisualisierung und vermeidet wiederholte externe Routenabfragen
            für bereits bekannte Abschnittspaare.
        """
        return (
            f"{float(current['lat']):.5f},{float(current['lon']):.5f}>"
            f"{float(nxt['lat']):.5f},{float(nxt['lon']):.5f}"
        )


    def _fetch_osrm_leg_points(self, current: dict[str, object], nxt: dict[str, object]) -> list[list[float]]:
        """
        Lädt Straßenroutenpunkte für einen Streckenabschnitt über den OSRM-Routingdienst.

        Die Methode formuliert aus zwei Haltepunkten eine OSRM-Anfrage und ruft eine
        Straßenroute im GeoJSON-Format ab. Anschließend werden die gelieferten
        Koordinaten in das im Projekt verwendete Format [lat, lon] überführt. Das
        urllib-Modul unterstützt dabei Timeouts für blockierende Netzwerkoperationen,
        was hier genutzt wird, um die GUI nicht unnötig lange zu verzögern. [web:101]

        Parameter:
            current (dict[str, object]): Startpunkt des Abschnitts mit geographischen
                Koordinaten.
            nxt (dict[str, object]): Zielpunkt des Abschnitts mit geographischen
                Koordinaten.

        Rückgabewerte:
            list[list[float]]: Liste von Routenpunkten als [lat, lon]-Paare; leer,
            wenn keine nutzbare Route geliefert werden konnte.

        Fehler/Sonderfälle:
            Netzwerkfehler, Timeouts, ungültige Antworten oder fehlende Routen führen
            nicht zu einer Ausnahme nach außen, sondern zu einer leeren Ergebnisliste.
            Ebenso werden Routen mit weniger als zwei Punkten verworfen.

        Projektkontext:
            Die Methode ermöglicht eine realitätsnähere Kartenanimation entlang
            echter Straßenverläufe statt bloßer gerader Linien zwischen Haltestellen.
        """
        query = (
            f"{float(current['lon']):.6f},{float(current['lat']):.6f};"
            f"{float(nxt['lon']):.6f},{float(nxt['lat']):.6f}"
        )
        url = (
            "https://router.project-osrm.org/route/v1/driving/"
            + urllib.parse.quote(query, safe=",;")
            + "?overview=full&geometries=geojson&alternatives=false&steps=false"
        )
        try:
            with urllib.request.urlopen(url, timeout=1.2) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            return []


        routes = payload.get("routes") or []
        if not routes:
            return []
        coordinates = ((routes[0].get("geometry") or {}).get("coordinates")) or []
        points = [[float(lat), float(lon)] for lon, lat in coordinates if lon is not None and lat is not None]
        return points if len(points) >= 2 else []


    def _street_leg_points(
        self,
        current: dict[str, object],
        nxt: dict[str, object],
        allow_route_fetch: bool,
    ) -> list[list[float]]:
        """
        Liefert Straßenroutenpunkte für einen Streckenabschnitt unter Nutzung von Cache und Budget.

        Die Methode prüft zunächst, ob für den Abschnitt bereits Routenpunkte im
        Cache vorliegen oder ein bekannter Fehlschlag existiert. Nur wenn externe
        Routenabfragen erlaubt sind und das aktuelle Anfragebudget noch nicht
        aufgebraucht ist, wird eine neue OSRM-Abfrage ausgelöst. Erfolgreiche
        Ergebnisse werden persistent zwischengespeichert.

        Parameter:
            current (dict[str, object]): Startpunkt des Abschnitts.
            nxt (dict[str, object]): Zielpunkt des Abschnitts.
            allow_route_fetch (bool): Gibt an, ob externe Routenabfragen in diesem
                Aufruf zulässig sind.

        Rückgabewerte:
            list[list[float]]: Straßenroutenpunkte des Abschnitts oder eine leere
            Liste, falls keine Route verfügbar ist.

        Fehler/Sonderfälle:
            Bereits bekannte Fehlschläge werden nicht erneut abgefragt. Ist kein
            Anfragebudget mehr vorhanden oder sind externe Abfragen deaktiviert,
            wird unmittelbar eine leere Liste zurückgegeben.

        Projektkontext:
            Die Methode ist zentral für eine performante und robuste Busanimation,
            da sie externe Routingkosten begrenzt und vorhandene Ergebnisse
            systematisch wiederverwendet.
        """
        key = self._route_leg_key(current, nxt)
        cached = self.map_fleet_route_cache.get(key)
        if cached:
            return cached
        if key in self.map_fleet_route_failures:
            return []
        if not allow_route_fetch or getattr(self, "_route_fetch_budget", 0) <= 0:
            return []


        self._route_fetch_budget -= 1
        points = self._fetch_osrm_leg_points(current, nxt)
        if points:
            self.map_fleet_route_cache[key] = points
            self.map_fleet_cache_dirty = True
            self._save_route_cache()
        else:
            self.map_fleet_route_failures.add(key)
        return points


    def _point_along_polyline(self, points: list[list[float]], share: float) -> tuple[float, float]:
        """
        Berechnet einen interpolierten Punkt entlang einer Polylinie.

        Die Methode interpretiert die übergebenen Punkte als zusammenhängende Linie,
        bestimmt deren kumulierte Segmentlängen und liefert den Punkt, der dem
        angegebenen Anteil an der Gesamtlänge entspricht. Dadurch bewegt sich ein
        animierter Marker proportional entlang des tatsächlichen Linienverlaufs.

        Parameter:
            points (list[list[float]]): Polylinie als Liste von [lat, lon]-Punkten.
            share (float): Relativer Fortschritt auf der Linie im Intervall von 0 bis 1.

        Rückgabewerte:
            tuple[float, float]: Interpolierte Position als Latitude- und
            Longitude-Wert.

        Fehler/Sonderfälle:
            Bei leerer Punktliste wird (0.0, 0.0) zurückgegeben. Besteht die Linie
            nur aus einem Punkt oder besitzt Gesamtlänge 0, wird dieser Punkt
            direkt geliefert. Der Anteil wird auf den Bereich 0 bis 1 begrenzt.

        Projektkontext:
            Die Methode ist ein Kernbaustein der Buspositionsanimation auf der Karte,
            da sie die kontinuierliche Bewegung zwischen diskreten Haltepunkten
            ermöglicht.
        """
        if not points:
            return 0.0, 0.0
        if len(points) == 1:
            return float(points[0][0]), float(points[0][1])
        share = min(max(share, 0.0), 1.0)
        segment_lengths = []
        total_length = 0.0
        for index in range(len(points) - 1):
            lat_a, lon_a = points[index]
            lat_b, lon_b = points[index + 1]
            segment_length = ((float(lat_b) - float(lat_a)) ** 2 + (float(lon_b) - float(lon_a)) ** 2) ** 0.5
            segment_lengths.append(segment_length)
            total_length += segment_length
        if total_length == 0:
            return float(points[0][0]), float(points[0][1])


        target = total_length * share
        walked = 0.0
        for index, segment_length in enumerate(segment_lengths):
            if walked + segment_length >= target:
                local_share = 0.0 if segment_length == 0 else (target - walked) / segment_length
                lat_a, lon_a = points[index]
                lat_b, lon_b = points[index + 1]
                lat = float(lat_a) + (float(lat_b) - float(lat_a)) * local_share
                lon = float(lon_a) + (float(lon_b) - float(lon_a)) * local_share
                return lat, lon
            walked += segment_length
        return float(points[-1][0]), float(points[-1][1])


    def _fleet_position_at_minute(
        self,
        trip: dict[str, object],
        minute: float,
        allow_route_fetch: bool,
    ) -> tuple[float, float, dict[str, object], dict[str, object], float] | None:
        """
        Ermittelt die interpolierte Position einer Fahrt zu einer gegebenen Minute.

        Die Methode durchsucht die aufeinanderfolgenden Haltestellenabschnitte einer
        Fahrt nach demjenigen Segment, in das die angefragte Minute fällt. Für dieses
        Segment wird der relative Fortschritt berechnet und anhand der Straßenroute
        eine konkrete Fahrzeugposition bestimmt.

        Parameter:
            trip (dict[str, object]): Beschreibung einer Fahrt mit Stoppliste und
                Zeitintervallen.
            minute (float): Zielminute seit Tagesbeginn.
            allow_route_fetch (bool): Gibt an, ob bei fehlenden Cache-Daten neue
                Straßenrouten geladen werden dürfen.

        Rückgabewerte:
            tuple[float, float, dict[str, object], dict[str, object], float] | None:
            Latitude, Longitude, aktueller Halt, nächster Halt und relativer
            Segmentfortschritt oder None, falls keine Position bestimmbar ist.

        Fehler/Sonderfälle:
            Enthält eine Fahrt weniger als zwei Stopps oder keine nutzbare
            Straßenroute für das relevante Segment, wird None zurückgegeben.

        Projektkontext:
            Die Methode verbindet Fahrplanzeiten mit geographischer Bewegung und ist
            damit essenziell für die Live-Darstellung fahrender Busse.
        """
        stops = trip["stops"]
        if not isinstance(stops, list) or len(stops) < 2:
            return None
        for index in range(len(stops) - 1):
            current = stops[index]
            nxt = stops[index + 1]
            start = float(current["minute"])
            end = float(nxt["minute"])
            if start <= minute <= end:
                share = 0.0 if end == start else (minute - start) / (end - start)
                points = self._street_leg_points(current, nxt, allow_route_fetch)
                if not points:
                    return None
                lat, lon = self._point_along_polyline(points, share)
                return lat, lon, current, nxt, share
        return None


    def _sync_fleet_to_minute(self, minute: float, allow_route_fetch: bool = False) -> tuple[int, int]:
        """
        Synchronisiert alle Flottenmarker mit einer gegebenen Kartenminute.

        Die Methode bestimmt für jede aktuell relevante Fahrt, ob sie zur
        angegebenen Minute aktiv ist, berechnet gegebenenfalls ihre Position und
        aktualisiert oder erzeugt den zugehörigen Marker. Nicht mehr aktive Marker
        werden entfernt. Zusätzlich wird gezählt, wie viele Fahrten sichtbar
        dargestellt werden konnten und wie viele wegen fehlender Routendaten
        übersprungen wurden.

        Parameter:
            minute (float): Zielminute seit Tagesbeginn, auf die die Flottenanzeige
                synchronisiert werden soll.
            allow_route_fetch (bool): Gibt an, ob fehlende Straßenrouten während
                dieses Durchlaufs extern nachgeladen werden dürfen.

        Rückgabewerte:
            tuple[int, int]: Anzahl sichtbarer Fahrten und Anzahl übersprungener
            Fahrten.

        Fehler/Sonderfälle:
            Ist kein Kartenwidget vorhanden, wird (0, 0) zurückgegeben. Fehlende oder
            nicht auflösbare Routen einzelner Fahrten führen nur zum Überspringen
            dieser Fahrten, nicht zum Abbruch der Gesamtsynchronisation.

        Projektkontext:
            Diese Methode ist der zentrale Aktualisierungsmechanismus der
            Live-Flottenansicht und koppelt Zeitsteuerung, Routing und Markerzustand
            der Karte.
        """
        if self.map_widget is None:
            return 0, 0
        if self.map_fleet_icon is None:
            self.map_fleet_icon = self._make_bus_icon()


        self._route_fetch_budget = 6 if allow_route_fetch else 0
        active_ids = set()
        visible = 0
        skipped = 0
        for trip in self.map_fleet_trips:
            if minute < float(trip["start"]) or minute > float(trip["end"]):
                continue
            position = self._fleet_position_at_minute(trip, minute, allow_route_fetch)
            if position is None:
                skipped += 1
                continue


            lat, lon, current, nxt, share = position
            marker_id = str(trip["id"])
            active_ids.add(marker_id)
            marker_data = {
                "line": trip["line"],
                "journey": trip["journey"],
                "from": current["station"],
                "to": nxt["station"],
                "share": share,
            }
            if marker_id in self.map_fleet_markers:
                marker = self.map_fleet_markers[marker_id]
                marker.set_position(lat, lon)
                marker.set_text(None)
                marker.data = marker_data
            else:
                marker = self.map_widget.set_marker(
                    lat,
                    lon,
                    text=None,
                    icon=self.map_fleet_icon,
                    icon_anchor="center",
                    data=marker_data,
                )
                self.map_fleet_markers[marker_id] = marker
            visible += 1


        for marker_id in list(self.map_fleet_markers):
            if marker_id not in active_ids:
                try:
                    self.map_fleet_markers[marker_id].delete()
                except Exception:
                    pass
                self.map_fleet_markers.pop(marker_id, None)
        return visible, skipped


    def _refresh_bus_lines(self) -> None:
        """
        Aktualisiert die Linienauswahl für die Einzelbus-Animation.

        Die Methode erzeugt aus den aktuell ausgewählten oder verfügbaren Linien
        eine GUI-taugliche Auswahlstruktur und setzt bei Bedarf einen
        Standardwert. Anschließend wird die Liste der verfügbaren Fahrten für die
        gewählte Linie neu geladen.

        

        Rückgabewerte:
            None: Die Methode aktualisiert internen Zustand und GUI-Menüs.

        Fehler/Sonderfälle:
            Sind keine Linien verfügbar, wird ein Platzhalterwert gesetzt. Nicht
            initialisierte GUI-Menüs werden toleriert.

        Projektkontext:
            Die Methode bildet den Einstieg in die fokussierte Animation einer
            einzelnen Busfahrt innerhalb der Kartenansicht.
        """
        self.map_bus_line_lookup = {}
        values = []
        candidate_lines = self.selected_lines or self.repo.available_lines
        for line in candidate_lines:
            label = self.repo.label_for_line(line)
            values.append(label)
            self.map_bus_line_lookup[label] = line
        if not values:
            values = ["Keine Linie"]
        if self.map_bus_line_menu is not None:
            self.map_bus_line_menu.configure(values=values)
        if self.map_bus_line_choice.get() not in self.map_bus_line_lookup:
            self.map_bus_line_choice.set(values[0])
        self._refresh_bus_journeys()


    def _selected_bus_line(self) -> int | None:
        """
        Liefert die aktuell für die Busanimation ausgewählte Linie.

        Die Methode übersetzt den sichtbaren Auswahlwert des Linienmenüs in die
        zugehörige numerische Linienkennung.

        

        Rückgabewerte:
            int | None: Gewählte Liniennummer oder None, falls keine gültige Auswahl
            vorliegt.

        Fehler/Sonderfälle:
            Ungültige oder nicht aufgelöste Auswahlwerte führen zu None.

        Projektkontext:
            Die Methode verbindet die GUI-Auswahl der Einzelbusansicht mit den
            linienbezogenen Repository-Abfragen.
        """
        return self.map_bus_line_lookup.get(self.map_bus_line_choice.get())


    def _refresh_bus_journeys(self) -> None:
        """
        Aktualisiert die verfügbaren Fahrten für die aktuell gewählte Linie und den gewählten Tag.

        Vor dem Neuladen werden laufende Animationen gestoppt und vorhandene
        Routenelemente entfernt. Anschließend werden die für Linie und Tag
        verfügbaren Fahrten aus dem Repository gelesen, in eine Auswahlliste
        überführt und die aktuell ausgewählte Route neu geladen.

        

        Rückgabewerte:
            None: Die Methode aktualisiert Menüs, internen Zustand und gegebenenfalls
            die geladene Busroute.

        Fehler/Sonderfälle:
            Ist keine gültige Linie gewählt oder existieren keine Fahrten, wird ein
            Platzhalterwert gesetzt. Die Routenladung erfolgt in stillem Modus, um
            unnötige Statusmeldungen zu vermeiden.

        Projektkontext:
            Die Methode stellt sicher, dass die Busanimation stets mit der aktuellen
            Linien- und Tagesauswahl synchron bleibt.
        """
        self._pause_bus_animation()
        self._clear_bus_route()
        selected_line = self._selected_bus_line()
        selected_day = self.map_day_entry.get_date() if self.map_day_entry is not None else self.start_picker.get_date()
        self.map_bus_journey_lookup = {}
        if selected_line is None:
            values = ["Keine Fahrt"]
        else:
            journeys = self.repo.journey_options_for_day(selected_line, selected_day)
            values = journeys["label"].tolist() if not journeys.empty else ["Keine Fahrt"]
            self.map_bus_journey_lookup = {
                str(row["label"]): row["journey"]
                for _, row in journeys.iterrows()
            }
        if self.map_bus_journey_menu is not None:
            self.map_bus_journey_menu.configure(values=values)
        if self.map_bus_journey_choice.get() not in self.map_bus_journey_lookup:
            self.map_bus_journey_choice.set(values[0])
        self._load_selected_bus_route(silent=True)


    def _make_bus_icon(self) -> tk.PhotoImage:
        """
        Erzeugt oder lädt das Kartensymbol für Busmarker.

        Zunächst wird versucht, ein vorbereitetes Icon aus einer Datei zu laden und
        bei Bedarf auf eine geeignete Größe zu verkleinern. Falls dies nicht gelingt,
        wird programmatisch ein einfaches Bussymbol als PhotoImage erzeugt. Tkinter
        PhotoImage ist für die Anzeige solcher Icons in Widgets und auf Canvas-
        ähnlichen Oberflächen vorgesehen. [web:110][web:114]

        

        Rückgabewerte:
            tk.PhotoImage: Verwendbares Bussymbol für Kartenmarker.

        Fehler/Sonderfälle:
            Fehler beim Laden der externen Bilddatei führen nicht zum Abbruch,
            sondern zu einer intern erzeugten Fallback-Grafik.

        Projektkontext:
            Das Bussymbol ist die visuelle Grundlage für Live-Flottenmarker und die
            Animation einzelner Fahrten auf der Karte.
        """
        if BUS_ICON_PATH.exists():
            try:
                image = tk.PhotoImage(file=str(BUS_ICON_PATH))
                shrink = max(1, int(max(image.width() / 46, image.height() / 32)))
                if shrink > 1:
                    image = image.subsample(shrink, shrink)
                return image
            except (tk.TclError, OSError):
                pass


        icon = tk.PhotoImage(width=42, height=28)
        body = PALETTE["accent"]
        border = "#7f1d1d"
        window = "#dbeafe"
        tire = "#0f172a"
        icon.put(border, to=(5, 6, 37, 22))
        icon.put(body, to=(7, 8, 35, 20))
        icon.put(window, to=(11, 10, 18, 15))
        icon.put(window, to=(21, 10, 28, 15))
        icon.put("#ffffff", to=(31, 10, 33, 15))
        icon.put(tire, to=(10, 20, 16, 26))
        icon.put(tire, to=(27, 20, 33, 26))
        icon.put("#f8fafc", to=(12, 22, 14, 24))
        icon.put("#f8fafc", to=(29, 22, 31, 24))
        return icon


    def _clear_bus_route(self) -> None:
        """
        Entfernt alle aktuell dargestellten Elemente der Einzelbus-Route.

        Die Methode setzt die intern gehaltenen Routendaten zurück und löscht Marker,
        Pfad und Haltestellenmarkierungen der aktuell geladenen Busfahrt.

        

        Rückgabewerte:
            None: Die Methode bereinigt ausschließlich Karten- und Zustandsobjekte.

        Fehler/Sonderfälle:
            Fehler beim Löschen einzelner Marker oder Pfade werden tolerant
            behandelt, damit nachfolgende Ladevorgänge nicht blockiert werden.

        Projektkontext:
            Die Methode ist wichtig für saubere Zustandswechsel beim Wechsel von
            Linie, Fahrt oder Datum in der Einzelbus-Animation.
        """
        self.map_bus_route_data = pd.DataFrame()
        if self.map_bus_marker is not None:
            try:
                self.map_bus_marker.delete()
            except Exception:
                pass
            self.map_bus_marker = None
        if self.map_bus_path is not None:
            try:
                self.map_bus_path.delete()
            except Exception:
                pass
            self.map_bus_path = None
        for marker in self.map_bus_stop_markers:
            try:
                marker.delete()
            except Exception:
                pass
        self.map_bus_stop_markers = []


    def _load_selected_bus_route(self, _choice: str | None = None, silent: bool = False) -> None:
        """
        Lädt die aktuell ausgewählte Busfahrt und bereitet sie für die Kartenanimation auf.

        Die Methode ermittelt die gewählte Linie, Fahrt und den Tag, lädt die
        zugehörige Route aus dem Repository und stellt sie als Kartenpfad,
        Fahrzeugmarker und ausgewählte Haltestellenmarker dar. Anschließend wird die
        Startposition gesetzt und die Kartenansicht auf die erste Minute der Fahrt
        synchronisiert.

        Parameter:
            _choice (str | None): Optionaler Auswahlwert des GUI-Menüs; wird nicht
                direkt fachlich verwendet.
            silent (bool): Unterdrückt Statusmeldungen, wenn True.

        Rückgabewerte:
            None: Die Methode aktualisiert Kartenobjekte und internen Animationszustand.

        Fehler/Sonderfälle:
            Fehlt eine gültige Linie oder Fahrt, wird die Methode beendet. Routen mit
            weniger als zwei Punkten werden verworfen, da keine sinnvolle Animation
            möglich ist. Bei silent=True werden Hinweismeldungen nicht angezeigt.

        Projektkontext:
            Diese Methode bereitet die fokussierte Darstellung einer konkreten Fahrt
            vor und verbindet Fahrtdaten, Kartenpfad und Busanimation.
        """
        self._pause_bus_animation()
        self._clear_bus_route()
        if self.map_widget is None:
            return
        selected_line = self._selected_bus_line()
        selected_journey = self.map_bus_journey_lookup.get(self.map_bus_journey_choice.get())
        selected_day = self.map_day_entry.get_date() if self.map_day_entry is not None else self.start_picker.get_date()
        if selected_line is None or selected_journey is None:
            if not silent and self.map_status_label is not None:
                self.map_status_label.configure(text="Keine Fahrt für die Busanimation ausgewählt.")
            return
        route = self.repo.journey_route_for_day(selected_line, selected_day, selected_journey)
        if len(route) < 2:
            if not silent and self.map_status_label is not None:
                self.map_status_label.configure(text="Für diese Fahrt fehlen ausreichend gematchte Haltestellen.")
            return
        self.map_bus_route_data = route
        points = [(float(row["lat"]), float(row["lon"])) for _, row in route.iterrows()]
        try:
            self.map_bus_path = self.map_widget.set_path(points, color=PALETTE["amber"], width=4)
        except TypeError:
            self.map_bus_path = self.map_widget.set_path(points)
        self.map_bus_icon = self._make_bus_icon()
        first = route.iloc[0]
        self.map_bus_marker = self.map_widget.set_marker(
            float(first["lat"]),
            float(first["lon"]),
            text="Bus",
            icon=self.map_bus_icon,
            icon_anchor="center",
        )
        for _, row in route.iloc[:: max(1, len(route) // 8)].iterrows():
            marker = self.map_widget.set_marker(
                float(row["lat"]),
                float(row["lon"]),
                text=f"{int(row['stop_sequence'])}. {str(row['station'])[:14]}",
            )
            self.map_bus_stop_markers.append(marker)
        self.map_bus_current_minute = float(route.iloc[0]["minute_of_day"])
        self._sync_bus_to_minute(self.map_bus_current_minute, update_map_time=False)
        if not silent and self.map_status_label is not None:
            start_time = pd.Timestamp(first["departure_plan_station"]).strftime("%H:%M")
            last = route.iloc[-1]
            end_time = pd.Timestamp(last["departure_plan_station"]).strftime("%H:%M")
            self.map_status_label.configure(
                text=f"Busroute geladen: Linie {selected_line}, Fahrt {selected_journey}, {start_time}-{end_time}, {len(route)} Stops."
            )


    def _toggle_bus_animation(self) -> None:
        """
        Startet oder pausiert die Animation einer ausgewählten Einzelbusfahrt.

        Ist bereits eine Busanimation aktiv, wird sie angehalten. Andernfalls wird
        bei Bedarf zunächst die ausgewählte Route geladen, eine eventuell laufende
        Kartenanimation pausiert und die Busanimation ab der ersten Fahrtminute
        gestartet.

        

        Rückgabewerte:
            None: Die Methode steuert ausschließlich den Animationszustand.

        Fehler/Sonderfälle:
            Kann keine Route geladen werden oder bleibt die Routentabelle leer, wird
            keine Animation gestartet.

        Projektkontext:
            Die Methode ist die zentrale Benutzerinteraktion für die gezielte
            Wiedergabe einer einzelnen Busfahrt auf der Karte.
        """
        if self.map_bus_animation_running:
            self._pause_bus_animation()
            return
        if self.map_bus_route_data.empty:
            self._load_selected_bus_route()
        if self.map_bus_route_data.empty:
            return
        self._pause_map_animation()
        self.map_bus_animation_running = True
        self.map_bus_current_minute = float(self.map_bus_route_data.iloc[0]["minute_of_day"])
        if self.map_bus_play_button is not None:
            self.map_bus_play_button.configure(text="Bus Pause")
        self._animate_bus_step()


    def _pause_bus_animation(self, update_button: bool = True) -> None:
        """
        Pausiert die laufende Einzelbus-Animation und bereinigt ihren Timerzustand.

        Die Methode stoppt die Wiederholungslogik der Animation, entfernt einen
        gegebenenfalls registrierten after-Callback und setzt optional die
        Beschriftung des Play-Buttons zurück.

        Parameter:
            update_button (bool): Gibt an, ob der Buttontext ebenfalls aktualisiert
                werden soll.

        Rückgabewerte:
            None: Die Methode verändert ausschließlich Animations- und GUI-Zustände.

        Fehler/Sonderfälle:
            Fehler beim Abbrechen eines gespeicherten after-Callbacks werden tolerant
            behandelt.

        Projektkontext:
            Die Methode ist notwendig, um Animationen kontrolliert zu stoppen, etwa
            bei Auswahlwechseln, Pausen oder am Fahrtende.
        """
        self.map_bus_animation_running = False
        if self.map_bus_animation_after_id is not None:
            try:
                self.after_cancel(self.map_bus_animation_after_id)
            except Exception:
                pass
            self.map_bus_animation_after_id = None
        if update_button and self.map_bus_play_button is not None:
            self.map_bus_play_button.configure(text="Bus abspielen")


    def _bus_position_at_minute(self, minute: float) -> tuple[float, float, pd.Series, pd.Series, float]:
        """
        Berechnet die Position eines einzelnen Busses zu einer gegebenen Minute.

        Die Methode durchsucht die geladene Route nach dem passenden Zeitsegment und
        interpoliert die Fahrzeugposition linear zwischen zwei aufeinanderfolgenden
        Haltepunkten. Liegt die Minute vor dem Start oder nach dem Ende der Route,
        wird die Position des ersten beziehungsweise letzten Punkts zurückgegeben.

        Parameter:
            minute (float): Zielminute seit Tagesbeginn.

        Rückgabewerte:
            tuple[float, float, pd.Series, pd.Series, float]: Latitude, Longitude,
            aktueller Haltepunkt, nächster Haltepunkt und relativer Fortschritt auf
            dem Segment.

        Fehler/Sonderfälle:
            Minuten außerhalb des Routenumfangs werden auf Start- oder Endpunkt
            abgebildet. Die Route wird vor der Berechnung nach minute_of_day sortiert.

        Projektkontext:
            Die Methode bildet die geometrische Grundlage für die fokussierte
            Animation einer einzelnen Busfahrt.
        """
        route = self.map_bus_route_data.sort_values("minute_of_day")
        if minute <= float(route.iloc[0]["minute_of_day"]):
            row = route.iloc[0]
            return float(row["lat"]), float(row["lon"]), row, row, 0.0
        if minute >= float(route.iloc[-1]["minute_of_day"]):
            row = route.iloc[-1]
            return float(row["lat"]), float(row["lon"]), row, row, 1.0
        for index in range(len(route) - 1):
            current = route.iloc[index]
            nxt = route.iloc[index + 1]
            start = float(current["minute_of_day"])
            end = float(nxt["minute_of_day"])
            if start <= minute <= end:
                share = 0.0 if end == start else (minute - start) / (end - start)
                lat = float(current["lat"]) + (float(nxt["lat"]) - float(current["lat"])) * share
                lon = float(current["lon"]) + (float(nxt["lon"]) - float(current["lon"])) * share
                return lat, lon, current, nxt, share
        row = route.iloc[-1]
        return float(row["lat"]), float(row["lon"]), row, row, 1.0


    def _sync_bus_to_minute(self, minute: float, update_map_time: bool = True) -> None:
        """
        Synchronisiert die Einzelbus-Animation mit einer gegebenen Minute.

        Die Methode setzt den Busmarker auf die zur Minute passende Position,
        aktualisiert optional die allgemeine Kartenzeitanzeige und schreibt eine
        Statusmeldung mit aktuellem Streckenabschnitt und Fortschritt.

        Parameter:
            minute (float): Zielminute seit Tagesbeginn.
            update_map_time (bool): Gibt an, ob zusätzlich die allgemeine
                Kartenzeitdarstellung interpoliert werden soll.

        Rückgabewerte:
            None: Die Methode aktualisiert Marker, Kartenansicht und Statuslabel.

        Fehler/Sonderfälle:
            Ist kein Busmarker vorhanden oder keine Route geladen, wird die Methode
            ohne Wirkung beendet.

        Projektkontext:
            Die Methode koppelt die zeitliche Wiedergabe einer Einzelbusfahrt mit der
            restlichen Kartenlogik und macht den Fahrtfortschritt für Nutzer sichtbar.
        """
        if self.map_bus_marker is None or self.map_bus_route_data.empty:
            return
        lat, lon, current, nxt, share = self._bus_position_at_minute(minute)
        self.map_bus_marker.set_position(lat, lon)
        display = f"{int(minute // 60) % 24:02d}:{int(minute % 60):02d} Uhr"
        if update_map_time:
            from_hour = min(int(minute // 60), 23)
            to_hour = min(from_hour + 1, 23)
            progress = minute / 60 - from_hour
            self._render_map_interpolated_frame(from_hour, to_hour, progress)
        if self.map_status_label is not None:
            self.map_status_label.configure(
                text=(
                    f"Bus {display}: {str(current['station'])} → {str(nxt['station'])} "
                    f"({share * 100:.0f}% der Teilstrecke)"
                )
            )


    def _animate_bus_step(self) -> None:
        """
        Führt einen Animationsschritt der Einzelbusfahrt aus.

        Die Methode aktualisiert die Busposition für die aktuelle Minute, prüft das
        Fahrtende und plant andernfalls den nächsten Schritt per Tkinter-after. Die
        Animation schreitet dabei in Minutenschritten entlang der Route fort.

        

        Rückgabewerte:
            None: Die Methode steuert ausschließlich den Animationsablauf.

        Fehler/Sonderfälle:
            Ist keine Animation aktiv oder keine Route geladen, wird kein weiterer
            Schritt ausgeführt. Am Fahrtende wird die Animation automatisch pausiert.

        Projektkontext:
            Die Methode implementiert den zeitlichen Takt der Einzelbusanimation in
            der Kartenansicht.
        """
        if not self.map_bus_animation_running or self.map_bus_route_data.empty:
            return
        route = self.map_bus_route_data
        end_minute = float(route.iloc[-1]["minute_of_day"])
        self._sync_bus_to_minute(self.map_bus_current_minute)
        if self.map_bus_current_minute >= end_minute:
            self._pause_bus_animation()
            return
        self.map_bus_current_minute = min(self.map_bus_current_minute + 1.0, end_minute)
        self.map_bus_animation_after_id = self.after(140, self._animate_bus_step)


    def _map_marker_colors(self, boardings: float, max_boardings: float) -> tuple[str, str]:
        """
        Bestimmt Füll- und Randfarbe eines Nachfragemarkers anhand der relativen Einstiegshöhe.

        Die Methode setzt Schwellenwerte auf Basis des Verhältnisses zwischen
        aktuellem Einstiegswert und maximalem Referenzwert und ordnet daraus eine
        diskrete Farbklasse ab.

        Parameter:
            boardings (float): Einstiege der betrachteten Haltestelle.
            max_boardings (float): Referenzwert für maximale Einstiege im aktuellen
                Kontext.

        Rückgabewerte:
            tuple[str, str]: Füllfarbe und Randfarbe als Hex-Farbwerte.

        Fehler/Sonderfälle:
            Um Divisionen durch 0 zu vermeiden, wird mindestens 1.0 als Nenner
            verwendet.

        Projektkontext:
            Die Methode unterstützt die visuelle Hervorhebung stark nachgefragter
            Haltestellen in der Kartenansicht.
        """
        share = boardings / max(max_boardings, 1.0)
        if share >= 0.66:
            return "#dc2626", "#7f1d1d"
        if share >= 0.33:
            return "#f59e0b", "#92400e"
        return "#1f7cff", "#1e3a8a"


    def _blend_hex(self, start: str, end: str, amount: float) -> str:
        """
        Interpoliert zwischen zwei Hex-Farben.

        Die Methode mischt zwei RGB-Farben linear in Abhängigkeit eines normierten
        Anteils und liefert die resultierende Farbe erneut als Hex-String zurück.

        Parameter:
            start (str): Startfarbe im Format #RRGGBB.
            end (str): Zielfarbe im Format #RRGGBB.
            amount (float): Mischanteil zwischen 0 und 1.

        Rückgabewerte:
            str: Gemischte Farbe als Hex-String.

        Fehler/Sonderfälle:
            Der Mischanteil wird auf den Bereich 0 bis 1 begrenzt. Die Methode setzt
            korrekt formatierte Hex-Farbstrings voraus.

        Projektkontext:
            Die Farbmischung wird für weiche Farbverläufe in Nachfrage- und
            Eventsymbolen verwendet und trägt zur abgestuften visuellen Codierung bei.
        """
        amount = min(max(amount, 0.0), 1.0)
        start_rgb = tuple(int(start[index : index + 2], 16) for index in (1, 3, 5))
        end_rgb = tuple(int(end[index : index + 2], 16) for index in (1, 3, 5))
        mixed = tuple(round(a + (b - a) * amount) for a, b in zip(start_rgb, end_rgb))
        return "#{:02x}{:02x}{:02x}".format(*mixed)


    def _bar_color_for_share(self, share: float) -> str:
        """
        Ermittelt die Farbskala für Einstiegsbalken anhand eines normierten Anteils.

        Kleine Werte werden in helleren Tönen, größere Werte in kräftigeren Blau-
        bis Cyan-Tönen dargestellt. Dadurch entsteht eine kontinuierliche visuelle
        Kodierung der Nachfrageintensität.

        Parameter:
            share (float): Normierter Nachfrageanteil zwischen 0 und 1.

        Rückgabewerte:
            str: Hex-Farbwert für den Balken.

        Fehler/Sonderfälle:
            Der Anteil wird auf 0 bis 1 begrenzt. Für den Nullfall wird eine neutrale
            Farbe verwendet.

        Projektkontext:
            Die Methode dient der ästhetisch abgestuften Darstellung von
            Einstiegsvolumina in der Kartenvisualisierung.
        """
        share = min(max(share, 0.0), 1.0)
        if share == 0:
            return "#94a3b8"
        if share <= 0.5:
            return self._blend_hex("#a5f3fc", "#22d3ee", share / 0.5)
        return self._blend_hex("#22d3ee", "#2563eb", (share - 0.5) / 0.5)


    def _exit_color_for_share(self, share: float) -> str:
        """
        Ermittelt die Farbskala für Ausstiegsbalken anhand eines normierten Anteils.

        Die Methode nutzt eine grünbasierte Farbskala, um Ausstiege visuell von
        Einstiegen zu unterscheiden und gleichzeitig deren Intensität abzubilden.

        Parameter:
            share (float): Normierter Anteil zwischen 0 und 1.

        Rückgabewerte:
            str: Hex-Farbwert für den Ausstiegsbalken.

        Fehler/Sonderfälle:
            Der Anteil wird auf den Bereich 0 bis 1 begrenzt. Für den Nullfall wird
            eine neutrale Farbe verwendet.

        Projektkontext:
            Die Methode unterstützt die semantische Trennung von Ein- und Ausstiegen
            in der kartographischen Nachfragecodierung.
        """
        share = min(max(share, 0.0), 1.0)
        if share == 0:
            return "#94a3b8"
        if share <= 0.5:
            return self._blend_hex("#bbf7d0", "#34d399", share / 0.5)
        return self._blend_hex("#34d399", "#047857", (share - 0.5) / 0.5)


    def _boarding_gradient_for_share(self, share: float) -> tuple[str, str, str]:
        """
        Liefert einen dreistufigen Farbverlauf für Einstiegsbalken.

        Abhängig von der Nachfrageintensität wird ein heller, mittlerer und dunkler
        Farbton zurückgegeben, aus denen später ein vertikaler Gradient erzeugt wird.

        Parameter:
            share (float): Normierter Nachfrageanteil zwischen 0 und 1.

        Rückgabewerte:
            tuple[str, str, str]: Drei Hex-Farben für oberen, mittleren und unteren
            Farbverlauf.

        Fehler/Sonderfälle:
            Der Anteil wird auf den Bereich 0 bis 1 begrenzt.

        Projektkontext:
            Die Methode verbessert die visuelle Lesbarkeit und Wertigkeit der
            generierten Nachfrage-Icons.
        """
        share = min(max(share, 0.0), 1.0)
        if share >= 0.72:
            return "#a5f3fc", "#38bdf8", "#4f46e5"
        if share >= 0.36:
            return "#cffafe", "#22d3ee", "#2563eb"
        return "#ecfeff", "#67e8f9", "#0284c7"


    def _exit_gradient_for_share(self, share: float) -> tuple[str, str, str]:
        """
        Liefert einen dreistufigen Farbverlauf für Ausstiegsbalken.

        Die Methode erzeugt abhängig von der Intensität einen grünbasierten
        Verlauf für die spätere Icon-Erzeugung.

        Parameter:
            share (float): Normierter Anteil zwischen 0 und 1.

        Rückgabewerte:
            tuple[str, str, str]: Drei Hex-Farben für den Farbverlauf des
            Ausstiegsbalkens.

        Fehler/Sonderfälle:
            Der Anteil wird auf den Bereich 0 bis 1 begrenzt.

        Projektkontext:
            Der Verlauf ermöglicht eine klar unterscheidbare und dennoch konsistente
            Darstellung von Ausstiegsintensitäten auf der Karte.
        """
        share = min(max(share, 0.0), 1.0)
        if share >= 0.72:
            return "#ccfbf1", "#10b981", "#0f766e"
        if share >= 0.36:
            return "#dcfce7", "#34d399", "#059669"
        return "#f0fdf4", "#86efac", "#16a34a"


    def _map_bar_share(self, value: float) -> float:
        """
        Transformiert einen absoluten Nachfragewert in einen normierten Balkenanteil.

        Die Methode bildet kleine und mittlere Werte annähernd linear auf eine
        Balkenhöhe ab und komprimiert hohe Werte im oberen Bereich mit einer
        Potenzfunktion. Dadurch bleiben Unterschiede bei kleinen Werten sichtbar,
        ohne dass große Werte die Skala vollständig dominieren.

        Parameter:
            value (float): Absoluter Nachfragewert, etwa Einstiege oder Ausstiege.

        Rückgabewerte:
            float: Normierter Anteil zwischen 0 und 1 für die Balkendarstellung.

        Fehler/Sonderfälle:
            Negative Werte werden auf 0 begrenzt. Die Methode nutzt projektweit
            definierte Referenz- und Maximalwerte für die Skalierung.

        Projektkontext:
            Diese Transformation ist zentral für eine visuell ausgewogene Darstellung
            von Personenströmen in der Kartenansicht.
        """
        value = max(float(value), 0.0)
        if value <= 0:
            return 0.0
        if value <= MAP_BAR_REFERENCE_FLOW:
            return 0.45 * (value / MAP_BAR_REFERENCE_FLOW)
        extra_range = max(MAP_BAR_MAX_FLOW - MAP_BAR_REFERENCE_FLOW, 1.0)
        extra_share = min((value - MAP_BAR_REFERENCE_FLOW) / extra_range, 1.0)
        return min(1.0, 0.45 + 0.55 * (extra_share ** 0.7))


    def _make_bar_icon(self, boardings: float, exiting: float, max_flow: float = 1.0) -> tk.PhotoImage:
        """
        Erzeugt ein Balken-Icon zur kombinierten Darstellung von Einstiegen und Ausstiegen.

        Die Methode erstellt ein kleines Bitmap-Icon, das links einen Einstiegsbalken
        und rechts einen Ausstiegsbalken mit Richtungssymbolen darstellt. Die Höhe
        der Balken wird aus normierten Nachfragewerten abgeleitet und mit Farbverläufen
        visualisiert.

        Parameter:
            boardings (float): Anzahl der Einstiege.
            exiting (float): Anzahl der Ausstiege.
            max_flow (float): Reservierter Parameter für einen potenziellen
                Referenzbezug; in der aktuellen Implementierung wird die Skalierung
                über projektweite Konstanten vorgenommen.

        Rückgabewerte:
            tk.PhotoImage: Generiertes Icon für Kartenmarker.

        Fehler/Sonderfälle:
            Balken mit Wert 0 werden nicht gezeichnet. Sehr kleine positive Werte
            erhalten dennoch eine Mindesthöhe, damit Nachfrage sichtbar bleibt.

        Projektkontext:
            Das Icon ist die visuelle Kernkomponente für die stündliche
            Nachfrageentwicklung pro Haltestelle in der Kartenansicht.
        """
        boarding_share = self._map_bar_share(boardings)
        exiting_share = self._map_bar_share(exiting)
        boarding_height = max(4, int(4 + boarding_share * 42)) if boardings > 0 else 0
        exiting_height = max(4, int(4 + exiting_share * 42)) if exiting > 0 else 0
        width = 34
        symbol_band = 18
        bottom_pad = 5
        max_bar_height = max(boarding_height, exiting_height, 8)
        height = symbol_band + max_bar_height + bottom_pad
        icon = tk.PhotoImage(width=width, height=height)


        def draw_bar(x1: int, x2: int, bar_height: int, gradient: tuple[str, str, str]) -> None:
            """
            Zeichnet einen einzelnen vertikalen Balken mit Farbverlauf in das Icon.

            Parameter:
                x1 (int): Linke x-Koordinate des Balkens.
                x2 (int): Rechte x-Koordinate des Balkens.
                bar_height (int): Höhe des Balkens in Pixeln.
                gradient (tuple[str, str, str]): Drei Farbstufen für den Verlauf.

            Rückgabewerte:
                None: Die Funktion zeichnet direkt in das PhotoImage.

            Fehler/Sonderfälle:
                Balken mit Höhe kleiner oder gleich 0 werden nicht gezeichnet.

            Projektkontext:
                Die lokale Hilfsfunktion kapselt die wiederverwendbare Logik zur
                Darstellung von Ein- und Ausstiegsbalken.
            """
            if bar_height <= 0:
                return
            y1 = symbol_band + (max_bar_height - bar_height)
            y2 = symbol_band + max_bar_height
            top_color, mid_color, bottom_color = gradient
            icon.put("#334155", to=(x1 - 2, y1 - 2, x2 + 2, y2 + 2))
            span = max(y2 - y1 - 1, 1)
            for y in range(y1, y2):
                amount = (y - y1) / span
                if amount <= 0.55:
                    row_color = self._blend_hex(top_color, mid_color, amount / 0.55)
                else:
                    row_color = self._blend_hex(mid_color, bottom_color, (amount - 0.55) / 0.45)
                icon.put(row_color, to=(x1, y, x2, y + 1))


        def draw_up_symbol(center_x: int, top_y: int, color: str) -> None:
            """
            Zeichnet ein Aufwärtssymbol für Einstiege oberhalb eines Balkens.

            Parameter:
                center_x (int): Horizontale Mittelposition des Symbols.
                top_y (int): Obere y-Koordinate des Symbols.
                color (str): Symbolfarbe als Hex-Wert.

            Rückgabewerte:
                None: Die Funktion zeichnet direkt in das PhotoImage.

            Fehler/Sonderfälle:
                Keine spezielle Fehlerbehandlung; gültige Koordinaten werden
                vorausgesetzt.

            Projektkontext:
                Das Symbol unterstützt die semantische Unterscheidung der linken
                Balkenseite als Einstiegsdarstellung.
            """
            for row, symbol_width in enumerate((1, 3, 5, 7)):
                half = symbol_width // 2
                icon.put(color, to=(center_x - half, top_y + row, center_x + half + 1, top_y + row + 1))
            icon.put(color, to=(center_x - 1, top_y + 4, center_x + 2, top_y + 12))


        def draw_down_symbol(center_x: int, top_y: int, color: str) -> None:
            """
            Zeichnet ein Abwärtssymbol für Ausstiege oberhalb eines Balkens.

            Parameter:
                center_x (int): Horizontale Mittelposition des Symbols.
                top_y (int): Obere y-Koordinate des Symbols.
                color (str): Symbolfarbe als Hex-Wert.

            Rückgabewerte:
                None: Die Funktion zeichnet direkt in das PhotoImage.

            Fehler/Sonderfälle:
                Keine spezielle Fehlerbehandlung; gültige Koordinaten werden
                vorausgesetzt.

            Projektkontext:
                Das Symbol unterstützt die semantische Unterscheidung der rechten
                Balkenseite als Ausstiegsdarstellung.
            """
            icon.put(color, to=(center_x - 1, top_y, center_x + 2, top_y + 8))
            for row, symbol_width in enumerate((7, 5, 3, 1)):
                half = symbol_width // 2
                y = top_y + 8 + row
                icon.put(color, to=(center_x - half, y, center_x + half + 1, y + 1))


        draw_bar(7, 14, boarding_height, self._boarding_gradient_for_share(boarding_share))
        draw_bar(20, 27, exiting_height, self._exit_gradient_for_share(exiting_share))
        if boarding_height > 0:
            draw_up_symbol(10, 3, "#0891b2")
        if exiting_height > 0:
            draw_down_symbol(24, 3, "#047857")
        return icon


    def _cached_bar_icon(self, boardings: float, exiting: float, max_flow: float) -> tk.PhotoImage:
        """
        Liefert ein zwischengespeichertes Balken-Icon für ähnliche Nachfragewerte.

        Die Methode quantisiert Ein- und Ausstiegswerte in diskrete Buckets und
        verwendet diese als Cache-Schlüssel. Dadurch müssen optisch ähnliche Icons
        nicht bei jedem Frame neu erzeugt werden.

        Parameter:
            boardings (float): Anzahl der Einstiege.
            exiting (float): Anzahl der Ausstiege.
            max_flow (float): Reservierter Referenzparameter; die aktuelle
                Bucket-Bildung nutzt projektweite Konstanten.

        Rückgabewerte:
            tk.PhotoImage: Passendes, gegebenenfalls neu generiertes Balken-Icon.

        Fehler/Sonderfälle:
            Die Werte werden auf gültige Bucket-Grenzen begrenzt, bevor ein
            Cache-Schlüssel erzeugt wird.

        Projektkontext:
            Icon-Caching ist wichtig für die Performance der Kartenanimation, da viele
            Marker über mehrere Zeitschritte hinweg ähnliche Ausprägungen besitzen.
        """
        boarding_bucket = min(30, max(0, int(round(self._map_bar_share(boardings) * 30))))
        exiting_bucket = min(30, max(0, int(round(self._map_bar_share(exiting) * 30))))
        key = (boarding_bucket, exiting_bucket)
        if key not in self.map_bar_icon_cache:
            bucket_to_value = MAP_BAR_MAX_FLOW / 30.0
            self.map_bar_icon_cache[key] = self._make_bar_icon(
                float(boarding_bucket) * bucket_to_value,
                float(exiting_bucket) * bucket_to_value,
            )
        return self.map_bar_icon_cache[key]


    def _event_duration_bucket(self, duration_hours: float) -> int:
        """
        Ordnet eine Eventdauer einer diskreten Größenklasse zu.

        Die Methode reduziert kontinuierliche Veranstaltungsdauern auf drei Buckets,
        die später für unterschiedlich große Eventmarker verwendet werden.

        Parameter:
            duration_hours (float): Veranstaltungsdauer in Stunden.

        Rückgabewerte:
            int: Dauerklasse 0, 1 oder 2.

        Fehler/Sonderfälle:
            Kurze oder negative Werte landen implizit in Bucket 0.

        Projektkontext:
            Die Diskretisierung vereinfacht die visuelle Kodierung von
            Veranstaltungsintensität in der Kartenansicht.
        """
        if duration_hours >= 8:
            return 2
        if duration_hours >= 4:
            return 1
        return 0


    def _make_event_icon(self, concert: bool, duration_bucket: int) -> tk.PhotoImage:
        """
        Erzeugt ein Event-Icon für aktive Veranstaltungen auf der Karte.

        Die Methode zeichnet ein kreisförmiges Symbol, dessen Farbe zwischen Konzert-
        und allgemeinem Eventtyp unterscheidet und dessen Größe von einer zuvor
        bestimmten Dauerklasse abhängt.

        Parameter:
            concert (bool): Kennzeichnet, ob es sich um ein Konzert handelt.
            duration_bucket (int): Diskrete Größenklasse der Veranstaltungsdauer.

        Rückgabewerte:
            tk.PhotoImage: Generiertes Event-Icon.

        Fehler/Sonderfälle:
            Die Größe wird aus dem Bucket abgeleitet; ungewöhnliche Bucketwerte wirken
            sich direkt auf die Symbolgröße aus.

        Projektkontext:
            Eventmarker ergänzen die Nachfragekarte um externe Einflussfaktoren, die
            potenzielle Nachfragespitzen fachlich erklären können.
        """
        size = 22 + duration_bucket * 5
        icon = tk.PhotoImage(width=size, height=size)
        center = size // 2
        radius = max(7, size // 2 - 3)
        fill = "#f472b6" if concert else "#fbbf24"
        mid = "#db2777" if concert else "#f97316"
        border = "#111827"


        for y in range(size):
            distance_y = abs(y - center)
            half_width = radius - distance_y
            if half_width < 0:
                continue
            x1 = max(0, center - half_width)
            x2 = min(size, center + half_width + 1)
            icon.put(border, to=(x1, y, x2, y + 1))


        inner_radius = radius - 2
        for y in range(size):
            distance_y = abs(y - center)
            half_width = inner_radius - distance_y
            if half_width < 0:
                continue
            amount = 0.0 if inner_radius == 0 else distance_y / inner_radius
            row_color = self._blend_hex(fill, mid, amount)
            x1 = max(0, center - half_width)
            x2 = min(size, center + half_width + 1)
            icon.put(row_color, to=(x1, y, x2, y + 1))


        dot = "#fff7ed" if not concert else "#fdf2f8"
        icon.put(dot, to=(center - 2, center - 2, center + 3, center + 3))
        return icon


    def _event_icon(self, concert: bool, duration_hours: float) -> tk.PhotoImage:
        """
        Liefert ein zwischengespeichertes Event-Icon für Typ und Dauer einer Veranstaltung.

        Die Methode übersetzt die Veranstaltungsdauer zunächst in eine Dauerklasse
        und verwendet anschließend Typ und Bucket als Cache-Schlüssel für das
        passende Symbol.

        Parameter:
            concert (bool): Kennzeichnet, ob es sich um ein Konzert handelt.
            duration_hours (float): Dauer der Veranstaltung in Stunden.

        Rückgabewerte:
            tk.PhotoImage: Passendes Event-Icon.

        Fehler/Sonderfälle:
            Existiert für die Kombination noch kein Icon im Cache, wird es neu erzeugt.

        Projektkontext:
            Das Icon-Caching reduziert Rechenaufwand bei wiederholter Darstellung
            aktiver Veranstaltungen über viele Kartenframes hinweg.
        """
        bucket = self._event_duration_bucket(duration_hours)
        key = (bool(concert), bucket)
        if key not in self.map_event_icon_cache:
            self.map_event_icon_cache[key] = self._make_event_icon(bool(concert), bucket)
        return self.map_event_icon_cache[key]


    def _clear_event_markers(self) -> None:
        """
        Entfernt alle aktuell sichtbaren Eventmarker von der Karte.

        Die Methode löscht sämtliche Markerobjekte aktiver Veranstaltungen und setzt
        die interne Markerstruktur zurück.

        

        Rückgabewerte:
            None: Die Methode verändert ausschließlich den Zustand der Eventmarker.

        Fehler/Sonderfälle:
            Fehler beim Löschen einzelner Marker werden tolerant behandelt.

        Projektkontext:
            Das vollständige Zurücksetzen der Eventmarker ist notwendig, bevor eine
            neue zeitabhängige Eventauswahl dargestellt wird.
        """
        for marker in self.map_event_markers.values():
            try:
                marker.delete()
            except Exception:
                pass
        self.map_event_markers = {}


    def _active_events_for_minute(self, selected_day: date, current_minute: float) -> pd.DataFrame:
        """
        Ermittelt die zum gegebenen Zeitpunkt aktiven Veranstaltungen.

        Die Methode konstruiert aus Datum und Minute einen Zeitstempel, filtert die
        georeferenzierten Eventdaten auf aktive Einträge und sortiert diese nach
        fachlicher Priorität. Pandas `sort_values` dient dabei zum Sortieren nach
        einer oder mehreren Spalten, was hier für Konzertstatus, Dauer und Namen
        genutzt wird. [web:102]

        Parameter:
            selected_day (date): Betrachteter Kalendertag.
            current_minute (float): Minute seit Tagesbeginn.

        Rückgabewerte:
            pd.DataFrame: DataFrame mit den zu diesem Zeitpunkt aktiven
            Veranstaltungen, begrenzt auf die fachlich wichtigsten Einträge.

        Fehler/Sonderfälle:
            Sind keine Eventdaten vorhanden oder ist zum angefragten Zeitpunkt keine
            Veranstaltung aktiv, wird ein leerer DataFrame zurückgegeben. Fehlende
            Dauern werden für die Sortierung mit 0 ersetzt.

        Projektkontext:
            Die Methode verknüpft externe Ereignisse mit der Nachfragevisualisierung
            und unterstützt dadurch die Interpretation zeitlich auffälliger Muster.
        """
        if self.repo.event_geo.empty:
            return pd.DataFrame()
        day_start = pd.Timestamp(datetime.combine(selected_day, datetime.min.time()))
        current_ts = day_start + pd.to_timedelta(float(current_minute), unit="m")
        events = self.repo.event_geo[
            (self.repo.event_geo["start"] <= current_ts)
            & (self.repo.event_geo["end"] >= current_ts)
        ].copy()
        if events.empty:
            return events
        events["sort_duration"] = events["duration_hours"].fillna(0)
        return events.sort_values(["concert", "sort_duration", "name"], ascending=[False, False, True]).head(30)

    def _render_event_markers(self, selected_day: date, current_minute: float, display_time: str) -> int:
        """
        Rendert alle zum aktuellen Zeitpunkt aktiven Eventmarker auf der Karte.

        Die Methode ermittelt zunächst die aktiven Veranstaltungen für den gegebenen
        Zeitpunkt, aktualisiert bestehende Marker oder erzeugt neue Marker und
        entfernt anschließend Marker inaktiver Events. Dabei werden Eventtyp,
        Dauerklasse und Anzeigezeit in den Markerdaten hinterlegt, sodass bei
        Interaktionen fachlich aussagekräftige Details verfügbar sind.

        Parameter:
            selected_day (date): Betrachteter Kalendertag der Kartenansicht.
            current_minute (float): Aktuelle Minute seit Tagesbeginn.
            display_time (str): Formatierte Uhrzeit zur Anzeige in Marker-Details.

        Rückgabewerte:
            int: Anzahl aktuell aktiver und dargestellter Events.

        Fehler/Sonderfälle:
            Ist kein Kartenwidget verfügbar, wird 0 zurückgegeben. Fehler beim
            Löschen alter Marker werden tolerant behandelt, damit das Rendering
            nicht abbricht.

        Projektkontext:
            Die Methode ergänzt die Nachfrage- und Busvisualisierung um zeitlich
            aktive externe Einflussfaktoren, etwa Konzerte oder andere Events, die
            Nachfrageveränderungen fachlich mit erklären können.
        """
        if self.map_widget is None:
            return 0


        events = self._active_events_for_minute(selected_day, current_minute)
        active_ids: set[str] = set()
        for _, row in events.iterrows():
            event_id = str(row["event_id"])
            active_ids.add(event_id)
            concert = bool(int(row.get("concert", 0)))
            duration_hours = float(row.get("duration_hours", 0.0))
            icon_key = (concert, self._event_duration_bucket(duration_hours))
            icon = self._event_icon(concert, duration_hours)
            marker_data = {
                "name": str(row["name"]),
                "start": row["start"],
                "end": row["end"],
                "concert": concert,
                "duration_hours": duration_hours,
                "time": display_time,
                "icon_key": icon_key,
            }
            if event_id in self.map_event_markers:
                marker = self.map_event_markers[event_id]
                marker.set_position(float(row["lat"]), float(row["lon"]))
                old_data = getattr(marker, "data", None) or {}
                if old_data.get("icon_key") != icon_key:
                    marker.change_icon(icon)
                marker.set_text(None)
                marker.data = marker_data
            else:
                marker = self.map_widget.set_marker(
                    float(row["lat"]),
                    float(row["lon"]),
                    text=None,
                    icon=icon,
                    icon_anchor="s",
                    command=self._show_event_marker_details,
                    data=marker_data,
                )
                self.map_event_markers[event_id] = marker


        for event_id in list(self.map_event_markers):
            if event_id not in active_ids:
                try:
                    self.map_event_markers[event_id].delete()
                except Exception:
                    pass
                self.map_event_markers.pop(event_id, None)
        return len(active_ids)


    def _clear_map_markers(self) -> None:
        """
        Entfernt alle aktuellen Nachfrage-Marker aus der Kartenansicht.

        Die Methode löscht sämtliche Markerobjekte für Haltestellen, setzt die
        zugehörigen Verwaltungsstrukturen zurück und leert auch die Liste der
        aktuell referenzierten Balken-Icons.

        

        Rückgabewerte:
            None: Die Methode bereinigt ausschließlich Karten- und Cachezustände.

        Fehler/Sonderfälle:
            Ist kein Kartenwidget vorhanden, endet die Methode ohne Wirkung.
            Fehler beim Löschen einzelner Marker werden toleriert.

        Projektkontext:
            Die Methode ist wichtig für Zustandswechsel zwischen verschiedenen
            Stunden, Datensätzen oder leeren Kartenframes, damit keine veralteten
            Marker sichtbar bleiben.
        """
        if self.map_widget is None:
            return
        for marker in self.map_markers:
            try:
                marker.delete()
            except Exception:
                pass
        self.map_markers = []
        self.map_marker_by_station = {}
        self.map_bar_icons = []


    def _render_demand_markers(self, hour_frame: pd.DataFrame) -> None:
        """
        Rendert aggregierte Nachfrage-Marker für Haltestellen auf der Karte.

        Die Methode verdichtet eingangs stündliche oder stopnahe Bewegungsdaten auf
        Haltestellenebene, summiert Ein- und Ausstiege je Station und aktualisiert
        danach die zugehörigen Kartenmarker. Die Aggregation über `groupby(...).agg(...)`
        entspricht dem in Pandas üblichen Split-Apply-Combine-Prinzip. [web:116][web:119][web:125]

        Parameter:
            hour_frame (pd.DataFrame): Datenrahmen mit Nachfragewerten und
                geographischen Informationen für den darzustellenden Zeitpunkt.

        Rückgabewerte:
            None: Die Methode aktualisiert ausschließlich Marker auf der Karte.

        Fehler/Sonderfälle:
            Ist kein Kartenwidget vorhanden, erfolgt keine Aktion. Leere Frames
            führen zum vollständigen Entfernen vorhandener Nachfrage-Marker.
            Haltestellen ohne gültige Koordinaten werden übersprungen.

        Projektkontext:
            Die Methode bildet die zentrale visuelle Verdichtung der Nachfrage auf
            Haltestellenebene und macht räumliche Lastschwerpunkte im System sichtbar.
        """
        if self.map_widget is None:
            return
        if hour_frame.empty:
            self._clear_map_markers()
            return


        frame = hour_frame.copy()
        if "station_key" not in frame:
            frame["station_key"] = frame["station"].map(normalize_station_name)
        frame["lines"] = frame["line"].astype(str) if "line" in frame else frame.get("lines", "")
        frame = (
            frame.groupby(["station_key", "station", "lat", "lon"], as_index=False)
            .agg(
                boardings=("boardings", "sum"),
                exiting=("exiting", "sum"),
                lines=("lines", lambda values: "/".join(sorted(set(str(value) for value in values if str(value))))),
            )
            .sort_values(["boardings", "exiting"], ascending=False)
        )


        max_flow = MAP_BAR_MAX_FLOW
        active_keys: set[str] = set()


        for _, row in frame.head(40).iterrows():
            event_id = None
            if pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
                continue
            station_key = str(row["station_key"])
            active_keys.add(station_key)
            boardings = float(row["boardings"])
            exiting = float(row["exiting"])
            icon_key = (
                min(30, max(0, int(round(self._map_bar_share(boardings) * 30)))),
                min(30, max(0, int(round(self._map_bar_share(exiting) * 30)))),
            )
            icon = self._cached_bar_icon(boardings, exiting, max_flow)
            marker_data = {
                "station": row["station"],
                "lines": row["lines"],
                "boardings": boardings,
                "exiting": exiting,
                "time": self.map_hour_label.cget("text") if self.map_hour_label is not None else "",
                "icon_key": icon_key,
            }
            if station_key in self.map_marker_by_station:
                marker = self.map_marker_by_station[station_key]
                marker.set_position(float(row["lat"]), float(row["lon"]))
                old_data = getattr(marker, "data", None) or {}
                if old_data.get("icon_key") != icon_key:
                    marker.change_icon(icon)
                marker.data = marker_data
                marker.set_text(None)
            else:
                marker = self.map_widget.set_marker(
                    float(row["lat"]),
                    float(row["lon"]),
                    text=None,
                    icon=icon,
                    icon_anchor="s",
                    command=self._show_map_marker_details,
                    data=marker_data,
                )
                self.map_marker_by_station[station_key] = marker
                self.map_markers.append(marker)


        for station_key in list(self.map_marker_by_station):
            if station_key not in active_keys:
                marker = self.map_marker_by_station.pop(station_key)
                try:
                    marker.delete()
                except Exception:
                    pass
                if marker in self.map_markers:
                    self.map_markers.remove(marker)


    def _refresh_map_animation_data(self, silent: bool = False) -> None:
        """
        Lädt die Datengrundlage für die Kartenanimation eines Tages neu.

        Die Methode pausiert zunächst eine laufende Animation, lädt anschließend die
        aggregierten Haltestellen-, Stop- und Eventdaten für den ausgewählten Tag
        und rendert den aktuellen Kartenframe neu.

        Parameter:
            silent (bool): Unterdrückt Statusmeldungen, wenn True.

        Rückgabewerte:
            None: Die Methode aktualisiert interne Daten und Kartenzustand.

        Fehler/Sonderfälle:
            Ist kein Datumsauswahlfeld vorhanden, wird die Methode ohne Wirkung
            beendet.

        Projektkontext:
            Die Methode stellt sicher, dass alle animierten Kartenkomponenten auf
            einer konsistenten und tagesbezogenen Datenbasis arbeiten.
        """
        if self.map_day_entry is None:
            return
        self._pause_map_animation()
        selected_day = self.map_day_entry.get_date()
        self.map_animation_data = self.repo.aggregate_station_stop_event_map_data(self._selected_map_lines(), selected_day)
        self._render_map_hour_frame()
        if not silent and self.map_status_label is not None:
            self.map_status_label.configure(text=f"{selected_day.isoformat()} geladen. Play startet die Stundenanimation.")


    def _on_map_hour_slider(self, value: float) -> None:
        """
        Reagiert auf Änderungen des Stunden-Sliders der Kartenanimation.

        Die Methode übersetzt den Sliderwert in Startstunde, Zielstunde und
        Fortschrittsanteil innerhalb der Übergangsphase, pausiert eine laufende
        Animation und rendert den entsprechenden interpolierten Kartenframe.

        Parameter:
            value (float): Aktueller Sliderwert als Stunde mit möglichem
                Zwischenanteil.

        Rückgabewerte:
            None: Die Methode aktualisiert ausschließlich den Kartenzustand.

        Fehler/Sonderfälle:
            Interne Slider-Updates können über ein Schutzflag ignoriert werden, um
            Rückkopplungsschleifen zu vermeiden. Der Wertebereich wird auf 0 bis 23
            begrenzt.

        Projektkontext:
            Die Methode ermöglicht die manuelle zeitliche Exploration des
            Kartenverlaufs zwischen zwei Stunden.
        """
        if self.map_ignore_slider_update:
            return
        slider_value = min(max(float(value), 0.0), 23.0)
        from_hour = min(int(slider_value), 23)
        to_hour = (from_hour + 1) % 24 if from_hour < 23 else 23
        progress = slider_value - from_hour
        self.map_current_hour = from_hour
        self.map_transition_from_hour = from_hour
        self._pause_map_animation()
        self._render_map_interpolated_frame(from_hour, to_hour, progress, allow_route_fetch=True)


    def _on_map_speed_slider(self, value: float) -> None:
        """
        Setzt den Geschwindigkeitsfaktor der Kartenanimation anhand des Speed-Sliders.

        Die Methode ordnet den Rohwert des Sliders auf die nächstgelegene vordefinierte
        Geschwindigkeitsstufe ab, aktualisiert die Anzeige und synchronisiert bei
        Bedarf den Slider selbst auf genau diesen diskreten Wert.

        Parameter:
            value (float): Gewählter Sliderwert für die Animationsgeschwindigkeit.

        Rückgabewerte:
            None: Die Methode aktualisiert nur den internen Geschwindigkeitsfaktor
            und die GUI-Anzeige.

        Fehler/Sonderfälle:
            Leichte Abweichungen zwischen Rohwert und diskreter Zielstufe werden
            korrigiert, um konsistente Geschwindigkeitswerte sicherzustellen.

        Projektkontext:
            Die Methode erlaubt eine kontrollierte Anpassung der zeitlichen Dynamik
            der Kartenanimation.
        """
        speed_steps = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
        raw_value = float(value)
        self.map_speed_factor = min(speed_steps, key=lambda step: abs(step - raw_value))
        if self.map_speed_label is not None:
            self.map_speed_label.configure(text=f"{self.map_speed_factor:.2g}x")
        if self.map_speed_slider is not None and abs(self.map_speed_slider.get() - self.map_speed_factor) > 0.01:
            self.map_speed_slider.set(self.map_speed_factor)


    def _map_hour_pause_ms(self) -> int:
        """
        Berechnet die Pausenlänge zwischen zwei Stundenübergängen der Kartenanimation.

        Die Dauer wird invers zum aktuellen Geschwindigkeitsfaktor skaliert und auf
        eine minimale Pausenzeit begrenzt.

        

        Rückgabewerte:
            int: Pausenzeit in Millisekunden.

        Fehler/Sonderfälle:
            Sehr kleine Geschwindigkeitsfaktoren werden durch eine Untergrenze im
            Nenner abgesichert.

        Projektkontext:
            Die Methode steuert die wahrgenommene Verweildauer eines Stundenframes
            vor dem nächsten Übergang.
        """
        return max(120, int(MAP_HOUR_PAUSE_MS / max(self.map_speed_factor, 0.1)))


    def _map_frame_delay_ms(self) -> int:
        """
        Berechnet die Verzögerung zwischen zwei Interpolationsframes der Kartenanimation.

        Die Frame-Verzögerung wird abhängig vom aktuellen Geschwindigkeitsfaktor
        bestimmt, damit sowohl langsame als auch schnelle Animationen flüssig
        dargestellt werden können.

        

        Rückgabewerte:
            int: Frame-Verzögerung in Millisekunden.

        Fehler/Sonderfälle:
            Sehr kleine Geschwindigkeitsfaktoren werden abgefangen, um extreme
            Verzögerungen zu vermeiden.

        Projektkontext:
            Die Methode bestimmt die zeitliche Feinauflösung der animierten
            Kartenübergänge.
        """
        return max(30, int(MAP_BUS_FRAME_DELAY_MS / max(self.map_speed_factor, 0.1)))


    def _toggle_map_animation(self) -> None:
        """
        Startet oder pausiert die stündliche Kartenanimation.

        Ist die Animation bereits aktiv, wird sie gestoppt. Andernfalls werden bei
        Bedarf zunächst die Animationsdaten geladen und anschließend die Wiedergabe
        gestartet, sofern Nachfrage- oder Flottendaten vorhanden sind.

        

        Rückgabewerte:
            None: Die Methode ändert ausschließlich den Animationszustand.

        Fehler/Sonderfälle:
            Sind weder Animationsdaten noch Flottenfahrten verfügbar, erfolgt kein
            Start der Animation.

        Projektkontext:
            Diese Methode ist die zentrale Benutzeraktion zum Abspielen der
            zeitorientierten Kartenvisualisierung.
        """
        if self.map_animation_running:
            self._pause_map_animation()
            return
        if self.map_animation_data.empty:
            self._refresh_map_animation_data(silent=True)
        if self.map_animation_data.empty and not self.map_fleet_trips:
            return
        self.map_animation_running = True
        if self.map_play_button is not None:
            self.map_play_button.configure(text="Pause")
        self._schedule_map_animation_step()


    def _pause_map_animation(self, update_button: bool = True) -> None:
        """
        Pausiert die laufende Kartenanimation und entfernt geplante Folgeschritte.

        Die Methode deaktiviert den Animationszustand, bricht einen eventuell
        registrierten Tkinter-after-Callback ab und setzt optional den Text des
        Play-Buttons zurück. Tkinter `after()` dient dabei zur zeitgesteuerten
        Ausführung von GUI-Aufgaben ohne Blockierung der Oberfläche. [web:127][web:130]

        Parameter:
            update_button (bool): Gibt an, ob der Buttontext aktualisiert werden soll.

        Rückgabewerte:
            None: Die Methode verändert ausschließlich GUI- und Animationszustände.

        Fehler/Sonderfälle:
            Fehler beim Abbrechen bereits geplanter Callbacks werden tolerant
            behandelt.

        Projektkontext:
            Die Methode ist notwendig, um Benutzerinteraktionen, Datumswechsel und
            konkurrierende Animationen sauber zu koordinieren.
        """
        self.map_animation_running = False
        if self.map_animation_after_id is not None:
            try:
                self.after_cancel(self.map_animation_after_id)
            except Exception:
                pass
            self.map_animation_after_id = None
        if update_button and self.map_play_button is not None:
            self.map_play_button.configure(text="Play")


    def _schedule_map_animation_step(self) -> None:
        """
        Plant den nächsten Hauptschritt der Kartenanimation.

        Die Methode verwendet einen zeitverzögerten Tkinter-Callback, um nach einer
        stundenbezogenen Pause den nächsten Übergang anzustoßen.

        

        Rückgabewerte:
            None: Die Methode registriert lediglich einen späteren Callback.

        Fehler/Sonderfälle:
            Ist keine Animation aktiv, wird kein weiterer Schritt geplant.

        Projektkontext:
            Die Methode bildet das Bindeglied zwischen statischer Stunde und
            nachfolgendem Stundenübergang in der Kartenanimation.
        """
        if not self.map_animation_running:
            return
        self.map_animation_after_id = self.after(self._map_hour_pause_ms(), self._advance_map_animation)


    def _advance_map_animation(self) -> None:
        """
        Leitet den Übergang zur nächsten Stunde der Kartenanimation ein.

        Die Methode bestimmt die nächste Stunde zyklisch über den Tagesverlauf,
        merkt sich die aktuelle Ausgangsstunde und startet danach die eigentliche
        Interpolationsanimation.

        

        Rückgabewerte:
            None: Die Methode steuert nur den weiteren Animationsablauf.

        Fehler/Sonderfälle:
            Ist die Animation nicht aktiv, wird kein Übergang ausgelöst.

        Projektkontext:
            Die Methode sorgt für den fortlaufenden zeitlichen Fortschritt der
            Stundenanimation über den gesamten Tag hinweg.
        """
        if not self.map_animation_running:
            return
        next_hour = (self.map_current_hour + 1) % 24
        self.map_transition_from_hour = self.map_current_hour
        self._animate_map_to_hour(next_hour, step=0)


    def _animate_map_to_hour(self, next_hour: int, step: int) -> None:
        """
        Animiert den Übergang von der aktuellen Stunde zur nächsten Stunde.

        Die Methode berechnet einen normierten Fortschritt, glättet diesen mit einer
        Easing-Funktion und rendert darauf basierend den interpolierten Kartenframe.
        Solange noch nicht alle Übergangsschritte abgearbeitet sind, wird der nächste
        Frame erneut per Tkinter-after geplant.

        Parameter:
            next_hour (int): Zielstunde des aktuellen Übergangs.
            step (int): Aktueller Interpolationsschritt.

        Rückgabewerte:
            None: Die Methode aktualisiert den Kartenframe und plant Folgeschritte.

        Fehler/Sonderfälle:
            Ist die Animation nicht aktiv, wird kein weiterer Übergang ausgeführt.
            Nach dem letzten Schritt wird die neue Stunde gesetzt und die nächste
            Pausenphase eingeplant.

        Projektkontext:
            Die Methode realisiert die visuell flüssige zeitliche Interpolation
            zwischen zwei Stundenzuständen der Kartenansicht.
        """
        if not self.map_animation_running:
            return
        progress = step / MAP_TRANSITION_STEPS
        eased = progress * progress * progress * (progress * (progress * 6 - 15) + 10)
        self._render_map_interpolated_frame(self.map_transition_from_hour, next_hour, eased, allow_route_fetch=True)
        if step < MAP_TRANSITION_STEPS:
            self.map_animation_after_id = self.after(
                self._map_frame_delay_ms(),
                lambda: self._animate_map_to_hour(next_hour, step + 1),
            )
            return
        self.map_current_hour = next_hour
        self._schedule_map_animation_step()


    def _render_map_hour_frame(self) -> None:
        """
        Rendert den Kartenzustand der aktuell gesetzten Stunde ohne Übergang.

        Die Methode verwendet denselben Interpolationsmechanismus wie die Animation,
        setzt jedoch Start- und Zielstunde identisch und den Fortschritt auf 1.0.

        

        Rückgabewerte:
            None: Die Methode aktualisiert ausschließlich die Karte.

        Fehler/Sonderfälle:
            Ist kein Kartenwidget verfügbar, erfolgt keine Aktion.

        Projektkontext:
            Die Methode dient der konsistenten Darstellung eines diskreten
            Stundenstands, etwa nach Datenneuladen oder direkter Auswahl.
        """
        if self.map_widget is None:
            return
        self._render_map_interpolated_frame(self.map_current_hour, self.map_current_hour, 1.0)


    def _hour_frame_for_map(self, hour: int) -> pd.DataFrame:
        """
        Extrahiert den stundenbezogenen Kartenframe für eine bestimmte Stunde.

        Die Methode filtert die vorhandenen Animationsdaten auf gültige Koordinaten
        und die gewünschte Stunde, sortiert nach Einstiegen und reduziert doppelte
        Haltestellen auf den jeweils relevantesten Eintrag.

        Parameter:
            hour (int): Zielstunde zwischen 0 und 23.

        Rückgabewerte:
            pd.DataFrame: Nach station_key indexierter Datenrahmen für die Stunde.

        Fehler/Sonderfälle:
            Existiert keine Stunde-Spalte, wird ein leerer DataFrame zurückgegeben.

        Projektkontext:
            Die Methode stellt die Grundlage für stundenbasierte Interpolation und
            Einzelbetrachtung in der Kartenanimation bereit.
        """
        matched = self.map_animation_data.dropna(subset=["lat", "lon"]).copy()
        if "hour" not in matched:
            return pd.DataFrame()
        frame = matched[matched["hour"] == hour].copy()
        frame = frame.sort_values("boardings", ascending=False).drop_duplicates("station_key")
        return frame.set_index("station_key", drop=False)


    def _stop_event_frame_for_map(self, minute: float) -> pd.DataFrame:
        """
        Ermittelt haltestellenbezogene Nachfrageereignisse in einem Zeitfenster vor der Zielminute.

        Die Methode filtert Datensätze anhand ihres zeitlichen Abstands zur
        Zielminute, fasst sie pro Station zusammen und priorisiert anschließend
        Stationen mit dem zeitlich nächsten Halt sowie hohem Gesamtfluss.

        Parameter:
            minute (float): Zielminute seit Tagesbeginn.

        Rückgabewerte:
            pd.DataFrame: Aggregierter DataFrame mit relevanten Haltestellenereignissen
            für den betrachteten Zeitpunkt.

        Fehler/Sonderfälle:
            Fehlen Animationsdaten, die Spalte minute_of_day oder gültige Koordinaten,
            wird ein leerer DataFrame zurückgegeben. Enthält das Zeitfenster keine
            Ereignisse, bleibt das Ergebnis ebenfalls leer.

        Projektkontext:
            Die Methode koppelt die Kartenvisualisierung näher an reale Stop-Ereignisse,
            sodass Nachfragespitzen zeitlich präziser sichtbar werden als bei einer
            reinen Stundenaggregation.
        """
        if self.map_animation_data.empty or "minute_of_day" not in self.map_animation_data:
            return pd.DataFrame()
        matched = self.map_animation_data.dropna(subset=["lat", "lon"]).copy()
        if matched.empty:
            return pd.DataFrame()
        matched["time_distance"] = minute - matched["minute_of_day"].astype(float)
        frame = matched[
            (matched["time_distance"] >= 0)
            & (matched["time_distance"] <= MAP_STOP_EVENT_WINDOW_MINUTES)
        ].copy()
        if frame.empty:
            return pd.DataFrame()
        frame["lines"] = frame["line"].astype(str)
        frame = (
            frame.groupby(["station_key", "station", "lat", "lon"], as_index=False)
            .agg(
                boardings=("boardings", "sum"),
                exiting=("exiting", "sum"),
                lines=("lines", lambda values: "/".join(sorted(set(str(value) for value in values if str(value))))),
                nearest_stop=("time_distance", "min"),
            )
        )
        frame["flow_total"] = frame["boardings"] + frame["exiting"]
        return frame.sort_values(["nearest_stop", "flow_total"], ascending=[True, False])


    def _display_time_between(self, from_hour: int, to_hour: int, progress: float) -> str:
        """
        Formatiert eine interpolierte Zeit zwischen zwei Stunden als Anzeigezeichenkette.

        Die Methode berechnet zunächst die aktuelle Zwischenminute und wandelt diese
        in ein Uhrzeitformat für Labels und Statusmeldungen um.

        Parameter:
            from_hour (int): Ausgangsstunde.
            to_hour (int): Zielstunde.
            progress (float): Interpolationsfortschritt zwischen 0 und 1.

        Rückgabewerte:
            str: Formatierte Uhrzeit im Stil „HH:MM Uhr“.

        Fehler/Sonderfälle:
            Übergänge über Mitternacht werden indirekt über die Minutenberechnung
            korrekt berücksichtigt.

        Projektkontext:
            Die Methode liefert die menschenlesbare Zeitrepräsentation für die
            animierte Kartenoberfläche.
        """
        current_minutes = self._minute_between(from_hour, to_hour, progress)
        return f"{int(current_minutes // 60):02d}:{int(current_minutes % 60):02d} Uhr"


    def _minute_between(self, from_hour: int, to_hour: int, progress: float) -> float:
        """
        Berechnet die interpolierte Minute zwischen zwei Stundenpunkten.

        Die Methode transformiert Start- und Zielstunde in Minuten und interpoliert
        linear zwischen beiden Zeitpunkten. Übergänge über Mitternacht werden durch
        eine 24-Stunden-Korrektur berücksichtigt.

        Parameter:
            from_hour (int): Ausgangsstunde.
            to_hour (int): Zielstunde.
            progress (float): Interpolationsanteil zwischen 0 und 1.

        Rückgabewerte:
            float: Minute im Bereich von 0 bis unter 1440.

        Fehler/Sonderfälle:
            Fällt die Zielstunde numerisch unter die Ausgangsstunde, wird ein
            Tageswechsel angenommen.

        Projektkontext:
            Die Methode stellt die zeitliche Basis für Bus-, Nachfrage- und
            Event-Interpolation innerhalb eines Kartenübergangs bereit.
        """
        start_minutes = from_hour * 60
        end_minutes = to_hour * 60
        if to_hour < from_hour:
            end_minutes += 24 * 60
        return (start_minutes + (end_minutes - start_minutes) * progress) % (24 * 60)


    def _set_map_slider_between(self, from_hour: int, to_hour: int, progress: float) -> None:
        """
        Synchronisiert den Karten-Slider mit einer interpolierten Zwischenzeit.

        Die Methode berechnet aus Ausgangsstunde, Zielstunde und Fortschritt einen
        passenden Sliderwert und setzt diesen unter temporärer Deaktivierung der
        Slider-Callback-Reaktion.

        Parameter:
            from_hour (int): Ausgangsstunde.
            to_hour (int): Zielstunde.
            progress (float): Interpolationsfortschritt zwischen 0 und 1.

        Rückgabewerte:
            None: Die Methode aktualisiert ausschließlich das Slider-Widget.

        Fehler/Sonderfälle:
            Übergänge über Mitternacht werden speziell behandelt. Interne Slider-
            Updates aktivieren kurzzeitig ein Schutzflag gegen Rekursion.

        Projektkontext:
            Die Methode hält die GUI-Zeitsteuerung konsistent mit dem tatsächlich
            gerenderten Kartenframe.
        """
        if self.map_hour_slider is None:
            return
        target_hour = to_hour + 24 if to_hour < from_hour else to_hour
        slider_value = from_hour + (target_hour - from_hour) * progress
        if slider_value > 23:
            slider_value = 0 if progress >= 1 else 23
        self.map_ignore_slider_update = True
        self.map_hour_slider.set(slider_value)
        self.map_ignore_slider_update = False


    def _fill_map_station_list(self, hour_frame: pd.DataFrame, display_time: str) -> None:
        """
        Füllt die textuelle Haltestellenliste der Kartenansicht mit den aktuellen Top-Stationen.

        Die Methode schreibt für die gegebene Uhrzeit eine tabellarische Übersicht
        der wichtigsten Haltestellen mit Linien, Einstiegen und Ausstiegen in die
        zugehörige Textbox.

        Parameter:
            hour_frame (pd.DataFrame): Datenrahmen mit den aktuell relevanten
                Haltestellen.
            display_time (str): Formatierte Anzeigezeit des aktuellen Frames.

        Rückgabewerte:
            None: Die Methode aktualisiert ausschließlich die Textbox.

        Fehler/Sonderfälle:
            Ist keine Textbox vorhanden, erfolgt keine Aktion. Bei leerem Frame wird
            statt einer Tabelle ein Hinweis auf fehlende aktive Nachfrage angezeigt.

        Projektkontext:
            Die Methode ergänzt die Kartenmarker um eine kompakte, rangbasierte
            textuelle Übersicht für Nutzer und Stakeholder.
        """
        if self.map_station_textbox is None:
            return
        self.map_station_textbox.configure(state="normal")
        self.map_station_textbox.delete("1.0", "end")
        if hour_frame.empty:
            self.map_station_textbox.insert("1.0", f"{display_time}\n\nKeine aktive Nachfrage.")
            self.map_station_textbox.configure(state="disabled")
            return


        total_boardings = int(round(float(hour_frame["boardings"].sum())))
        total_exiting = int(round(float(hour_frame["exiting"].sum())))
        rows = [
            f"{display_time}",
            f"Gesamt: {total_boardings:,} Einstiege | {total_exiting:,} Ausstiege".replace(",", "."),
            "",
            "Rang Haltestelle            Linien   Ein   Aus",
            "-" * 50,
        ]
        for rank, (_, row) in enumerate(hour_frame.head(22).iterrows(), start=1):
            station = str(row["station"])[:22]
            line_parts = [part.strip() for part in str(row["lines"]).split(",") if part.strip()]
            lines = "/".join(line_parts[:3])
            if len(line_parts) > 3:
                lines += "+"
            boardings = int(round(float(row["boardings"])))
            exiting = int(round(float(row["exiting"])))
            rows.append(f"{rank:>2}.  {station:<22} {lines:<7} {boardings:>5} {exiting:>5}")
        self.map_station_textbox.insert("1.0", "\n".join(rows))
        self.map_station_textbox.configure(state="disabled")


    def _interpolated_hour_frame(self, from_hour: int, to_hour: int, progress: float) -> pd.DataFrame:
        """
        Erzeugt einen interpolierten Nachfrage-Frame zwischen zwei Stunden.

        Die Methode vereinigt die in Start- und Endstunde vorkommenden Haltestellen
        und interpoliert für jede Station Ein- und Ausstiegswerte linear. Die
        übrigen Metadaten werden aus der jeweils verfügbaren Vorlagenzeile übernommen.

        Parameter:
            from_hour (int): Ausgangsstunde.
            to_hour (int): Zielstunde.
            progress (float): Interpolationsfortschritt zwischen 0 und 1.

        Rückgabewerte:
            pd.DataFrame: Interpolierter Datenrahmen über alle im Übergang
            relevanten Haltestellen.

        Fehler/Sonderfälle:
            Existiert für eine Station nur ein Start- oder Endeintrag, wird der
            fehlende Gegenwert als 0 interpretiert.

        Projektkontext:
            Die Methode unterstützt fließende Übergänge zwischen Stundenbildern und
            vermeidet abrupte Sprünge in der Nachfragevisualisierung.
        """
        start_frame = self._hour_frame_for_map(from_hour)
        end_frame = self._hour_frame_for_map(to_hour)
        station_keys = start_frame.index.union(end_frame.index)
        rows = []
        for station_key in station_keys:
            start_row = start_frame.loc[station_key] if station_key in start_frame.index else None
            end_row = end_frame.loc[station_key] if station_key in end_frame.index else None
            template = end_row if end_row is not None else start_row
            if template is None:
                continue
            start_boardings = float(start_row["boardings"]) if start_row is not None else 0.0
            end_boardings = float(end_row["boardings"]) if end_row is not None else 0.0
            start_exiting = float(start_row["exiting"]) if start_row is not None else 0.0
            end_exiting = float(end_row["exiting"]) if end_row is not None else 0.0
            row = template.to_dict()
            row["boardings"] = start_boardings + (end_boardings - start_boardings) * progress
            row["exiting"] = start_exiting + (end_exiting - start_exiting) * progress
            rows.append(row)
        return pd.DataFrame(rows)


    def _render_map_interpolated_frame(
        self,
        from_hour: int,
        to_hour: int,
        progress: float,
        allow_route_fetch: bool = False,
    ) -> None:
        """
        Rendert einen vollständigen interpolierten Kartenframe für Nachfrage, Events und Flotte.

        Die Methode koordiniert die zeitliche Ableitung des aktuellen Zeitpunkts,
        rendert aktive Events, setzt Uhrzeit und Slider, verarbeitet anschließend
        Nachfrage-Frames und synchronisiert schließlich die aktuelle Busflotte. Je
        nach Datenlage werden leere Zustände, Nachfragezustände und Statusmeldungen
        differenziert behandelt.

        Parameter:
            from_hour (int): Ausgangsstunde des Übergangs.
            to_hour (int): Zielstunde des Übergangs.
            progress (float): Interpolationsfortschritt zwischen 0 und 1.
            allow_route_fetch (bool): Gibt an, ob fehlende Straßenrouten für die
                Flottenanimation nachgeladen werden dürfen.

        Rückgabewerte:
            None: Die Methode aktualisiert mehrere Karten- und GUI-Komponenten.

        Fehler/Sonderfälle:
            Ist kein Kartenwidget verfügbar, wird die Methode beendet. Leere
            Animationsdaten oder leere Nachfrageframes führen zu bereinigten
            Kartenmarkern und reduzierten Statusmeldungen statt zu Fehlern.

        Projektkontext:
            Diese Methode ist die zentrale Orchestrierungsfunktion der gesamten
            Kartenansicht, weil sie Nachfrage, Fahrzeuge, Events, Zeitsteuerung und
            Statusdarstellung in einem konsistenten Frame zusammenführt.
        """
        if self.map_widget is None:
            return


        display_time = self._display_time_between(from_hour, to_hour, progress)
        current_minute = self._minute_between(from_hour, to_hour, progress)
        selected_day = self.map_day_entry.get_date() if self.map_day_entry is not None else self.start_picker.get_date()
        active_events = self._render_event_markers(selected_day, current_minute, display_time)
        if self.map_hour_label is not None:
            self.map_hour_label.configure(text=display_time)
        self._set_map_slider_between(from_hour, to_hour, progress)


        if self.map_animation_data.empty:
            self._clear_map_markers()
            self._fill_map_station_list(pd.DataFrame(), display_time)
            visible_buses, skipped_buses = self._sync_fleet_to_minute(current_minute, allow_route_fetch)
            if self.map_status_label is not None:
                self.map_status_label.configure(
                    text=self._fleet_status_text(display_time, visible_buses, skipped_buses, active_events=active_events)
                )
            return


        hour_frame = self._stop_event_frame_for_map(current_minute)
        if hour_frame.empty or "boardings" not in hour_frame:
            self._clear_map_markers()
            self._fill_map_station_list(pd.DataFrame(), display_time)
            visible_buses, skipped_buses = self._sync_fleet_to_minute(current_minute, allow_route_fetch)
            if self.map_status_label is not None:
                self.map_status_label.configure(
                    text=self._fleet_status_text(display_time, visible_buses, skipped_buses, active_events=active_events)
                )
            return
        hour_frame = hour_frame[(hour_frame["boardings"] > 0) | (hour_frame["exiting"] > 0)].copy()
        hour_frame["flow_total"] = hour_frame["boardings"] + hour_frame["exiting"]
        sort_columns = ["nearest_stop", "flow_total"] if "nearest_stop" in hour_frame else ["flow_total"]
        hour_frame = hour_frame.sort_values(sort_columns, ascending=[True, False] if len(sort_columns) == 2 else False)


        if hour_frame.empty:
            self._clear_map_markers()
            self._fill_map_station_list(pd.DataFrame(), display_time)
            visible_buses, skipped_buses = self._sync_fleet_to_minute(current_minute, allow_route_fetch)
            if self.map_status_label is not None:
                self.map_status_label.configure(
                    text=self._fleet_status_text(display_time, visible_buses, skipped_buses, active_events=active_events)
                )
            return


        self._fill_map_station_list(hour_frame, display_time)
        self._render_demand_markers(hour_frame)
        visible_buses, skipped_buses = self._sync_fleet_to_minute(current_minute, allow_route_fetch)
        if self.map_status_label is not None:
            total_boardings = int(round(float(hour_frame["boardings"].sum())))
            total_exiting = int(round(float(hour_frame["exiting"].sum())))
            self.map_status_label.configure(
                text=self._fleet_status_text(
                    display_time,
                    visible_buses,
                    skipped_buses,
                    active_stops=len(hour_frame),
                    total_boardings=total_boardings,
                    total_exiting=total_exiting,
                    active_events=active_events,
                )
            )


    def _fleet_status_text(
        self,
        display_time: str,
        visible_buses: int,
        skipped_buses: int,
        active_stops: int | None = None,
        total_boardings: int | None = None,
        total_exiting: int | None = None,
        active_events: int | None = None,
    ) -> str:
        """
        Erzeugt den Statuszeilentext für die kombinierte Kartenansicht.

        Die Methode kombiniert Zeitangabe, Nachfragekennzahlen, Eventanzahl sowie
        Sichtbarkeit und Routingstatus der Flotte zu einer kompakten textuellen
        Zusammenfassung für die Benutzeroberfläche.

        Parameter:
            display_time (str): Aktuelle Anzeigezeit.
            visible_buses (int): Anzahl aktuell sichtbarer Busse.
            skipped_buses (int): Anzahl wegen fehlender Routingdaten nicht
                darstellbarer Busse.
            active_stops (int | None): Anzahl aktiver Haltestellen im aktuellen Frame.
            total_boardings (int | None): Gesamtzahl der Einstiege im aktuellen Frame.
            total_exiting (int | None): Gesamtzahl der Ausstiege im aktuellen Frame.
            active_events (int | None): Anzahl aktiver Events.

        Rückgabewerte:
            str: Zusammengesetzter Statuszeilentext.

        Fehler/Sonderfälle:
            Liegen keine Flottenfahrten vor, wird ein gesonderter Hinweistext
            erzeugt. Eventangaben werden nur ausgegeben, wenn aktive Events
            vorhanden sind.

        Projektkontext:
            Die Methode verdichtet mehrere Teilzustände des Gesamtsystems zu einer
            für Nutzer und Stakeholder schnell erfassbaren Lagebeschreibung.
        """
        demand_text = ""
        if active_stops is not None and total_boardings is not None and total_exiting is not None:
            demand_text = (
                f" | {active_stops} aktive Stops | "
                f"{total_boardings:,} Einstiege | {total_exiting:,} Ausstiege"
            ).replace(",", ".")
        event_text = f" | {active_events} aktive Events" if active_events else ""
        if not self.map_fleet_trips:
            return f"{display_time}{demand_text}{event_text} | Keine Busfahrten für diesen Tag geladen."
        if skipped_buses:
            return (
                f"{display_time}{demand_text}{event_text} | {visible_buses} Busse sichtbar | "
                f"{skipped_buses} Abschnitte warten auf Straßenrouting."
            )
        return f"{display_time}{demand_text}{event_text} | {visible_buses} Busse gerade unterwegs."


    def _show_event_marker_details(self, marker) -> None:
        """
        Zeigt Detailinformationen zu einem Eventmarker in der Statuszeile an.

        Die Methode liest die im Marker gespeicherten Metadaten aus, formatiert
        Start- und Endzeit sowie Eventtyp und schreibt diese Informationen in das
        Statuslabel.

        Parameter:
            marker: Ausgewählter Kartenmarker mit Eventdaten im Attribut `data`.

        Rückgabewerte:
            None: Die Methode aktualisiert ausschließlich die Statusanzeige.

        Fehler/Sonderfälle:
            Fehlt ein Statuslabel, erfolgt keine Aktion. Fehlende oder ungültige
            Zeitstempel werden als Platzhalter „--:--“ dargestellt.

        Projektkontext:
            Die Methode ermöglicht eine fachliche Einordnung aktiver Events direkt
            aus der Karteninteraktion heraus.
        """
        if self.map_status_label is None:
            return
        data = marker.data or {}
        start = pd.Timestamp(data.get("start")) if data.get("start") is not None else None
        end = pd.Timestamp(data.get("end")) if data.get("end") is not None else None
        start_text = start.strftime("%H:%M") if start is not None and not pd.isna(start) else "--:--"
        end_text = end.strftime("%H:%M") if end is not None and not pd.isna(end) else "--:--"
        kind = "Konzert" if data.get("concert") else "Event"
        duration = float(data.get("duration_hours", 0.0))
        self.map_status_label.configure(
            text=(
                f"{kind}: {data.get('name', 'Unbekannt')} | "
                f"{start_text}-{end_text} Uhr | "
                f"Dauer {duration:.1f} h | aktiv bei {data.get('time', '')}"
            )
        )


    def _show_map_marker_details(self, marker) -> None:
        """
        Zeigt Detailinformationen zu einem Nachfrage-Marker in der Statuszeile an.

        Die Methode nutzt die im Marker hinterlegten Haltestellen- und
        Nachfrageinformationen, um eine kompakte textuelle Zusammenfassung des
        gewählten Nachfragepunkts anzuzeigen.

        Parameter:
            marker: Ausgewählter Nachfrage-Marker mit Metadaten im Attribut `data`.

        Rückgabewerte:
            None: Die Methode aktualisiert ausschließlich das Statuslabel.

        Fehler/Sonderfälle:
            Fehlt das Statuslabel, erfolgt keine Aktion. Für fehlende Zeitwerte wird
            auf die aktuelle Kartenstunde zurückgegriffen.

        Projektkontext:
            Die Methode macht die Kartenmarker interaktiv interpretierbar und bietet
            eine textuelle Detailansicht ohne zusätzlichen Dialog.
        """
        if self.map_status_label is None:
            return
        data = marker.data or {}
        self.map_status_label.configure(
            text=(
                f"{data.get('station', 'Haltestelle')} | Linie {data.get('lines', '-')} | "
                f"{data.get('time', f'{self.map_current_hour:02d}:00 Uhr')} | "
                f"{int(data.get('boardings', 0)):,} Einstiege | "
                f"{int(data.get('exiting', 0)):,} Ausstiege"
            ).replace(",", ".")
        )


    def _on_prediction_hour_slider(self, value: float) -> None:
        """
        Aktualisiert das Stundenlabel des Prognosebereichs anhand des Sliders.

        Die Methode rundet den Sliderwert auf eine volle Stunde und schreibt diese
        formatiert in das zugehörige Anzeige-Label.

        Parameter:
            value (float): Gewählter Sliderwert für die Prognosestunde.

        Rückgabewerte:
            None: Die Methode aktualisiert ausschließlich die GUI-Anzeige.

        Fehler/Sonderfälle:
            Fehlt das Stundenlabel, erfolgt keine Aktion.

        Projektkontext:
            Die Methode sorgt für eine verständliche zeitliche Rückmeldung in der
            Prognosekonfiguration.
        """
        hour = int(round(float(value)))
        if self.prediction_hour_label is not None:
            self.prediction_hour_label.configure(text=f"{hour:02d}:00 Uhr")


    def _refresh_prediction_line_menu(self) -> None:
        """
        Aktualisiert die Linienauswahl des Prognosebereichs.

        Die Methode baut die Zuordnung zwischen sichtbaren Linienlabels und internen
        Linienkennungen neu auf, setzt einen gültigen Standardwert und lädt danach
        die zugehörigen Haltestellenoptionen.

        

        Rückgabewerte:
            None: Die Methode aktualisiert Menüs und interne Lookup-Strukturen.

        Fehler/Sonderfälle:
            Sind keine Linien ausgewählt, wird ein Platzhalterwert gesetzt.

        Projektkontext:
            Die Methode verbindet die globale Linienselektion des Projekts mit der
            operativen Prognosefunktion.
        """
        self.prediction_line_lookup = {}
        values = [self.repo.label_for_line(line) for line in self.selected_lines] if self.selected_lines else ["Keine Linie"]
        self.prediction_line_lookup = {self.repo.label_for_line(line): line for line in self.selected_lines}
        if self.prediction_line_menu is not None:
            self.prediction_line_menu.configure(values=values)
        if self.prediction_line_choice.get() not in self.prediction_line_lookup:
            self.prediction_line_choice.set(values[0] if values else "Keine Linie")
        self._refresh_prediction_station_menu()


    def _selected_prediction_line(self) -> int | None:
        """
        Liefert die aktuell im Prognosebereich ausgewählte Linie.

        Die Methode übersetzt das im Menü angezeigte Linienlabel in die interne
        numerische Linienkennung.

        

        Rückgabewerte:
            int | None: Gewählte Liniennummer oder None bei ungültiger Auswahl.

        Fehler/Sonderfälle:
            Nicht auflösbare Menüwerte führen zu None.

        Projektkontext:
            Die Methode ist die Grundlage für linienbezogene Prognose- und
            Haltestellenabfragen.
        """
        return self.prediction_line_lookup.get(self.prediction_line_choice.get())


    def _refresh_prediction_station_menu(self) -> None:
        """
        Aktualisiert die Haltestellenauswahl für den Prognosebereich.

        Die Methode bestimmt abhängig von der aktuell gewählten Linie die
        verfügbaren Haltestellen, bevorzugt aus einem vorbereiteten Stationskatalog
        und greift bei Bedarf auf Rohdaten aus dem Repository zurück. Die
        Haltestellen werden fachlich nach Stationsnummer sortiert.

        

        Rückgabewerte:
            None: Die Methode aktualisiert Menüs, Lookup-Tabellen und Hilfslisten.

        Fehler/Sonderfälle:
            Existiert kein Katalogeintrag für die Linie, wird ein Fallback über
            Rohdaten versucht. Bei Fehlern oder fehlenden Daten wird auf
            „Alle Haltestellen“ zurückgefallen.

        Projektkontext:
            Die Methode bereitet die Auswahl lokaler Prognoseausschnitte vor, etwa
            für Vorhersagen der nächsten Haltestellen einer Linie.
        """
        self.prediction_station_lookup = {}
        self.prediction_station_rows = []
        line = self._selected_prediction_line()
        values = ["Alle Haltestellen"]
        if line is not None:
            catalog = self.prediction_service.station_catalog
            if not catalog.empty and int(line) in set(catalog["line"].astype(int)):
                station_frame = catalog[catalog["line"].astype(int) == int(line)].copy()
                station_frame = station_frame.sort_values(["station_number", "station"])
                values = []
                for _, row in station_frame.iterrows():
                    label = f"{int(round(float(row['station_number']))):02d} | {row['station']}"
                    values.append(label)
                    self.prediction_station_lookup[label] = str(row["station_key"])
                    self.prediction_station_rows.append(
                        {
                            "label": label,
                            "station_key": str(row["station_key"]),
                            "station_number": float(row["station_number"]),
                            "station": str(row["station"]),
                        }
                    )
            else:
                try:
                    raw = self.repo.load_line_range(line, self.start_picker.get_date(), self.end_picker.get_date())
                except Exception:
                    raw = pd.DataFrame()
                if not raw.empty and {"station", "station_number"}.issubset(raw.columns):
                    station_frame = (
                        raw.assign(station_key=raw["station"].map(normalize_station_name))
                        .groupby(["station_key", "station"], as_index=False)
                        .agg(station_number=("station_number", "median"))
                        .sort_values(["station_number", "station"])
                    )
                    values = []
                    for _, row in station_frame.iterrows():
                        label = f"{int(round(float(row['station_number']))):02d} | {row['station']}"
                        values.append(label)
                        self.prediction_station_lookup[label] = str(row["station_key"])
                        self.prediction_station_rows.append(
                            {
                                "label": label,
                                "station_key": str(row["station_key"]),
                                "station_number": float(row["station_number"]),
                                "station": str(row["station"]),
                            }
                        )
        if not values:
            values = ["Alle Haltestellen"]
        if self.prediction_station_menu is not None:
            self.prediction_station_menu.configure(values=values)
        if self.prediction_station_choice.get() not in self.prediction_station_lookup:
            self.prediction_station_choice.set(values[0])
        self._sync_prediction_mode_controls()


    def _sync_prediction_mode_controls(self) -> None:
        """
        Synchronisiert Sichtbarkeit und Aktivierbarkeit der Prognose-Steuerelemente mit dem Modus.

        Je nach gewähltem Prognosemodus werden Horizon-, Stations- und
        Haltestellenanzahl-Steuerelemente ein- oder ausgeblendet sowie aktiviert oder
        deaktiviert.

        

        Rückgabewerte:
            None: Die Methode verändert ausschließlich den GUI-Zustand.

        Fehler/Sonderfälle:
            Nicht initialisierte Widgets werden tolerant behandelt.

        Projektkontext:
            Die Methode sorgt dafür, dass die Benutzeroberfläche nur die im aktuellen
            Prognosemodus relevanten Eingabemöglichkeiten anbietet.
        """
        station_mode = self.prediction_mode_choice.get() == "Nächste Haltestellen"
        if self.prediction_horizon_box is not None:
            if station_mode:
                self.prediction_horizon_box.grid_remove()
            else:
                self.prediction_horizon_box.grid()
        for box in (self.prediction_station_box, self.prediction_stop_count_box):
            if box is not None:
                if station_mode:
                    box.grid()
                else:
                    box.grid_remove()
        if self.prediction_horizon_menu is not None:
            self.prediction_horizon_menu.configure(state="disabled" if station_mode else "normal")
        if self.prediction_station_menu is not None:
            state = "normal" if station_mode and bool(self.prediction_station_lookup) else "disabled"
            self.prediction_station_menu.configure(state=state)
        if self.prediction_stop_count_menu is not None:
            state = "normal" if station_mode and bool(self.prediction_station_lookup) else "disabled"
            self.prediction_stop_count_menu.configure(state=state)


    def _selected_prediction_station_key(self) -> str | None:
        """
        Liefert den aktuell ausgewählten Stationsschlüssel für haltestellenbezogene Prognosen.

        Die Methode gibt nur dann einen Stationsschlüssel zurück, wenn sich die
        Oberfläche im Modus „Nächste Haltestellen“ befindet.

        

        Rückgabewerte:
            str | None: Gewählter station_key oder None, falls der aktuelle Modus
            keinen Stationsbezug vorsieht.

        Fehler/Sonderfälle:
            Ungültige Menüwerte führen zu None.

        Projektkontext:
            Die Methode trennt globale Linienprognosen von lokal fokussierten
            Haltestellenprognosen.
        """
        if self.prediction_mode_choice.get() != "Nächste Haltestellen":
            return None
        return self.prediction_station_lookup.get(self.prediction_station_choice.get())


    def _prediction_stop_count(self) -> int:
        """
        Extrahiert die gewünschte Anzahl zu prognostizierender Folgestopps aus der GUI-Auswahl.

        Die Methode sucht per regulärem Ausdruck nach einer Zahl im Anzeigetext und
        verwendet diese als Stopanzahl.

        

        Rückgabewerte:
            int: Anzahl der einzubeziehenden Stopps.

        Fehler/Sonderfälle:
            Wird keine Zahl gefunden, fällt die Methode auf den Standardwert 5
            zurück.

        Projektkontext:
            Die Methode steuert den Umfang lokaler Mehrstopp-Prognosen entlang einer
            Linienfolge.
        """
        match = re.search(r"\d+", self.prediction_stop_count_choice.get())
        return int(match.group(0)) if match else 5


    def _selected_prediction_station_keys(self) -> set[str] | None:
        """
        Bestimmt die Menge der Stationsschlüssel für eine sequenzielle Haltestellenprognose.

        Ausgehend von der aktuell gewählten Startstation werden so viele
        aufeinanderfolgende Haltestellen aus der vorbereiteten Stationsliste
        ausgewählt, wie durch die Stopanzahl vorgegeben sind.

        

        Rückgabewerte:
            set[str] | None: Menge ausgewählter Stationsschlüssel oder None, falls
            keine stationsbezogene Prognose aktiv ist.

        Fehler/Sonderfälle:
            Ist die Startstation nicht in der vorbereiteten Reihenfolge auffindbar,
            wird ersatzweise nur diese einzelne Station zurückgegeben.

        Projektkontext:
            Die Methode ermöglicht Vorhersagen für einen zusammenhängenden
            Streckenabschnitt statt nur für eine einzelne Haltestelle.
        """
        station_key = self._selected_prediction_station_key()
        if station_key is None:
            return None
        rows = self.prediction_station_rows
        start_index = next((index for index, row in enumerate(rows) if row["station_key"] == station_key), None)
        if start_index is None:
            return {station_key}
        stop_count = self._prediction_stop_count()
        selected_rows = rows[start_index : start_index + stop_count]
        return {str(row["station_key"]) for row in selected_rows}


    def _set_prediction_busy(self, busy: bool, message: str = "") -> None:
        """
        Setzt den Busy-Zustand des Prognosebereichs und passt die Oberfläche entsprechend an.

        Die Methode aktiviert oder deaktiviert relevante Eingabeelemente, verwaltet
        einen eventuell laufenden Busy-Animations-Callback und aktualisiert die
        Statusanzeige des Prognosebereichs.

        Parameter:
            busy (bool): Gibt an, ob sich der Prognosebereich in einem laufenden
                Arbeitszustand befindet.
            message (str): Statusnachricht für den Busy-Zustand.

        Rückgabewerte:
            None: Die Methode verändert GUI-Zustände und interne Busy-Flags.

        Fehler/Sonderfälle:
            Fehler beim Abbrechen vorhandener Busy-Callbacks oder beim Umschalten
            einzelner Widgets werden tolerant behandelt.

        Projektkontext:
            Die Methode schützt die Prognoseoberfläche während Trainings- oder
            Vorhersageprozessen vor inkonsistenten Benutzereingaben.
        """
        self.prediction_busy = busy
        self.prediction_busy_message = message
        self.prediction_busy_step = 0
        if self.prediction_busy_after_id is not None:
            try:
                self.after_cancel(self.prediction_busy_after_id)
            except tk.TclError:
                pass
            self.prediction_busy_after_id = None


        state = "disabled" if busy else "normal"
        for widget in (
            self.prediction_train_button,
            self.prediction_predict_button,
            self.prediction_line_menu,
            self.prediction_mode_menu,
            self.prediction_horizon_menu,
            self.prediction_station_menu,
            self.prediction_stop_count_menu,
            self.prediction_hour_slider,
        ):
            if widget is not None:
                try:
                    widget.configure(state=state)
                except tk.TclError:
                    pass
        if self.prediction_day_entry is not None:
            try:
                self.prediction_day_entry.configure(state=state)
            except tk.TclError:
                pass


        if self.prediction_status_label is not None:
            self.prediction_status_label.configure(
                text=f"{message} | Fenster bleibt aktiv" if busy else "Bereit für kurzfristige operative Prognosen.",
                text_color=PALETTE["text"] if busy else PALETTE["muted"],
            )
        if busy:
            self.status_text.set(message)
        else:
            self._sync_prediction_mode_controls()
            self._refresh_main_scroll()


    def _refresh_main_scroll(self) -> None:
        """
        Aktualisiert den Layout- und Scrollzustand des Hauptcontainers.

        Die Methode stößt Re-Layout und Größenberechnung des Scrollcontainers und
        seines Canvas an, damit Änderungen in Unterbereichen der Oberfläche korrekt
        sichtbar werden.

        Rückgabewerte:
            None: Die Methode aktualisiert ausschließlich GUI-Layoutzustände.

        Fehler/Sonderfälle:
            Fehlende Canvas-Referenzen oder Tkinter-Fehler werden tolerant behandelt.

        Projektkontext:
            Die Methode stellt sicher, dass dynamisch ein- und ausgeblendete
            Prognose- oder Kartenbereiche korrekt in den Haupt-Scrollbereich
            integriert bleiben.
        """
        if self.main_scroll is None:
            return
        try:
            self.main_scroll.update_idletasks()
            self.main_scroll._parent_canvas.update_idletasks()
        except (AttributeError, tk.TclError):
            pass


    def _animate_prediction_busy(self) -> None:
        """
        Animiert die Busy-Anzeige des Prognosebereichs durch periodisch wechselnde Punkte.

        Die Methode ergänzt zyklisch Punkte an die Busy-Nachricht, aktualisiert die
        Prognose-Statusanzeige sowie den globalen Statustext und plant sich selbst
        anschließend erneut über Tkinter-after ein.

        Rückgabewerte:
            None: Die Methode aktualisiert Statusanzeigen und registriert einen
            Folge-Callback.

        Fehler/Sonderfälle:
            Ist der Prognosebereich nicht im Busy-Zustand, wird keine weitere
            Animation ausgeführt.

        Projektkontext:
            Die Methode signalisiert laufende Berechnungen im Prognosebereich, ohne
            die Oberfläche zu blockieren.
        """
        if not self.prediction_busy:
            return
        dots = "." * (self.prediction_busy_step % 4)
        text = f"{self.prediction_busy_message}{dots} | Fenster bleibt aktiv"
        if self.prediction_status_label is not None:
            self.prediction_status_label.configure(text=text)
        self.status_text.set(self.prediction_busy_message + dots)
        self.prediction_busy_step += 1
        self.prediction_busy_after_id = self.after(450, self._animate_prediction_busy)

    def _show_prediction_message(self, text: str) -> None:
        """
        Zeigt einen Textinhalt im Prognose-Textboxbereich an.

        Die Methode entsperrt die Textbox temporär, ersetzt den bisherigen Inhalt
        vollständig durch den übergebenen Text und deaktiviert die Box anschließend
        wieder, damit die Ausgabe als reine Ergebnis- oder Statusanzeige dient.

        Parameter:
            text (str): Anzuzeigender Text im Prognosebereich.

        Rückgabewerte:
            None: Die Methode aktualisiert ausschließlich die GUI-Ausgabe.

        Fehler/Sonderfälle:
            Ist keine Prognose-Textbox vorhanden, wird die Methode ohne Wirkung
            beendet.

        Projektkontext:
            Die Methode bildet die zentrale Textausgabe für Statusmeldungen,
            Trainingshinweise und Vorhersageergebnisse im Prognosemodul.
        """
        if self.prediction_textbox is None:
            return
        self.prediction_textbox.configure(state="normal")
        self.prediction_textbox.delete("1.0", "end")
        self.prediction_textbox.insert("1.0", text)
        self.prediction_textbox.configure(state="disabled")


    def _finish_prediction_worker(self, result: PredictionResult, renderer) -> None:
        """
        Beendet einen asynchronen Prognose-Worker erfolgreich und übergibt das Ergebnis an den Renderer.

        Die Methode hebt zunächst den Busy-Zustand des Prognosebereichs auf und
        ruft danach die übergebene Renderfunktion mit dem berechneten Ergebnis auf.
        Fehler in der Ergebnisdarstellung werden abgefangen und an die zentrale
        Fehlerbehandlung weitergegeben.

        Parameter:
            result (PredictionResult): Ergebnisobjekt eines Trainings- oder
                Prognoselaufs.
            renderer: Callback zur Darstellung des Worker-Ergebnisses in der GUI.

        Rückgabewerte:
            None: Die Methode koordiniert Zustandswechsel und Ergebnisdarstellung.

        Fehler/Sonderfälle:
            Löst der Renderer eine Ausnahme aus, wird diese nicht unkontrolliert
            propagiert, sondern über die zentrale Fehlerbehandlung ausgegeben.

        Projektkontext:
            Die Methode trennt Hintergrundberechnung und GUI-Aktualisierung sauber
            voneinander und ist damit zentral für eine stabile asynchrone
            Prognoseverarbeitung.
        """
        self._set_prediction_busy(False)
        try:
            renderer(result)
        except Exception:
            self._handle_prediction_error(traceback.format_exc())


    def _handle_prediction_error(self, details: str) -> None:
        """
        Behandelt Fehler aus dem Prognoseprozess und macht sie in der GUI sichtbar.

        Die Methode beendet den Busy-Zustand, extrahiert eine kurze Fehlermeldung
        aus dem übergebenen Traceback und zeigt zusätzlich gekürzte technische
        Details im Prognose-Textboxbereich an.

        Parameter:
            details (str): Vollständiger Fehlertext, typischerweise aus einem
                Traceback erzeugt.

        Rückgabewerte:
            None: Die Methode aktualisiert Statusanzeige und Prognoseausgabe.

        Fehler/Sonderfälle:
            Ist kein verwertbarer Fehlertest vorhanden, wird „Unbekannter Fehler“
            angezeigt. Die Detailausgabe wird auf die letzten 3500 Zeichen begrenzt.

        Projektkontext:
            Die Methode sorgt dafür, dass Fehler in Trainings- oder Prognoseabläufen
            nachvollziehbar bleiben, ohne die Anwendung unkontrolliert zu stoppen.
        """
        self._set_prediction_busy(False)
        short = details.strip().splitlines()[-1] if details.strip() else "Unbekannter Fehler"
        self.status_text.set(f"Prediction-Fehler: {short}")
        self._show_prediction_message(
            "Prediction konnte nicht berechnet werden.\n\n"
            f"{short}\n\n"
            "Technische Details:\n"
            f"{details[-3500:]}"
        )


    def _start_prediction_worker(self, message: str, worker, renderer) -> None:
        """
        Startet eine Hintergrundausführung für Training oder Vorhersage im Prognosebereich.

        Die Methode aktiviert zunächst den Busy-Zustand und startet danach einen
        separaten Hintergrund-Thread, der die eigentliche Rechenarbeit übernimmt.
        Python-Threads können als Daemon-Threads gestartet werden, sodass sie das
        Beenden des Programms nicht blockieren, wenn nur solche Threads noch laufen. [web:131]

        Parameter:
            message (str): Statusmeldung für die Busy-Anzeige.
            worker: Aufzurufende Funktion für die Hintergrundberechnung.
            renderer: Callback zur Darstellung des Ergebnisses im GUI-Thread.

        Rückgabewerte:
            None: Die Methode startet den asynchronen Ablauf und aktualisiert die GUI.

        Fehler/Sonderfälle:
            Läuft bereits ein Prognoseprozess, wird kein neuer gestartet. Fehler im
            Worker werden abgefangen und über den GUI-Thread an die zentrale
            Fehlerbehandlung delegiert. Tkinter-Fehler beim Rückmelden in die GUI
            werden tolerant behandelt.

        Projektkontext:
            Die Methode ist zentral für eine reaktionsfähige Anwendung, da rechen-
            intensive Prognoseaufgaben die GUI nicht blockieren sollen.
        """
        if self.prediction_busy:
            self.status_text.set("Prediction läuft bereits im Hintergrund.")
            return
        self._set_prediction_busy(True, message)
        self._show_prediction_message(f"{message}...\n\nDie GUI bleibt aktiv.")


        def run_worker() -> None:
            """
            Führt die eigentliche Worker-Funktion im Hintergrund aus.

            Die lokale Hilfsfunktion kapselt Fehlerbehandlung und die Rückgabe in den
            GUI-Thread via `after`.

            Parameter:
                Keine.

            Rückgabewerte:
                None: Die Funktion übergibt Ergebnisse indirekt an GUI-Callbacks.

            Fehler/Sonderfälle:
                Ausnahmen im Worker werden abgefangen und als formatierter Traceback
                an die GUI gemeldet.

            Projektkontext:
                Die Hilfsfunktion trennt Rechenlogik vom GUI-Thread und ermöglicht
                eine sichere asynchrone Verarbeitung.
            """
            try:
                result = worker()
            except Exception:
                details = traceback.format_exc()
                try:
                    self.after(0, lambda: self._handle_prediction_error(details))
                except tk.TclError:
                    pass
                return
            try:
                self.after(0, lambda: self._finish_prediction_worker(result, renderer))
            except tk.TclError:
                pass


        threading.Thread(target=run_worker, daemon=True).start()


    def _train_prediction_model(self) -> None:
        """
        Startet das Training des Prognosemodells für die aktuell gewählte Linie.

        Die Methode prüft zunächst die Linienauswahl und den Datumsbereich, korrigiert
        bei Bedarf ein ungültiges Enddatum und startet anschließend den asynchronen
        Trainingsprozess über den Prediction-Service.

        Parameter:
            Keine.

        Rückgabewerte:
            None: Die Methode stößt einen Trainingsprozess an und aktualisiert danach
            GUI-Elemente über Callbacks.

        Fehler/Sonderfälle:
            Ist keine Linie ausgewählt, wird der Benutzerstatus aktualisiert und
            das Training nicht gestartet. Liegt das Enddatum vor dem Startdatum,
            wird es automatisch angepasst.

        Projektkontext:
            Die Methode bildet den Einstiegspunkt zum Aufbau oder Aktualisieren
            persistenter Prognosemodelle je Linie.
        """
        line = self._selected_prediction_line()
        if line is None:
            self._refresh_prediction_line_menu()
            line = self._selected_prediction_line()
        if line is None:
            self.status_text.set("Prediction braucht eine aktive Linie.")
            return
        lines = [line]
        start = self.start_picker.get_date()
        end = self.end_picker.get_date()
        if start > end:
            self.end_picker.set_date(start)
            end = start
            self.status_text.set("Enddatum wurde automatisch an das Startdatum angepasst.")


        def worker() -> PredictionResult:
            """
            Führt das Modelltraining im Hintergrund aus.

            Parameter:
                Keine.

            Rückgabewerte:
                PredictionResult: Ergebnisobjekt des Trainings.

            Fehler/Sonderfälle:
                Fehler werden nicht lokal behandelt, sondern an den aufrufenden
                Worker-Mechanismus weitergereicht.

            Projektkontext:
                Die lokale Funktion kapselt den reinen Trainingsaufruf für die
                asynchrone Ausführung.
            """
            return self.prediction_service.fit(lines, start, end)


        def renderer(result: PredictionResult) -> None:
            """
            Rendert das Trainingsergebnis in der GUI.

            Parameter:
                result (PredictionResult): Ergebnis des Trainingslaufs.

            Rückgabewerte:
                None: Die Funktion aktualisiert Metriken, Status und Textausgabe.

            Fehler/Sonderfälle:
                Keine lokale Fehlerbehandlung; Ausnahmen werden im aufrufenden
                Abschlussmechanismus abgefangen.

            Projektkontext:
                Die lokale Funktion formuliert die fachliche Rückmeldung nach
                erfolgreichem Modelltraining für den Benutzer.
            """
            self._update_prediction_metrics(result)
            self.status_text.set(result.message)
            self._show_prediction_message(
                f"{result.message}\n\n"
                "Das Modell wurde dauerhaft unter prediction/models gespeichert.\n"
                "Danach kannst du für diese Linie eine kurzfristige operative Prognose berechnen."
            )


        self._start_prediction_worker(f"Modell für Linie {line} trainiert und gespeichert", worker, renderer)


    def _prediction_horizon_hours(self) -> int:
        """
        Extrahiert den Prognosehorizont in Stunden aus der GUI-Auswahl.

        Die Methode liest den Text der Horizontauswahl aus und sucht darin eine
        numerische Stundenangabe.

        Parameter:
            Keine.

        Rückgabewerte:
            int: Anzahl der Prognosestunden.

        Fehler/Sonderfälle:
            Wird keine Zahl gefunden, fällt die Methode auf den Standardwert 4
            Stunden zurück.

        Projektkontext:
            Die Methode verbindet die sprachliche Benutzerwahl mit dem numerischen
            Horizontparameter für den Prediction-Service.
        """
        text = self.prediction_horizon_choice.get()
        match = re.search(r"\d+", text)
        return int(match.group(0)) if match else 4


    def _run_prediction(self) -> None:
        """
        Startet eine kurzfristige operative Prognose für die aktuell gewählte Linie.

        Die Methode bestimmt Linie, Datum, Stunde, Modus und gegebenenfalls die
        ausgewählten Zielhaltestellen. Falls für die Linie noch kein trainiertes
        Modell verfügbar ist, wird zunächst versucht, ein Modell zu laden oder
        ersatzweise neu zu trainieren.

        Parameter:
            Keine.

        Rückgabewerte:
            None: Die Methode stößt einen asynchronen Prognoselauf an.

        Fehler/Sonderfälle:
            Fehlt eine gültige Linie oder im Haltestellenmodus eine Startstation,
            wird keine Prognose gestartet. Ein ungültiger Datumsbereich wird
            automatisch korrigiert.

        Projektkontext:
            Diese Methode ist der zentrale Einstiegspunkt für kurzfristige
            Nachfrageprognosen im operativen Planungskontext.
        """
        line = self._selected_prediction_line()
        if line is None:
            self._refresh_prediction_line_menu()
            line = self._selected_prediction_line()
        if line is None:
            self.status_text.set("Keine Linie für Prediction ausgewählt.")
            return
        selected_day = self.prediction_day_entry.get_date() if self.prediction_day_entry is not None else self.start_picker.get_date()
        hour = int(round(self.prediction_hour_slider.get())) if self.prediction_hour_slider is not None else 8
        station_keys = self._selected_prediction_station_keys()
        station_mode = self.prediction_mode_choice.get() == "Nächste Haltestellen"
        if station_mode and not station_keys:
            self.status_text.set("Bitte eine Start-Haltestelle für die Prediction auswählen.")
            return
        horizon_hours = 1 if station_mode else self._prediction_horizon_hours()
        lines = [line]
        start = self.start_picker.get_date()
        end = self.end_picker.get_date()
        if start > end:
            self.end_picker.set_date(start)
            end = start
            self.status_text.set("Enddatum wurde automatisch an das Startdatum angepasst.")


        def worker() -> PredictionResult:
            """
            Führt die eigentliche Kurzfristprognose im Hintergrund aus.

            Die Funktion lädt bei Bedarf ein vorhandenes Modell oder trainiert
            ersatzweise eines neu und ruft danach die operative Vorhersage auf.

            Parameter:
                Keine.

            Rückgabewerte:
                PredictionResult: Ergebnis der Vorhersage oder des gegebenenfalls
                zuvor notwendigen Trainings.

            Fehler/Sonderfälle:
                Schlägt das Laden eines Modells fehl, wird ein Training ausgelöst.
                Falls danach weiterhin kein trainiertes Modell verfügbar ist, wird
                das Trainingsergebnis direkt zurückgegeben.

            Projektkontext:
                Die Hilfsfunktion kapselt die fachliche Logik „Modell verfügbar machen
                und anschließend prognostizieren“ in einem Hintergrund-Worker.
            """
            trained_lines = getattr(self.prediction_service, "trained_lines", set())
            if not self.prediction_service.trained or int(line) not in trained_lines:
                loaded = False
                if hasattr(self.prediction_service, "load_for_lines"):
                    loaded = self.prediction_service.load_for_lines(lines)
                if not loaded:
                    train_result = self.prediction_service.fit(lines, start, end)
                    if not self.prediction_service.trained:
                        return train_result
            return self.prediction_service.predict_short_term(
                line,
                selected_day,
                hour,
                horizon_hours,
                station_keys=station_keys,
                cost_per_bus_hour=self._current_bus_hourly_cost(),
            )


        def renderer(result: PredictionResult) -> None:
            """
            Rendert das Ergebnis der Kurzfristprognose in der GUI.

            Parameter:
                result (PredictionResult): Ergebnisobjekt der Prognose.

            Rückgabewerte:
                None: Die Funktion aktualisiert Metriken, Diagramm, Textausgabe und Status.

            Fehler/Sonderfälle:
                Keine lokale Fehlerbehandlung; Ausnahmen werden im übergeordneten
                Abschlussmechanismus behandelt.

            Projektkontext:
                Die lokale Funktion bildet den Übergang von der Modelllogik zur
                nutzerorientierten Darstellung des Prognoseergebnisses.
            """
            self._update_prediction_metrics(result)
            self._render_prediction_result(result)
            self.status_text.set(result.message)


        self._start_prediction_worker("Kurzfrist-Forecast berechnet", worker, renderer)


    def _update_prediction_metrics(self, result: PredictionResult) -> None:
        """
        Aktualisiert die Kennzahlenanzeige des Prognosebereichs anhand eines Ergebnisses.

        Die Methode liest Modellmetriken und Ensemble-Gewichte aus dem
        Ergebnisobjekt aus und schreibt sie in die zugehörigen GUI-Labels.

        Parameter:
            result (PredictionResult): Ergebnisobjekt mit Metriken und Gewichten.

        Rückgabewerte:
            None: Die Methode aktualisiert ausschließlich die Metrik-Labels.

        Fehler/Sonderfälle:
            Nicht endliche Werte werden als „-“ dargestellt. Fehlende Kennzahlen oder
            Gewichte werden mit Standardwerten behandelt.

        Projektkontext:
            Die Methode macht die Qualität und Gewichtung der Modellkomponenten
            transparent, insbesondere im Zusammenspiel von Random Forest und GNN.
        """
        def metric_text(value: float) -> str:
            """
            Formatiert einen Metrikwert für die Anzeige.

            Parameter:
                value (float): Zu formatierender numerischer Metrikwert.

            Rückgabewerte:
                str: Formatierter Textwert oder „-“ bei nicht endlichen Werten.

            Fehler/Sonderfälle:
                Nicht endliche Werte werden explizit abgefangen.

            Projektkontext:
                Die Hilfsfunktion vereinheitlicht die Darstellung von Modellgütemaßen
                im Prognose-Dashboard.
            """
            return "-" if not math.isfinite(value) else f"{value:.2f}"


        metrics = result.metrics
        weights = result.weights
        if "RF MAE" in self.prediction_metric_labels:
            self.prediction_metric_labels["RF MAE"].configure(text=metric_text(metrics.get("rf_mae", math.nan)))
        if "Graph MAE" in self.prediction_metric_labels:
            self.prediction_metric_labels["Graph MAE"].configure(text=metric_text(metrics.get("gnn_mae", math.nan)))
        if "Forecast MAE" in self.prediction_metric_labels:
            self.prediction_metric_labels["Forecast MAE"].configure(text=metric_text(metrics.get("ensemble_mae", math.nan)))
        if "Gewichtung" in self.prediction_metric_labels:
            self.prediction_metric_labels["Gewichtung"].configure(
                text=f"RF {weights.get('random_forest', 0.5):.0%} | GNN {weights.get('gnn', 0.5):.0%}"
            )


    def _render_prediction_result(self, result: PredictionResult) -> None:
        """
        Rendert das textuelle Ergebnis einer Kurzfristprognose.

        Die Methode erzeugt abhängig von der Struktur des Prognoseergebnisses
        entweder eine fahrplannahe Stundenübersicht, eine abschnittsbezogene
        Haltestellenansicht oder eine aggregierte Forecast-Liste. Zusätzlich werden
        optionale Hotspot-Informationen und Top-Haltestellen dargestellt.

        Parameter:
            result (PredictionResult): Ergebnisobjekt mit Prognosedaten und
                gegebenenfalls Fahrplanempfehlungen.

        Rückgabewerte:
            None: Die Methode aktualisiert Diagramm und Textbox.

        Fehler/Sonderfälle:
            Ist die Ergebnisstruktur leer, wird nur die Ergebnisnachricht angezeigt.
            Ohne verfügbare Textbox wird zwar das Diagramm gerendert, aber keine
            Textausgabe erzeugt.

        Projektkontext:
            Die Methode transformiert numerische Prognoseergebnisse in eine fachlich
            lesbare Form für operative Entscheidungen und Stakeholder-Kommunikation.
        """
        frame = result.predictions
        self._render_prediction_chart(frame)
        if self.prediction_textbox is None:
            return
        self.prediction_textbox.configure(state="normal")
        self.prediction_textbox.delete("1.0", "end")
        if frame.empty:
            self.prediction_textbox.insert("1.0", result.message)
            self.prediction_textbox.configure(state="disabled")
            return


        schedule = result.schedule
        station_section = schedule.empty and frame["hour"].nunique() == 1
        lines = (
            ["Nr  Haltestelle                 Ein   Aus  Demand", "-" * 58]
            if station_section
            else ["Std  Aktion      Kurse  Leer  Kapazität  Ausl.", "-" * 50]
        )
        hotspot_lines: list[str] = []
        if not schedule.empty:
            for _, row in schedule.iterrows():
                hour = int(row["hour"])
                action = str(row["action"])[:9]
                runs = int(row["recommended_runs"])
                empty_runs = int(row.get("avoidable_empty_runs", 0))
                capacity = int(round(float(row.get("offered_capacity", 0))))
                utilization = float(row.get("predicted_utilization", 0))
                hotspots = str(row["hotspots"])
                lines.append(f"{hour:02d}   {action:<10} {runs:>5}  {empty_runs:>4}  {capacity:>9}  {utilization:>4.0%}")
                if hotspots:
                    hotspot_lines.append(f"{hour:02d}: {hotspots[:34]}")
        elif station_section:
            station_frame = frame.sort_values(["station_number", "station"]).copy()
            if "section_predicted_utilization" in station_frame.columns:
                first = station_frame.iloc[0]
                section_demand = int(round(float(first.get("section_predicted_demand", station_frame["pred_total"].sum()))))
                section_capacity = int(round(float(first.get("section_offered_capacity", 0))))
                section_runs = int(round(float(first.get("section_bus_runs", 0))))
                capacity_per_run = int(round(float(first.get("avg_vehicle_capacity", 0))))
                section_utilization = float(first.get("section_predicted_utilization", 0))
                lines = [
                    f"Abschnitts-Demand: {section_demand}",
                    f"Kapazität/h: {section_capacity} ({section_runs} Kurse x ca. {capacity_per_run})",
                    f"Profil-Auslastung: {section_utilization:.0%}",
                    "",
                    "Nr  Haltestelle                 Ein   Aus  Demand",
                    "-" * 58,
                ]
            for _, row in station_frame.iterrows():
                number = int(round(float(row["station_number"])))
                station = str(row["station"])[:24]
                boardings = int(round(float(row["pred_boardings"])))
                exiting = int(round(float(row["pred_exiting"])))
                total = int(round(float(row["pred_total"])))
                lines.append(f"{number:>2}  {station:<24} {boardings:>5} {exiting:>5} {total:>7}")
        else:
            hourly = frame.groupby(["date", "hour"], as_index=False).agg(pred_total=("pred_total", "sum"))
            for _, row in hourly.iterrows():
                lines.append(f"{int(row['hour']):02d}   Forecast     -      -   {int(round(float(row['pred_total']))):>6}")
        if hotspot_lines:
            lines.extend(["", "Hotspots", *hotspot_lines[:8]])
        if not station_section:
            lines.extend(["", "Top-Haltestellen"])
            for rank, (_, row) in enumerate(frame.sort_values("pred_total", ascending=False).head(8).iterrows(), start=1):
                station = str(row["station"])[:24]
                total = int(round(float(row["pred_total"])))
                lines.append(f"{rank:>2}. {int(row['hour']):02d}:00 {station:<24} {total:>6}")
        self.prediction_textbox.insert("1.0", "\n".join(lines))
        self.prediction_textbox.configure(state="disabled")


    def _render_prediction_chart(self, frame: pd.DataFrame) -> None:
        """
        Rendert das Diagramm für die Kurzfristprognose im Prognosebereich.

        Die Methode erzeugt eine Matplotlib-Figur, fügt eine Achse per
        `Figure.add_subplot(...)` hinzu und entscheidet dann abhängig von der
        Datenstruktur zwischen einer stationsbezogenen und einer stundenbezogenen
        Balkendarstellung. `add_subplot` fügt einer Figure eine Achse im Rahmen
        eines Subplot-Layouts hinzu. [web:132][web:133]

        Parameter:
            frame (pd.DataFrame): Prognose-Datenrahmen mit vorhergesagten Ein- und
                Ausstiegswerten.

        Rückgabewerte:
            None: Die Methode erzeugt und bettet ein Diagramm in die GUI ein.

        Fehler/Sonderfälle:
            Ist kein Chart-Container vorhanden, wird die Methode beendet. Bei leerem
            Frame wird statt eines Diagramms ein Hinweistext angezeigt. Der nach dem
            `return` stehende Codepfad ist in der aktuellen Implementierung nicht
            mehr erreichbar.

        Projektkontext:
            Die Methode übersetzt Prognosewerte in eine schnell erfassbare visuelle
            Darstellung für operative Entscheidungen.
        """
        if self.prediction_chart_host is None:
            return
        for child in self.prediction_chart_host.winfo_children():
            child.destroy()


        fig = Figure(figsize=(10.8, 4.1), dpi=100, facecolor=PALETTE["surface"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(PALETTE["surface"])
        if frame.empty:
            ax.text(0.5, 0.5, "Keine Kurzfrist-Prognose verfügbar", color=PALETTE["text"], ha="center", va="center")
            ax.set_axis_off()
        elif frame["hour"].nunique() == 1 and frame["station_key"].nunique() > 1:
            station_frame = frame.sort_values(["station_number", "station"]).copy()
            labels = [str(value)[:16] for value in station_frame["station"]]
            x_positions = list(range(len(station_frame)))
            ax.bar(x_positions, station_frame["pred_boardings"], color=PALETTE["blue"], alpha=0.9, label="Einstiege")
            ax.bar(
                x_positions,
                station_frame["pred_exiting"],
                bottom=station_frame["pred_boardings"],
                color=PALETTE["rose"],
                alpha=0.9,
                label="Ausstiege",
            )
            ax.set_xticks(x_positions)
            ax.set_xticklabels(labels, color=PALETTE["text"], fontsize=8, rotation=30, ha="right")
            ax.tick_params(axis="y", colors=PALETTE["muted"], labelsize=9)
            ax.set_ylabel("erwarteter Demand", color=PALETTE["muted"])
            ax.grid(axis="y", color=PALETTE["border"], alpha=0.6)
            ax.legend(facecolor=PALETTE["surface"], edgecolor=PALETTE["border"], labelcolor=PALETTE["text"])
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color(PALETTE["border"])
            ax.spines["bottom"].set_color(PALETTE["border"])
        else:
            hourly = (
                frame.groupby(["date", "hour"], as_index=False)
                .agg(pred_boardings=("pred_boardings", "sum"), pred_exiting=("pred_exiting", "sum"))
                .sort_values(["date", "hour"])
            )
            labels = [f"{int(row.hour):02d}:00" for row in hourly.itertuples()]
            x_positions = list(range(len(hourly)))
            ax.bar(x_positions, hourly["pred_boardings"], color=PALETTE["blue"], alpha=0.9, label="Einstiege")
            ax.bar(
                x_positions,
                hourly["pred_exiting"],
                bottom=hourly["pred_boardings"],
                color=PALETTE["rose"],
                alpha=0.9,
                label="Ausstiege",
            )
            ax.set_xticks(x_positions)
            ax.set_xticklabels(labels, color=PALETTE["text"], fontsize=9)
            ax.tick_params(axis="y", colors=PALETTE["muted"], labelsize=9)
            ax.set_ylabel("erwarteter Demand", color=PALETTE["muted"])
            ax.grid(axis="y", color=PALETTE["border"], alpha=0.6)
            ax.legend(facecolor=PALETTE["surface"], edgecolor=PALETTE["border"], labelcolor=PALETTE["text"])
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color(PALETTE["border"])
            ax.spines["bottom"].set_color(PALETTE["border"])
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.prediction_chart_host)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return
        if frame.empty:
            ax.text(0.5, 0.5, "Keine Prediction verfügbar", color=PALETTE["text"], ha="center", va="center")
            ax.set_axis_off()
        else:
            plot_frame = frame.head(12).iloc[::-1]
            y_positions = list(range(len(plot_frame)))
            ax.barh(y_positions, plot_frame["pred_boardings"], color=PALETTE["blue"], alpha=0.9, label="Einstiege")
            ax.barh(
                y_positions,
                plot_frame["pred_exiting"],
                left=plot_frame["pred_boardings"],
                color=PALETTE["rose"],
                alpha=0.9,
                label="Ausstiege",
            )
            ax.set_yticks(y_positions)
            ax.set_yticklabels([str(value)[:24] for value in plot_frame["station"]], color=PALETTE["text"], fontsize=9)
            ax.tick_params(axis="x", colors=PALETTE["muted"], labelsize=9)
            ax.set_xlabel("erwartete Bewegungen", color=PALETTE["muted"])
            ax.grid(axis="x", color=PALETTE["border"], alpha=0.6)
            ax.legend(facecolor=PALETTE["surface"], edgecolor=PALETTE["border"], labelcolor=PALETTE["text"])
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color(PALETTE["border"])
            ax.spines["bottom"].set_color(PALETTE["border"])
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.prediction_chart_host)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


    def _metric_card(self, parent, column: int, title: str, initial: str) -> None:
        """
        Erzeugt eine KPI-Karte für den Dashboard-Kopfbereich.

        Die Methode erstellt ein visuell hervorgehobenes Karten-Widget mit Titel und
        initialem Wert und speichert das zugehörige Wertelabel in einer Lookup-
        Struktur zur späteren Aktualisierung.

        Parameter:
            parent: Übergeordnetes Container-Widget.
            column (int): Zielspalte im Grid-Layout.
            title (str): Titel der Kennzahl.
            initial (str): Initial anzuzeigender Wert.

        Rückgabewerte:
            None: Die Methode erzeugt GUI-Elemente und registriert das Wertelabel.

        Fehler/Sonderfälle:
            Keine explizite Fehlerbehandlung; ein gültiger Parent-Container wird
            vorausgesetzt.

        Projektkontext:
            Die Methode standardisiert die Darstellung zentraler Kennzahlen im
            Dashboard und erleichtert spätere Aktualisierungen.
        """
        card = ctk.CTkFrame(
            parent,
            fg_color=PALETTE["surface"],
            corner_radius=24,
            border_width=1,
            border_color=PALETTE["border"],
            height=138,
        )
        card.grid(row=0, column=column, sticky="ew", padx=6)
        card.grid_propagate(False)
        ctk.CTkLabel(
            card,
            text=title,
            text_color=PALETTE["muted"],
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        ).pack(anchor="w", padx=18, pady=(16, 8))
        value = ctk.CTkLabel(
            card,
            text=initial,
            text_color=PALETTE["text"],
            font=ctk.CTkFont(family="Bahnschrift", size=26, weight="bold"),
        )
        value.pack(anchor="w", padx=18, pady=(0, 12))
        self.metric_cards[title] = value


    def _card(self, parent, row: int, title: str, subtitle: str, column: int = 0, padx: tuple[int, int] = (0, 0)) -> ctk.CTkFrame:
        """
        Erzeugt eine standardisierte Inhaltskarte für Dashboard-Bereiche.

        Die Methode erstellt einen einheitlich gestalteten Container mit
        Rahmenstil und Titelbeschriftung, der in verschiedenen Ansichtsbereichen
        wiederverwendet werden kann.

        Parameter:
            parent: Übergeordnetes Container-Widget.
            row (int): Zielzeile im Grid-Layout.
            title (str): Überschrift der Karte.
            subtitle (str): Untertitel oder fachlicher Kontext; in der aktuellen
                Implementierung wird er nicht weiter verwendet.
            column (int): Zielspalte im Grid-Layout.
            padx (tuple[int, int]): Horizontaler Außenabstand der Karte.

        Rückgabewerte:
            ctk.CTkFrame: Erzeugter Kartencontainer.

        Fehler/Sonderfälle:
            Der Parameter subtitle ist aktuell ungenutzt, bleibt aber Teil der
            Schnittstelle.

        Projektkontext:
            Die Methode schafft ein konsistentes visuelles Grundmuster für
            Dashboard-, Karten- und Analysebereiche.
        """
        card = ctk.CTkFrame(
            parent,
            fg_color=PALETTE["surface"],
            corner_radius=28,
            border_width=1,
            border_color=PALETTE["border"],
        )
        card.grid(row=row, column=column, sticky="nsew", pady=(0, 12), padx=padx)
        ctk.CTkLabel(
            card,
            text=title,
            text_color=PALETTE["text"],
            width=1,
            anchor="w",
            justify="left",
            font=ctk.CTkFont(family="Bahnschrift", size=24, weight="bold"),
        ).pack(anchor="w", fill="x", padx=16, pady=(16, 12))
        return card


    def _inline_insight_label(self, parent: ctk.CTkFrame) -> ctk.CTkLabel:
        """
        Erzeugt ein standardisiertes Insight-Label innerhalb einer Karte.

        Die Methode erstellt ein formatiertes Label mit Platzhaltertext, das später
        mit automatisch berechneten analytischen Erkenntnissen überschrieben werden kann.

        Parameter:
            parent (ctk.CTkFrame): Übergeordneter Kartencontainer.

        Rückgabewerte:
            ctk.CTkLabel: Erzeugtes Insight-Label.

        Fehler/Sonderfälle:
            Keine explizite Fehlerbehandlung; ein gültiger Parent-Container wird
            vorausgesetzt.

        Projektkontext:
            Die Methode unterstützt die einheitliche Darstellung interpretierender
            Analysehinweise im Dashboard.
        """
        label = ctk.CTkLabel(
            parent,
            text="Insight wird nach der Analyse berechnet.",
            fg_color=PALETTE["surface_alt"],
            corner_radius=18,
            text_color=PALETTE["text"],
            wraplength=1180,
            justify="left",
            anchor="w",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
        )
        label.pack(fill="x", padx=12, pady=(0, 14), ipady=10)
        return label


    def _refresh_line_menu(self) -> None:
        """
        Aktualisiert die allgemeine Linienauswahl im Dashboard.

        Die Methode erzeugt aus den im Repository verfügbaren Linien eine Liste
        darstellbarer Labels, setzt diese in die Auswahlkomponente und synchronisiert
        anschließend die sichtbaren Linien-Chips.

        Parameter:
            Keine.

        Rückgabewerte:
            None: Die Methode aktualisiert GUI-Auswahl und Linien-Chips.

        Fehler/Sonderfälle:
            Sind keine Linien verfügbar, wird ein Platzhaltertext verwendet. Eine
            nicht mehr gültige Auswahl wird zurückgesetzt.

        Projektkontext:
            Die Methode hält die zentrale Linienauswahl konsistent mit der
            verfügbaren Datenbasis des Repositories.
        """
        labels = [self.repo.label_for_line(line) for line in self.repo.available_lines] or ["Keine Linien gefunden"]
        if self.line_list is not None:
            self.line_list.set_items(labels)
        if self.line_choice.get() not in labels:
            self.line_choice.set("")
        self.render_line_chips()


    def _refresh_flex_line_menu(self) -> int | None:
        """
        Aktualisiert die Linienauswahl für das flexible Vergleichsdiagramm.

        Die Methode baut eine Zuordnung zwischen sichtbaren Linienlabels und internen
        Liniennummern auf und setzt einen gültigen Standardwert für die Auswahl des
        Hauptdiagramms.

        Parameter:
            Keine.

        Rückgabewerte:
            int | None: Die aktuell ausgewählte Hauptlinie oder None, falls keine
            gültige Auswahl vorliegt.

        Fehler/Sonderfälle:
            Sind keine Linien aktiv, wird ein Platzhalterwert verwendet.

        Projektkontext:
            Die Methode verknüpft die globale Linienselektion mit der flexiblen,
            dimensionsabhängigen Vergleichsanalyse.
        """
        self.flex_line_lookup = {}
        if not self.selected_lines:
            values = ["Keine Linie"]
        else:
            values = [self.repo.label_for_line(line) for line in self.selected_lines]
            self.flex_line_lookup = {self.repo.label_for_line(line): line for line in self.selected_lines}


        if self.flex_line_menu is not None:
            self.flex_line_menu.configure(values=values)
        if self.flex_line_choice.get() not in self.flex_line_lookup:
            self.flex_line_choice.set(values[0])
        return self.flex_line_lookup.get(self.flex_line_choice.get())


    def _refresh_flex_compare_menu(self, main_line: int | None) -> int | None:
        """
        Aktualisiert die Vergleichslinienauswahl für das flexible Diagramm.

        Die Methode erstellt eine Liste potenzieller Vergleichslinien, wobei die
        aktuell als Hauptlinie ausgewählte Linie ausgeschlossen wird.

        Parameter:
            main_line (int | None): Derzeit ausgewählte Hauptlinie des Diagramms.

        Rückgabewerte:
            int | None: Gewählte Vergleichslinie oder None, falls keine gültige
            Vergleichsauswahl vorliegt.

        Fehler/Sonderfälle:
            Ist keine Vergleichslinie ausgewählt, bleibt der Standardwert
            „Kein Vergleich“ aktiv.

        Projektkontext:
            Die Methode ermöglicht kontrastive Linienanalysen innerhalb derselben
            flexiblen Visualisierung.
        """
        self.flex_compare_lookup = {}
        values = ["Kein Vergleich"]
        for line in self.selected_lines:
            if line == main_line:
                continue
            label = self.repo.label_for_line(line)
            values.append(label)
            self.flex_compare_lookup[label] = line


        if self.flex_compare_menu is not None:
            self.flex_compare_menu.configure(values=values)
        if self.flex_compare_choice.get() not in self.flex_compare_lookup:
            self.flex_compare_choice.set("Kein Vergleich")
        return self.flex_compare_lookup.get(self.flex_compare_choice.get())


    def _schedule_dashboard_update(self, _event=None) -> None:
        """
        Plant eine verzögerte Aktualisierung des Dashboards.

        Die Methode verwirft zunächst einen eventuell bereits geplanten Update-Callback
        und setzt danach einen neuen zeitverzögerten Aufruf von `update_dashboard`.
        Dadurch werden schnelle Folgeänderungen entprellt.

        Parameter:
            _event: Optionales GUI-Ereignisobjekt; wird inhaltlich nicht verwendet.

        Rückgabewerte:
            None: Die Methode registriert einen späteren Update-Aufruf.

        Fehler/Sonderfälle:
            Ein bereits vorhandener Callback wird vor dem Neuplanen abgebrochen.

        Projektkontext:
            Die Methode reduziert unnötige Mehrfachberechnungen bei schneller
            Benutzerinteraktion, etwa bei Datums- oder Filteränderungen.
        """
        if self.dashboard_update_after_id is not None:
            self.after_cancel(self.dashboard_update_after_id)
        self.dashboard_update_after_id = self.after(180, self.update_dashboard)


    def _select_line_from_list(self, label: str, _index: int) -> None:
        """
        Übernimmt eine Linienauswahl aus der animierten Liste in die aktive Selektion.

        Die Methode setzt zunächst den ausgewählten Labeltext als aktuelle Linienwahl
        und stößt danach das Hinzufügen dieser Linie zur aktiven Auswahl an.

        Parameter:
            label (str): Ausgewähltes Linienlabel.
            _index (int): Positionsindex in der Liste; wird fachlich nicht verwendet.

        Rückgabewerte:
            None: Die Methode aktualisiert den Auswahlzustand.

        Fehler/Sonderfälle:
            Keine spezielle Fehlerbehandlung; die weitere Prüfung erfolgt in
            `add_selected_line`.

        Projektkontext:
            Die Methode verbindet die Listeninteraktion des Nutzers mit der
            aktiven Dashboard-Selektion.
        """
        self.line_choice.set(label)
        self.add_selected_line()


    def _set_sidebar_active(self, key: str) -> None:
        """
        Markiert den aktuell aktiven Bereich in der Sidebar visuell.

        Die Methode übersetzt interne View-Schlüssel in die zugehörigen Sidebar-
        Einträge und passt Farben, Hover-Verhalten und Schriftgewicht der Buttons an.

        Parameter:
            key (str): Interner Schlüssel der aktiven Ansicht oder Sektion.

        Rückgabewerte:
            None: Die Methode aktualisiert ausschließlich die Sidebar-Gestaltung.

        Fehler/Sonderfälle:
            Unbekannte Schlüssel führen dazu, dass kein Button als aktiv markiert wird.

        Projektkontext:
            Die Methode unterstützt die Orientierung innerhalb der Multi-View-
            Anwendung durch konsistente visuelle Navigation.
        """
        active_map = {
            "dashboard": "dashboard",
            "map": "kartenansicht",
            "prediction": "prediction",
            "timetable": "fahrplan",
            "kpi": "kpi vergleich",
        }
        selected = active_map.get(key, "")
        for name, button in self.sidebar_buttons.items():
            is_active = name == selected
            button.configure(
                fg_color=PALETTE["sidebar_active"] if is_active else PALETTE["accent_soft"],
                hover_color=PALETTE["accent_dark"] if is_active else PALETTE["surface_alt"],
                text_color="white" if is_active else PALETTE["text"],
                font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold" if is_active else "normal"),
            )


    def _show_view(self, view: str) -> None:
        """
        Wechselt zwischen den Hauptansichten der Anwendung.

        Die Methode blendet je nach gewünschter Ansicht die relevanten Frames ein,
        versteckt nicht benötigte Bereiche, aktualisiert die Sidebar-Markierung und
        stößt ansichtsspezifische Initialisierungen wie Datenladen oder Menürefreshs an.

        Parameter:
            view (str): Zielansicht, etwa „map“, „prediction“, „timetable“, „kpi“
                oder das Standard-Dashboard.

        Rückgabewerte:
            None: Die Methode aktualisiert ausschließlich den GUI-Zustand.

        Fehler/Sonderfälle:
            Nicht explizit behandelte Werte führen zur Standard-Dashboardansicht.
            Beim Wechsel in andere Ansichten wird eine laufende Kartenanimation
            gegebenenfalls pausiert.

        Projektkontext:
            Die Methode ist die zentrale Navigationslogik zwischen Analyse,
            Kartenansicht, Prognose, Fahrplanvergleich und KPI-Bereich.
        """
        if view == "map":
            if self.filter_shell is not None:
                self.filter_shell.grid_remove()
            if self.line_chips_panel is not None:
                self.line_chips_panel.grid_remove()
            self.dashboard_frame.grid_remove()
            self.prediction_frame.grid_remove()
            self.timetable_frame.grid_remove()
            self.kpi_frame.grid_remove()
            self.map_frame.grid()
            self._set_sidebar_active("map")
            self._focus_wuerzburg_map()
            if self.map_animation_data.empty:
                self._refresh_map_animation_data(silent=True)
            return


        if view == "prediction":
            if self.filter_shell is not None:
                self.filter_shell.grid_remove()
            if self.line_chips_panel is not None:
                self.line_chips_panel.grid_remove()
            self.dashboard_frame.grid_remove()
            self.map_frame.grid_remove()
            self.timetable_frame.grid_remove()
            self.kpi_frame.grid_remove()
            self.prediction_frame.grid()
            self._set_sidebar_active("prediction")
            self._pause_map_animation()
            self._refresh_prediction_line_menu()
            return


        if view == "timetable":
            if self.filter_shell is not None:
                self.filter_shell.grid_remove()
            if self.line_chips_panel is not None:
                self.line_chips_panel.grid_remove()
            self.dashboard_frame.grid_remove()
            self.map_frame.grid_remove()
            self.prediction_frame.grid_remove()
            self.kpi_frame.grid_remove()
            self.timetable_frame.grid()
            self._set_sidebar_active("timetable")
            self._pause_map_animation()
            self._refresh_timetable_line_menu()
            self._calculate_timetable_comparison()
            return


        if view == "kpi":
            if self.filter_shell is not None:
                self.filter_shell.grid_remove()
            if self.line_chips_panel is not None:
                self.line_chips_panel.grid_remove()
            self.dashboard_frame.grid_remove()
            self.map_frame.grid_remove()
            self.prediction_frame.grid_remove()
            self.timetable_frame.grid_remove()
            self.kpi_frame.grid()
            self._set_sidebar_active("kpi")
            self._pause_map_animation()
            self._refresh_kpi_line_menu()
            if not self.kpi_annual_cache and not self.kpi_busy:
                self.after(80, lambda: self._calculate_kpi_comparison(scope="total"))
            return


        if self.filter_shell is not None:
            self.filter_shell.grid()
        if self.line_chips_panel is not None:
            self.line_chips_panel.grid()
        self.map_frame.grid_remove()
        self.prediction_frame.grid_remove()
        self.timetable_frame.grid_remove()
        self.kpi_frame.grid_remove()
        self.dashboard_frame.grid()
        self._set_sidebar_active("dashboard")
        self._pause_map_animation()


    def _go_to_section(self, section: str) -> None:
        """
        Navigiert innerhalb des Dashboards zu einer bestimmten Sektion.

        Die Methode markiert den zugehörigen Sidebar-Eintrag aktiv, scrollt mit
        leichter Verzögerung zur Zielkomponente und setzt im Fall des Kalender-
        Bereichs zusätzlich den Fokus auf das Startdatumsfeld.

        Parameter:
            section (str): Schlüssel der Zielsektion.

        Rückgabewerte:
            None: Die Methode verändert Navigations- und Fokuszustände.

        Fehler/Sonderfälle:
            Existiert kein Zielwidget für die Sektion, wird nur die Sidebar-Markierung
            aktualisiert.

        Projektkontext:
            Die Methode unterstützt die schnelle Navigation innerhalb eines
            scrollbaren analytischen Dashboards.
        """
        self._set_sidebar_active(section)
        target = self.section_targets.get(section)
        if target is not None:
            self.after(50, lambda: self._scroll_to_widget(target))
        if section == "kalender":
            self.start_picker.entry.focus_set()


    def _scroll_to_widget(self, widget) -> None:
        """
        Scrollt den Hauptbereich so, dass ein Zielwidget sichtbar wird.

        Die Methode ermittelt aus der y-Position des Widgets eine relative
        Scrollposition und verschiebt den zugrunde liegenden Canvas entsprechend.

        Parameter:
            widget: Ziel-Widget, zu dem gescrollt werden soll.

        Rückgabewerte:
            None: Die Methode steuert ausschließlich den Scrollzustand.

        Fehler/Sonderfälle:
            Ist der Canvas nicht verfügbar oder tritt ein Fehler beim Zugriff auf den
            Scrollcontainer auf, wird die Methode beendet.

        Projektkontext:
            Die Methode realisiert die inhaltliche Navigation zu Analyseabschnitten
            innerhalb des scrollbaren Hauptlayouts.
        """
        try:
            canvas = self.main_scroll._parent_canvas
        except Exception:
            return
        self.update_idletasks()
        total_height = max(canvas.bbox("all")[3], 1) if canvas.bbox("all") else 1
        y = max(widget.winfo_y() - 16, 0)
        fraction = min(max(y / total_height, 0), 1)
        canvas.yview_moveto(fraction)


    def render_line_chips(self) -> None:
        """
        Rendert die aktuell ausgewählten Linien als entfernbare Chips.

        Die Methode leert zunächst den Chip-Container und erzeugt anschließend für
        jede aktive Linie ein eigenes Chip-Element mit Label und Entfernen-Button.

        Parameter:
            Keine.

        Rückgabewerte:
            None: Die Methode erzeugt und aktualisiert GUI-Elemente.

        Fehler/Sonderfälle:
            Ist kein Chip-Container vorhanden, erfolgt keine Aktion.

        Projektkontext:
            Die Methode macht die Mehrfachauswahl aktiver Linien visuell transparent
            und direkt bearbeitbar.
        """
        if self.line_chip_frame is None:
            return
        for child in self.line_chip_frame.winfo_children():
            child.destroy()
        for line in self.selected_lines:
            chip = ctk.CTkFrame(
                self.line_chip_frame,
                fg_color=PALETTE["chip"],
                corner_radius=16,
                border_width=1,
                border_color=PALETTE["border"],
            )
            chip.pack(side="left", padx=(0, 10))
            ctk.CTkLabel(
                chip,
                text=self.repo.label_for_line(line),
                text_color=PALETTE["text"],
                font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            ).pack(side="left", padx=(12, 6), pady=8)
            ctk.CTkButton(
                chip,
                text="x",
                width=24,
                height=24,
                corner_radius=12,
                fg_color=PALETTE["surface_alt"],
                hover_color=PALETTE["accent_soft"],
                text_color=PALETTE["muted"],
                command=lambda selected=line: self.remove_line(selected),
            ).pack(side="left", padx=(0, 8), pady=6)


    def add_selected_line(self) -> None:
        """
        Fügt die aktuell ausgewählte Linie zur aktiven Analyseauswahl hinzu.

        Die Methode sucht die ausgewählte Bezeichnung in den verfügbaren Linien,
        prüft auf Dubletten und aktualisiert danach Linien-Chips, Flottenplan,
        Prognosemenüs und Dashboard.

        Parameter:
            Keine.

        Rückgabewerte:
            None: Die Methode verändert Auswahlzustand und Folgeansichten.

        Fehler/Sonderfälle:
            Leere Auswahl oder bereits aktive Linien werden mit einer passenden
            Statusmeldung abgefangen. Wird die Linie nicht in den Daten gefunden,
            erfolgt ebenfalls ein Hinweis.

        Projektkontext:
            Die Methode ist der zentrale Einstieg für die dynamische Erweiterung des
            aktuellen Analysefokus auf mehrere Linien.
        """
        selected = self.line_choice.get()
        if not selected:
            self.status_text.set("Bitte zuerst eine Linie aus der animierten Liste auswählen.")
            return
        for line in self.repo.available_lines:
            if self.repo.label_for_line(line) == selected:
                if line in self.selected_lines:
                    self.status_text.set(f"{self.repo.label_for_line(line)} ist schon aktiv.")
                    return
                self.selected_lines.append(line)
                self.selected_lines.sort()
                self.render_line_chips()
                self._refresh_fleet_schedule()
                self._refresh_prediction_line_menu()
                self.status_text.set(f"{self.repo.label_for_line(line)} wurde hinzugefügt.")
                self.update_dashboard()
                return
        self.status_text.set("Diese Linie wurde in den Daten nicht gefunden.")


    def remove_line(self, line: int) -> None:
        """
        Entfernt eine Linie aus der aktiven Analyseauswahl.

        Die Methode verhindert, dass die letzte verbleibende Linie entfernt wird,
        aktualisiert sonst die aktiven Linien, die zugehörigen GUI-Komponenten und
        stößt anschließend ein Dashboard-Update an.

        Parameter:
            line (int): Zu entfernende Liniennummer.

        Rückgabewerte:
            None: Die Methode verändert Auswahl- und Anzeigezustände.

        Fehler/Sonderfälle:
            Die letzte verbleibende aktive Linie darf nicht entfernt werden und wird
            per Hinweisdialog geschützt.

        Projektkontext:
            Die Methode ermöglicht die kontrollierte Reduktion des Analysefokus, ohne
            die Anwendung in einen leeren Zustand zu überführen.
        """
        if len(self.selected_lines) == 1:
            mb.showinfo("Hinweis", "Mindestens eine Linie sollte aktiv bleiben.")
            return
        self.selected_lines = [item for item in self.selected_lines if item != line]
        self.render_line_chips()
        self._refresh_fleet_schedule()
        self._refresh_prediction_line_menu()
        self.status_text.set(f"{self.repo.label_for_line(line)} wurde entfernt.")
        self.update_dashboard()


    def update_dashboard(self) -> None:
        """
        Aktualisiert das gesamte Dashboard auf Basis der aktuellen Linien- und Datumswahl.

        Die Methode prüft zunächst den Datumsbereich und die aktive Linienselektion,
        lädt dann aggregierte Daten für Tages-, Stunden- und Haltestellenebene und
        aktualisiert anschließend Kennzahlen, Diagramme, Insights, Stationspanel
        sowie die Kartenanimation.

        Parameter:
            Keine.

        Rückgabewerte:
            None: Die Methode aktualisiert zahlreiche GUI-Komponenten und interne Daten.

        Fehler/Sonderfälle:
            Ein ungültiger Datumsbereich wird automatisch korrigiert. Sind keine
            Linien aktiv oder liefert die Aggregation keine Daten, werden leere
            Zustände gerendert und Kennzahlen auf 0 gesetzt.

        Projektkontext:
            Diese Methode ist die zentrale Orchestrierungsfunktion des Analyse-
            Dashboards und bündelt die Aktualisierung nahezu aller datengetriebenen
            Oberflächenelemente.
        """
        self.dashboard_update_after_id = None
        start = self.start_picker.get_date()
        end = self.end_picker.get_date()
        if start > end:
            self.end_picker.set_date(start)
            end = start
            self.status_text.set("Enddatum wurde automatisch an das Startdatum angepasst.")
        if not self.selected_lines:
            mb.showinfo("Hinweis", "Bitte mindestens eine Linie auswählen.")
            return


        if self.flex_x_choice.get() not in FLEX_X_OPTIONS:
            self.flex_x_choice.set(FLEX_X_OPTIONS[0])
        if self.flex_y_choice.get() not in FLEX_Y_OPTIONS:
            self.flex_y_choice.set(FLEX_Y_OPTIONS[0])


        daily, hourly, stations = self.repo.aggregate_selection(self.selected_lines, start, end)
        line_filter = self._refresh_flex_line_menu()
        compare_line_filter = self._refresh_flex_compare_menu(line_filter)
        flex_data = self.repo.aggregate_flexible_chart(
            self.selected_lines,
            start,
            end,
            self.flex_x_choice.get(),
            self.flex_y_choice.get(),
            line_filter=line_filter,
        )
        compare_data = None
        if compare_line_filter is not None:
            compare_data = self.repo.aggregate_flexible_chart(
                self.selected_lines,
                start,
                end,
                self.flex_x_choice.get(),
                self.flex_y_choice.get(),
                line_filter=compare_line_filter,
            )
        if daily.empty:
            self.status_text.set("Keine Daten für die aktuelle Auswahl gefunden.")
            self._set_metric("Gesamt Einstiege", "0")
            self._set_metric("Tagesdurchschnitt", "0")
            self._set_metric("Peak Day", "0")
            self._set_metric("Aktive Fahrten", "0")
            self._render_empty_states()
            self.map_animation_data = pd.DataFrame()
            self._render_map_hour_frame()
            return


        total_boardings = int(daily["boardings"].sum())
        avg_daily = int(daily.groupby("date")["boardings"].sum().mean())
        peak_day = int(daily.groupby("date")["boardings"].sum().max())
        total_journeys = int(daily["journeys"].sum())


        self._set_metric("Gesamt Einstiege", f"{total_boardings:,}".replace(",", "."))
        self._set_metric("Tagesdurchschnitt", f"{avg_daily:,}".replace(",", "."))
        self._set_metric("Peak Day", f"{peak_day:,}".replace(",", "."))
        self._set_metric("Aktive Fahrten", f"{total_journeys:,}".replace(",", "."))


        self.status_text.set(
            f"{len(self.selected_lines)} Linien | {start.isoformat()} bis {end.isoformat()} | {total_boardings:,} Einstiege".replace(",", ".")
        )
        self._draw_daily_chart(daily)
        self._draw_flexible_chart(
            flex_data,
            self.flex_x_choice.get(),
            self.flex_y_choice.get(),
            compare_data=compare_data,
            main_label=self.flex_line_choice.get(),
            compare_label=self.flex_compare_choice.get(),
        )
        self._update_chart_insights(
            daily,
            hourly,
            flex_data,
            self.flex_x_choice.get(),
            self.flex_y_choice.get(),
            compare_data=compare_data,
            main_label=self.flex_line_choice.get(),
            compare_label=self.flex_compare_choice.get(),
        )
        self._fill_station_panel(stations)
        self._fill_insights(daily, hourly, stations)
        self._refresh_map_animation_data(silent=True)


    def _daily_totals(self, daily: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregiert Tagesdaten auf Datumsebene für weiterführende Analysen.

        Die Methode definiert eine Reihe fachlich relevanter Aggregationsregeln und
        wendet diese nur auf tatsächlich vorhandene Spalten an. Tagessummen,
        Ereignisindikatoren und maximale Eventmerkmale werden so in einer
        konsolidierten Tagesstruktur zusammengeführt. Pandas `groupby(...).agg(...)`
        ermöglicht dabei die gleichzeitige Anwendung mehrerer Aggregationsfunktionen
        auf unterschiedliche Spalten. [web:125][web:116]

        Parameter:
            daily (pd.DataFrame): Tagesnahe Roh- oder Voraggregationsdaten.

        Rückgabewerte:
            pd.DataFrame: Nach Datum aggregierter und sortierter Tagesdatenrahmen.

        Fehler/Sonderfälle:
            Nicht vorhandene Spalten werden automatisch aus der Aggregation
            ausgeschlossen, sodass die Methode robust gegenüber variierenden
            Datenschemata bleibt.

        Projektkontext:
            Die Methode liefert eine standardisierte Tagesbasis für Kennzahlen,
            Insights und ereignisbezogene Auswertungen.
        """
        aggregations = {
            "boardings": "sum",
            "journeys": "sum",
            "lecture_period_jmu": "max",
            "public_holiday": "max",
            "school_holiday": "max",
            "event_hours": "max",
            "concert_hours": "max",
            "event_count": "max",
            "concert_event_count": "max",
            "total_event_duration_hours": "max",
            "max_event_duration_hours": "max",
        }
        available = {key: value for key, value in aggregations.items() if key in daily.columns}
        return daily.groupby("date", as_index=False).agg(available).sort_values("date")


    def _format_insight_value(self, value: float, metric: str = "Einstiege") -> str:
        """
        Formatiert numerische Insight-Werte für die textuelle Ausgabe.

        Die Methode unterscheidet zwischen Auslastungsmetriken mit Dezimaldarstellung
        und ganzzahligen Zählwerten mit Tausendertrennung.

        Parameter:
            value (float): Zu formatierender Kennzahlenwert.
            metric (str): Fachliche Kennzahlart zur Bestimmung des Ausgabeformats.

        Rückgabewerte:
            str: Formatierter Textwert für die Insight-Ausgabe.

        Fehler/Sonderfälle:
            Fehlende Werte werden als „0“ dargestellt.

        Projektkontext:
            Die Methode sorgt für eine konsistente und fachlich passende
            Darstellung interpretierender Kennzahlen im Dashboard.
        """
        if pd.isna(value):
            return "0"
        if metric in {"Peak-Auslastung", "Ø Auslastung"}:
            return f"{float(value):.1f}".replace(".", ",")
        return f"{int(round(float(value))):,}".replace(",", ".")


    def _metric_unit(self, metric: str) -> str:
        """
        Liefert die sprachlich passende Einheit zu einer fachlichen Kennzahl.

        Die Methode ordnet internen Metrikbezeichnungen eine für erläuternde
        Insight-Texte geeignete Einheit oder Formulierung zu.

        Parameter:
            metric (str): Bezeichnung der Kennzahl.

        Rückgabewerte:
            str: Passende sprachliche Einheit oder Sammelbezeichnung.

        Fehler/Sonderfälle:
            Für unbekannte Metriken wird die generische Einheit „Werten“ verwendet.

        Projektkontext:
            Die Methode unterstützt natürlich lesbare analytische Erläuterungen im
            Dashboard und in automatisiert erzeugten Insights.
        """
        return {
            "Einstiege": "Einstiegen",
            "Ausstiege": "Ausstiegen",
            "Ein- und Ausstiege": "Bewegungen",
            "Fahrten": "Fahrten",
            "Peak-Auslastung": "Auslastung",
            "Ø Auslastung": "Auslastung",
        }.get(metric, "Werten")

    def _metric_series(self, data: pd.DataFrame, metric: str) -> pd.Series:
        """
        Ermittelt die darzustellende Kennzahlenserie für eine flexible Analyse.

        Die Methode leitet aus dem übergebenen Datenrahmen die fachlich passende
        Zielserie ab. Für die kombinierte Metrik „Ein- und Ausstiege“ werden
        Einstiege und Ausstiege addiert, andernfalls wird die allgemeine Spalte
        `value` als metrische Serie verwendet.

        Parameter:
            data (pd.DataFrame): Datenbasis der flexiblen Analyse.
            metric (str): Name der auszuwertenden Kennzahl.

        Rückgabewerte:
            pd.Series: Numerische Serie zur weiteren Analyse oder Visualisierung.

        Fehler/Sonderfälle:
            Bei leerem Datenrahmen wird eine leere Float-Serie zurückgegeben.

        Projektkontext:
            Die Methode abstrahiert die Metriklogik für flexible Diagramme und
            Insight-Berechnungen, sodass unterschiedliche Kennzahltypen einheitlich
            weiterverarbeitet werden können.
        """
        if data.empty:
            return pd.Series(dtype=float)
        if metric == "Ein- und Ausstiege":
            return data["boardings"].astype(float) + data["exiting"].astype(float)
        return data["value"].astype(float)


    def _trend_sentence(self, first_value: float, last_value: float, subject: str) -> str:
        """
        Formuliert eine kurze sprachliche Trendbeschreibung zwischen zwei Vergleichswerten.

        Die Methode berechnet die relative Veränderung zwischen einem Anfangs- und
        Endwert und übersetzt sie in eine verständliche textuelle Einschätzung wie
        „stabil“, „steigt“ oder „fällt“.

        Parameter:
            first_value (float): Referenzwert zu Beginn des Vergleichsfensters.
            last_value (float): Vergleichswert am Ende des Fensters.
            subject (str): Fachlicher Bezug der Aussage, etwa „Die Tagesnachfrage“.

        Rückgabewerte:
            str: Natürlichsprachlicher Satz zur Trendbeschreibung.

        Fehler/Sonderfälle:
            Ist der Startwert kleiner oder gleich null, wird kein Prozentvergleich
            berechnet, sondern eine vorsichtige Formulierung zur schlechten
            Einschätzbarkeit zurückgegeben. Kleine Änderungen unter 6 Prozent werden
            als stabil interpretiert.

        Projektkontext:
            Die Methode bildet die sprachliche Grundlage für automatisch erzeugte
            Insights und soll numerische Trends auch für fachliche Stakeholder ohne
            Rohdatenlektüre verständlich machen.
        """
        if first_value <= 0:
            return f"{subject} bleibt im Vergleichsfenster schwer einschätzbar."
        delta = ((last_value - first_value) / first_value) * 100
        if abs(delta) < 6:
            return f"{subject} bleibt insgesamt stabil."
        direction = "steigt" if delta > 0 else "fällt"
        return f"{subject} {direction} um {abs(delta):.0f}%."


    def _build_daily_chart_insight(self, daily: pd.DataFrame, hourly: pd.DataFrame) -> str:
        """
        Erzeugt einen automatisch formulierten Insight-Text für das Tagesdiagramm.

        Die Methode verdichtet Tages- und Stundendaten zu einer kurzen fachlichen
        Zusammenfassung mit Peak-Tag, führender Linie, stärkster Stunde und optional
        einer Trendinterpretation über das betrachtete Zeitfenster.

        Parameter:
            daily (pd.DataFrame): Tagesbezogene Nachfrage- und Linieninformationen.
            hourly (pd.DataFrame): Stundenbezogene Nachfrageinformationen.

        Rückgabewerte:
            str: Formulierter Insight-Text für die Tagesanalyse.

        Fehler/Sonderfälle:
            Liegen keine Tagesdaten vor, wird eine standardisierte Leermeldung
            zurückgegeben. Eine Trendaussage wird erst bei ausreichend vielen
            Tageswerten erzeugt.

        Projektkontext:
            Die Methode übersetzt aggregierte Nachfrageverläufe in eine kompakte
            Management- bzw. Professoren-taugliche Interpretation des Tagesverlaufs.
        """
        daily_totals = self._daily_totals(daily)
        if daily_totals.empty:
            return "Insight: Keine Tagesdaten für die aktuelle Auswahl verfügbar."


        totals = daily_totals.set_index("date")["boardings"].astype(float).sort_index()
        peak_date = totals.idxmax()
        peak_value = totals.max()
        leader_line = int(daily.groupby("line")["boardings"].sum().idxmax())
        parts = [
            f"Insight: Peak am {peak_date.isoformat()} mit {self._format_insight_value(peak_value)} Einstiegen.",
            f"Führende Linie ist {self.repo.label_for_line(leader_line).split('|')[0].strip()}.",
        ]


        hour_totals = hourly.groupby("hour")["boardings"].sum()
        if not hour_totals.empty:
            peak_hour = int(hour_totals.idxmax())
            parts.append(f"Stärkste Stunde liegt bei {peak_hour:02d}:00 Uhr.")


        if len(totals) >= 10:
            window = min(14, max(3, len(totals) // 4))
            parts.append(self._trend_sentence(float(totals.head(window).mean()), float(totals.tail(window).mean()), "Die Tagesnachfrage"))
        return " ".join(parts)


    def _build_flexible_chart_insight(
        self,
        data: pd.DataFrame,
        x_axis: str,
        y_metric: str,
        compare_data: pd.DataFrame | None = None,
        main_label: str = "",
        compare_label: str = "",
    ) -> str:
        """
        Erzeugt einen automatisch formulierten Insight-Text für das flexible Diagramm.

        Die Methode analysiert das aktuell dargestellte flexible Diagramm abhängig von
        Achsentyp und Kennzahl. Sie beschreibt dabei Peaks, zeitliche Veränderungen,
        besonders starke Haltestellen oder Wochentage sowie optional den Unterschied
        zu einer Vergleichslinie.

        Parameter:
            data (pd.DataFrame): Hauptdaten der flexiblen Analyse.
            x_axis (str): Gewählte Ausprägung der x-Achse, z. B. Stunde, Datum oder Haltestelle.
            y_metric (str): Gewählte Zielkennzahl.
            compare_data (pd.DataFrame | None): Optionale Vergleichsdaten einer zweiten Linie.
            main_label (str): Anzeigename der Hauptauswahl.
            compare_label (str): Anzeigename der Vergleichsauswahl.

        Rückgabewerte:
            str: Formulierter Insight-Text zur flexiblen Analyse.

        Fehler/Sonderfälle:
            Bei leerer Datenbasis oder leerer Metrikserie wird eine passende
            Standardmeldung zurückgegeben. Vergleichsaussagen werden nur erzeugt,
            wenn tatsächlich verwertbare Vergleichsdaten vorliegen.

        Projektkontext:
            Die Methode ermöglicht eine automatisch erzeugte fachliche Interpretation
            beliebiger Diagrammkombinationen, ohne für jede Ansicht eigene
            Erklärungstexte manuell pflegen zu müssen.
        """
        if data.empty:
            return "Insight: Keine flexible Analyse für die aktuelle Auswahl verfügbar."


        series = self._metric_series(data, y_metric)
        if series.empty:
            return "Insight: Keine Werte für die aktuelle Kombination verfügbar."


        main_name = main_label.split("|")[0].strip() or "Auswahl"
        peak_index = int(series.idxmax())
        peak_label = str(data.iloc[peak_index]["x_label"])
        peak_value = float(series.iloc[peak_index])
        parts = [
            f"Insight: {main_name} hat den Peak bei {peak_label} mit "
            f"{self._format_insight_value(peak_value, y_metric)} {self._metric_unit(y_metric)}."
        ]


        if x_axis == "Stunde":
            by_hour = data.assign(metric_value=series).set_index("x_value")["metric_value"]
            if 7 in by_hour.index and 9 in by_hour.index:
                parts.append(self._trend_sentence(float(by_hour.loc[7]), float(by_hour.loc[9]), "Die Nachfrage zwischen 07-09 Uhr"))
            afternoon = by_hour.loc[by_hour.index.to_series().between(15, 17)].mean()
            evening = by_hour.loc[by_hour.index.to_series() >= 18].mean()
            if not pd.isna(afternoon) and not pd.isna(evening):
                parts.append(self._trend_sentence(float(afternoon), float(evening), "Die Nachfrage ab 18 Uhr"))
        elif x_axis == "Datum":
            if len(series) >= 6:
                window = min(7, max(2, len(series) // 4))
                parts.append(self._trend_sentence(float(series.head(window).mean()), float(series.tail(window).mean()), "Der Datentrend"))
        elif x_axis == "Haltestelle":
            parts.append(f"Stärkster Halt im Linienablauf ist {peak_label}.")
        elif x_axis == "Wochentag":
            parts.append(f"Der stärkste Wochentag ist {peak_label}.")


        if compare_data is not None and not compare_data.empty:
            compare_series = self._metric_series(compare_data, y_metric)
            if not compare_series.empty:
                main_total = float(series.sum())
                compare_total = float(compare_series.sum())
                compare_name = compare_label.split("|")[0].strip() or "Vergleich"
                if compare_total:
                    delta = ((main_total - compare_total) / compare_total) * 100
                    relation = "über" if delta >= 0 else "unter"
                    parts.append(f"{main_name} liegt insgesamt {abs(delta):.0f}% {relation} {compare_name}.")
        return " ".join(parts)


    def _update_chart_insights(
        self,
        daily: pd.DataFrame,
        hourly: pd.DataFrame,
        flex_data: pd.DataFrame,
        x_axis: str,
        y_metric: str,
        compare_data: pd.DataFrame | None = None,
        main_label: str = "",
        compare_label: str = "",
    ) -> None:
        """
        Aktualisiert die Insight-Bereiche für Tages- und Flexible-Chart.

        Die Methode ruft die passenden Insight-Generatoren für beide Diagrammtypen
        auf und schreibt die erzeugten Texte in die zugehörigen GUI-Labels.

        Parameter:
            daily (pd.DataFrame): Tagesbezogene Daten.
            hourly (pd.DataFrame): Stundenbezogene Daten.
            flex_data (pd.DataFrame): Daten des flexiblen Diagramms.
            x_axis (str): Gewählte Ausprägung der x-Achse im flexiblen Diagramm.
            y_metric (str): Gewählte Zielkennzahl im flexiblen Diagramm.
            compare_data (pd.DataFrame | None): Optionale Vergleichsdaten.
            main_label (str): Anzeigename der Hauptauswahl.
            compare_label (str): Anzeigename der Vergleichsauswahl.

        Rückgabewerte:
            None: Die Methode aktualisiert ausschließlich GUI-Textkomponenten.

        Fehler/Sonderfälle:
            Insight-Labels werden nur aktualisiert, wenn die entsprechenden Widgets
            vorhanden sind.

        Projektkontext:
            Die Methode bündelt die automatische verbale Interpretation der
            Diagramme und hält die Erklärungsebenen synchron mit den Visualisierungen.
        """
        if self.daily_chart_insight is not None:
            self.daily_chart_insight.configure(text=self._build_daily_chart_insight(daily, hourly))
        if self.flex_chart_insight is not None:
            self.flex_chart_insight.configure(
                text=self._build_flexible_chart_insight(
                    flex_data,
                    x_axis,
                    y_metric,
                    compare_data=compare_data,
                    main_label=main_label,
                    compare_label=compare_label,
                )
            )


    def _context_counts(self, daily_totals: pd.DataFrame) -> dict[str, int]:
        """
        Zählt kontextbezogene Tage innerhalb aggregierter Tagesdaten.

        Die Methode ermittelt, an wie vielen Tagen bestimmte Einflussfaktoren wie
        Vorlesungszeit, Events, Feiertage, Schulferien oder Konzertbezug aktiv waren.

        Parameter:
            daily_totals (pd.DataFrame): Tagesaggregierte Daten mit Kontextmerkmalen.

        Rückgabewerte:
            dict[str, int]: Wörterbuch mit den Häufigkeiten je Kontexttyp.

        Fehler/Sonderfälle:
            Nicht vorhandene Kontextspalten werden als 0 behandelt.

        Projektkontext:
            Die Methode liefert kompakte Kontextkennzahlen für erklärende Insights
            und unterstützt die fachliche Einordnung von Nachfrageabweichungen.
        """
        return {
            "lectures": int((daily_totals.get("lecture_period_jmu", 0) > 0).sum()) if "lecture_period_jmu" in daily_totals else 0,
            "events": int((daily_totals.get("event_hours", 0) > 0).sum()) if "event_hours" in daily_totals else 0,
            "holidays": int((daily_totals.get("public_holiday", 0) > 0).sum()) if "public_holiday" in daily_totals else 0,
            "school_holidays": int((daily_totals.get("school_holiday", 0) > 0).sum()) if "school_holiday" in daily_totals else 0,
            "concert_events": int((daily_totals.get("concert_event_count", 0) > 0).sum()) if "concert_event_count" in daily_totals else 0,
        }


    def _safe_percent_delta(self, with_context: pd.Series, without_context: pd.Series) -> str | None:
        """
        Berechnet eine robuste prozentuale Differenz zwischen Kontext- und Referenztagen.

        Die Methode vergleicht die durchschnittliche Nachfrage zweier Serien und
        gibt die prozentuale Abweichung als formatierten Text zurück.

        Parameter:
            with_context (pd.Series): Werte für Tage mit aktivem Kontextfaktor.
            without_context (pd.Series): Werte für Tage ohne den Kontextfaktor.

        Rückgabewerte:
            str | None: Formatierte prozentuale Differenz oder None, falls keine
            belastbare Berechnung möglich ist.

        Fehler/Sonderfälle:
            Leere Serien oder ein Referenzmittelwert von 0 führen zu None, damit
            keine irreführende Prozentangabe erzeugt wird. `Series.mean()` berechnet
            dabei den Mittelwert der Werte einer Serie. [web:146]

        Projektkontext:
            Die Methode ermöglicht vorsichtige, aber verständliche Aussagen über den
            Einfluss externer Kontextfaktoren auf die Nachfrage.
        """
        if with_context.empty or without_context.empty:
            return None
        base = without_context.mean()
        if not base:
            return None
        delta = ((with_context.mean() - base) / base) * 100
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta:.0f}%"


    def _event_marker_style(
        self,
        event_hours: float,
        concert_hours: float,
        event_count: float,
        total_duration_hours: float,
        concert_event_count: float,
    ) -> tuple[int, str, str, float, str]:
        """
        Liefert Darstellungsparameter für Event-Markierungen im Tagesdiagramm.

        Die Methode kapselt Markergröße, Farbe, Symbol, Transparenz und
        Klassifikationslabel für Eventtage. In der aktuellen Implementierung wird
        noch ein einheitlicher Stil zurückgegeben.

        Parameter:
            event_hours (float): Gesamte Eventstunden des Tages.
            concert_hours (float): Konzertbezogene Stunden des Tages.
            event_count (float): Anzahl der Events am Tag.
            total_duration_hours (float): Gesamtdauer aller Events.
            concert_event_count (float): Anzahl konzertbezogener Events.

        Rückgabewerte:
            tuple[int, str, str, float, str]: Markergröße, Farbe, Markerform,
            Transparenz und Klassifikationslabel.

        Fehler/Sonderfälle:
            Die Eingabeparameter beeinflussen aktuell den Rückgabewert noch nicht;
            die Methode ist somit als Erweiterungspunkt angelegt.

        Projektkontext:
            Die Methode schafft eine zentrale Stelle für die visuelle Kodierung
            ereignisgetriebener Nachfragekontexte im Dashboard.
        """
        return 78, "#ea580c", "D", 0.86, "event"


    def _set_metric(self, name: str, value: str) -> None:
        """
        Aktualisiert den Text einer KPI-Karte.

        Die Methode greift über den Kennzahlnamen auf das zugehörige Label im
        KPI-Kartenregister zu und setzt den neuen Anzeigewert.

        Parameter:
            name (str): Name der Kennzahl.
            value (str): Neuer darzustellender Textwert.

        Rückgabewerte:
            None: Die Methode aktualisiert ausschließlich die GUI-Anzeige.

        Fehler/Sonderfälle:
            Ein vorhandener Eintrag in `self.metric_cards` wird vorausgesetzt.

        Projektkontext:
            Die Methode standardisiert die Aktualisierung zentraler Dashboard-
            Kennzahlen.
        """
        self.metric_cards[name].configure(text=value)


    def _render_empty_states(self) -> None:
        """
        Rendert Standardzustände für den Fall fehlender Daten.

        Die Methode setzt Textboxen auf erklärende Leermeldungen zurück und ersetzt
        Diagramme durch Platzhalterdarstellungen, damit die Oberfläche auch bei
        leerer Datenbasis konsistent und verständlich bleibt.

        Parameter:
            Keine.

        Rückgabewerte:
            None: Die Methode aktualisiert mehrere GUI-Komponenten.

        Fehler/Sonderfälle:
            Einzelne Widgets werden nur aktualisiert, wenn sie vorhanden sind.

        Projektkontext:
            Die Methode stellt sicher, dass Datenlücken nicht wie Fehler wirken,
            sondern als nachvollziehbarer Analysezustand kommuniziert werden.
        """
        if self.station_textbox is not None:
            self.station_textbox.delete("1.0", "end")
            self.station_textbox.insert("1.0", "Keine Haltestellendaten im gewählten Zeitraum.")
        if self.insight_textbox is not None:
            self.insight_textbox.delete("1.0", "end")
            self.insight_textbox.insert("1.0", "Die aktuelle Kombination aus Linien und Zeitraum liefert keine Daten.")
        self._draw_placeholder(self.chart_host, "Keine Nachfragekurve verfügbar")
        if self.flex_host is not None:
            self._draw_placeholder(self.flex_host, "Keine flexible Analyse verfügbar")
        if self.daily_chart_insight is not None:
            self.daily_chart_insight.configure(text="Insight: Keine Tagesdaten für die aktuelle Auswahl verfügbar.")
        if self.flex_chart_insight is not None:
            self.flex_chart_insight.configure(text="Insight: Keine flexible Analyse für die aktuelle Auswahl verfügbar.")


    def _draw_placeholder(self, host: ctk.CTkFrame, text: str) -> None:
        """
        Zeichnet einen textuellen Platzhalter in einen Diagramm- oder Inhaltscontainer.

        Die Methode entfernt zunächst alle vorhandenen Kind-Widgets des Hosts und
        fügt anschließend ein einfaches Label mit Hinweistext ein.

        Parameter:
            host (ctk.CTkFrame): Zielcontainer für den Platzhalter.
            text (str): Anzuzeigender Hinweistext.

        Rückgabewerte:
            None: Die Methode ersetzt den sichtbaren Inhalt des Containers.

        Fehler/Sonderfälle:
            Ein gültiger Container wird vorausgesetzt.

        Projektkontext:
            Die Methode unterstützt eine konsistente Leerdarstellung über mehrere
            Dashboard-Bereiche hinweg.
        """
        for child in host.winfo_children():
            child.destroy()
        ctk.CTkLabel(
            host,
            text=text,
            text_color=PALETTE["muted"],
            font=ctk.CTkFont(family="Segoe UI", size=16),
        ).pack(expand=True)


    def _draw_daily_chart(self, daily: pd.DataFrame) -> None:
        """
        Rendert das Tagesdiagramm mit Nachfrageverlauf und Kontextmarkierungen.

        Die Methode visualisiert linienbezogene Tagesverläufe als Linien- und
        Flächendiagramm und ergänzt optional Kontextinformationen wie Vorlesungstage,
        Feiertage, Schulferien und Events durch farbliche Markierungen, vertikale
        Linien oder Marker. Matplotlib unterstützt für solche Overlays unter anderem
        `axvspan` für markierte Zeitbereiche, `axvline` für vertikale Bezugslinien
        und `scatter` für punktuelle Ereignismarkierungen. [web:159]

        Parameter:
            daily (pd.DataFrame): Tagesbezogene Nachfrage- und Kontextdaten.

        Rückgabewerte:
            None: Die Methode erzeugt ein eingebettetes Diagramm im Dashboard.

        Fehler/Sonderfälle:
            Vor dem Rendern wird der Zielcontainer geleert. Kontextlayer werden nur
            gezeichnet, wenn die jeweiligen Optionen aktiv sind und die benötigten
            Spalten vorliegen.

        Projektkontext:
            Die Methode stellt den zentralen visuellen Überblick über Tagesdynamik
            und externe Einflussfaktoren des Systems bereit.
        """
        for child in self.chart_host.winfo_children():
            child.destroy()


        fig = Figure(figsize=(12.8, 5.6), dpi=100, facecolor=PALETTE["surface"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(PALETTE["surface"])
        colors = ["#ef4444", "#14b8a6", "#f59e0b", "#38bdf8", "#a78bfa", "#fb7185"]
        daily_totals = self._daily_totals(daily)
        context_handles: list[object] = []
        line_handles: list[object] = []
        has_event_days = False


        if self.show_lectures.get() and "lecture_period_jmu" in daily_totals:
            lecture_days = daily_totals.loc[daily_totals["lecture_period_jmu"] > 0, "date"]
            if not lecture_days.empty:
                context_handles.append(Patch(facecolor="#8ec5ff", edgecolor="#8ec5ff", alpha=0.24, label="Vorlesung"))
            for current_date in lecture_days:
                start = pd.Timestamp(current_date) - pd.Timedelta(hours=12)
                end = pd.Timestamp(current_date) + pd.Timedelta(hours=12)
                ax.axvspan(start, end, color="#8ec5ff", alpha=0.24)


        if self.show_holidays.get() and "public_holiday" in daily_totals:
            holiday_days = daily_totals.loc[daily_totals["public_holiday"] > 0, "date"]
            if not holiday_days.empty:
                context_handles.append(Line2D([0], [0], color="#ff5a5f", linewidth=2.2, linestyle="--", label="Feiertag"))
            for current_date in holiday_days:
                ax.axvline(current_date, color="#ff5a5f", alpha=0.36, linewidth=2.2, linestyle="--")


        if self.show_school_holidays.get() and "school_holiday" in daily_totals:
            school_days = daily_totals.loc[daily_totals["school_holiday"] > 0, "date"]
            if not school_days.empty:
                context_handles.append(Patch(facecolor="#ffd166", edgecolor="#ffd166", alpha=0.18, label="Ferien"))
            for current_date in school_days:
                start = pd.Timestamp(current_date) - pd.Timedelta(hours=12)
                end = pd.Timestamp(current_date) + pd.Timedelta(hours=12)
                ax.axvspan(start, end, color="#ffd166", alpha=0.18)


        for idx, line in enumerate(sorted(daily["line"].unique())):
            frame = daily[daily["line"] == line]
            ax.plot(frame["date"], frame["boardings"], linewidth=2.6, color=colors[idx % len(colors)])
            ax.fill_between(frame["date"], frame["boardings"], color=colors[idx % len(colors)], alpha=0.08)
            line_handles.append(Line2D([0], [0], color=colors[idx % len(colors)], linewidth=2.6, label=f"Linie {line}"))


        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(PALETTE["border"])
        ax.spines["bottom"].set_color(PALETTE["border"])
        ax.grid(axis="y", color=PALETTE["border"], linewidth=1, alpha=0.72)
        if self.show_events.get() and "event_hours" in daily_totals:
            event_days = daily_totals[daily_totals["event_hours"] > 0]
            if not event_days.empty:
                y_top = max(daily_totals["boardings"].max(), 1)
                for _, row in event_days.iterrows():
                    current_date = row["date"]
                    size, color, marker, alpha, level = self._event_marker_style(
                        float(row.get("event_hours", 0)),
                        float(row.get("concert_hours", 0)),
                        float(row.get("event_count", 0)),
                        float(row.get("total_event_duration_hours", 0)),
                        float(row.get("concert_event_count", 0)),
                    )
                    has_event_days = True
                    ax.axvline(
                        current_date,
                        color=color,
                        alpha=alpha,
                        linewidth=2.2,
                        linestyle=":",
                    )
                    ax.scatter(
                        [current_date],
                        [y_top * 1.04],
                        color=color,
                        edgecolors="white",
                        linewidths=0.9,
                        marker=marker,
                        s=size,
                        alpha=alpha,
                        zorder=6,
                    )


        ax.tick_params(axis="x", colors=PALETTE["muted"], labelsize=10)
        ax.tick_params(axis="y", colors=PALETTE["muted"], labelsize=10)
        ax.set_ylabel("Einstiege", color=PALETTE["muted"])
        event_handles = []
        if has_event_days:
            event_handles.append(
                Line2D([0], [0], marker="D", color="none", markerfacecolor="#ea580c", markeredgecolor="white", markersize=10, label="Event")
            )


        all_handles = context_handles + line_handles + event_handles
        if all_handles:
            legend = ax.legend(
                handles=all_handles,
                frameon=False,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.22),
                ncol=min(4, max(1, len(all_handles))),
                borderaxespad=0.0,
            )
            for text in legend.get_texts():
                text.set_color(PALETTE["text"])
        fig.subplots_adjust(top=0.76, left=0.07, right=0.98, bottom=0.13)
        canvas = FigureCanvasTkAgg(fig, master=self.chart_host)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


    def _draw_flexible_chart(
        self,
        data: pd.DataFrame,
        x_axis: str,
        y_metric: str,
        compare_data: pd.DataFrame | None = None,
        main_label: str = "",
        compare_label: str = "",
    ) -> None:
        """
        Rendert das flexible Vergleichsdiagramm für unterschiedliche Analyseperspektiven.

        Die Methode visualisiert je nach gewählter Achsenkonfiguration und Kennzahl
        Linien-, Flächen- oder Balkendiagramme. Zusätzlich kann eine Vergleichslinie
        eingeblendet werden, um zwei Linien oder Auswahlzustände direkt gegenüberzustellen.

        Parameter:
            data (pd.DataFrame): Hauptdaten der flexiblen Analyse.
            x_axis (str): Gewählte Dimension der x-Achse.
            y_metric (str): Gewählte Zielkennzahl.
            compare_data (pd.DataFrame | None): Optionale Vergleichsdaten.
            main_label (str): Anzeigename der Hauptauswahl.
            compare_label (str): Anzeigename der Vergleichsauswahl.

        Rückgabewerte:
            None: Die Methode erzeugt ein eingebettetes Diagramm im Dashboard.

        Fehler/Sonderfälle:
            Ist kein Zielcontainer vorhanden, wird die Methode beendet. Bei leerer
            Datenbasis wird ein Platzhalter gerendert. Die Darstellungslogik
            unterscheidet mehrere Sonderfälle, etwa Datumsachsen, Haltestellen-
            vergleiche und die kombinierte Metrik „Ein- und Ausstiege“.

        Projektkontext:
            Die Methode ist der visuelle Kern der explorativen Analyse, da sie
            unterschiedliche fachliche Fragestellungen mit einer gemeinsamen
            Diagrammkomponente abbildet.
        """
        if self.flex_host is None:
            return
        for child in self.flex_host.winfo_children():
            child.destroy()
        if data.empty:
            self._draw_placeholder(self.flex_host, "Keine flexible Analyse verfügbar")
            return


        fig = Figure(figsize=(12.8, 4.2), dpi=100, facecolor=PALETTE["surface"])
        ax = fig.add_subplot(111)
        ax.set_facecolor(PALETTE["surface"])


        has_compare = compare_data is not None and not compare_data.empty
        main_name = main_label.split("|")[0].strip() or "Auswahl"
        compare_name = compare_label.split("|")[0].strip() if has_compare else ""
        values = data["value"].astype(float)
        show_flow_pair = y_metric == "Ein- und Ausstiege"
        if x_axis == "Datum" and show_flow_pair:
            x_values = pd.to_datetime(data["x_value"])
            ax.plot(x_values, data["boardings"].astype(float), linewidth=2.8, color=PALETTE["rose"], marker="o", markersize=3.2, label=f"{main_name} Einstiege")
            ax.plot(x_values, data["exiting"].astype(float), linewidth=2.8, color=PALETTE["teal"], marker="o", markersize=3.2, label=f"{main_name} Ausstiege")
            if has_compare:
                compare_x = pd.to_datetime(compare_data["x_value"])
                ax.plot(compare_x, compare_data["boardings"].astype(float), linewidth=2.4, color=PALETTE["amber"], linestyle="--", marker="o", markersize=2.8, label=f"{compare_name} Einstiege")
                ax.plot(compare_x, compare_data["exiting"].astype(float), linewidth=2.4, color=PALETTE["blue"], linestyle="--", marker="o", markersize=2.8, label=f"{compare_name} Ausstiege")
            fig.autofmt_xdate(rotation=25)
            legend = ax.legend(frameon=False, loc="upper right")
            for text in legend.get_texts():
                text.set_color(PALETTE["text"])
        elif x_axis == "Datum":
            x_values = pd.to_datetime(data["x_value"])
            ax.plot(x_values, values, linewidth=2.8, color=PALETTE["accent"], marker="o", markersize=3.2, label=main_name)
            ax.fill_between(x_values, values, color=PALETTE["accent"], alpha=0.12)
            if has_compare:
                compare_x = pd.to_datetime(compare_data["x_value"])
                ax.plot(compare_x, compare_data["value"].astype(float), linewidth=2.6, color=PALETTE["teal"], linestyle="--", marker="o", markersize=3.0, label=compare_name)
                legend = ax.legend(frameon=False, loc="upper right")
                for text in legend.get_texts():
                    text.set_color(PALETTE["text"])
            fig.autofmt_xdate(rotation=25)
        else:
            if has_compare and x_axis == "Haltestelle":
                main_frame = data.head(12).copy()
                compare_frame = compare_data.head(12).copy()
                main_frame["series_name"] = main_name
                compare_frame["series_name"] = compare_name
                plot_frame = pd.concat([main_frame, compare_frame], ignore_index=True)
                labels = [f"{row['series_name']} {row['x_label']}" for _, row in plot_frame.iterrows()]
                positions = list(range(len(plot_frame)))
                if show_flow_pair:
                    width = 0.36
                    ax.bar([position - width / 2 for position in positions], plot_frame["boardings"].astype(float), color=PALETTE["rose"], alpha=0.92, width=width, label="Einstiege")
                    ax.bar([position + width / 2 for position in positions], plot_frame["exiting"].astype(float), color=PALETTE["teal"], alpha=0.92, width=width, label="Ausstiege")
                else:
                    colors = [PALETTE["accent"] if name == main_name else PALETTE["teal"] for name in plot_frame["series_name"]]
                    ax.bar(positions, plot_frame["value"].astype(float), color=colors, alpha=0.92, width=0.68)
            else:
                labels = data["x_label"].astype(str).tolist()
                positions = list(range(len(labels)))
                if show_flow_pair:
                    width = 0.36 if not has_compare else 0.18
                    ax.bar([position - width / 2 for position in positions], data["boardings"].astype(float), color=PALETTE["rose"], alpha=0.92, width=width, label=f"{main_name} Einstiege")
                    ax.bar([position + width / 2 for position in positions], data["exiting"].astype(float), color=PALETTE["teal"], alpha=0.92, width=width, label=f"{main_name} Ausstiege")
                    if has_compare:
                        compare_map = compare_data.set_index("x_label")
                        compare_boardings = [float(compare_map["boardings"].get(label, 0)) if label in compare_map.index else 0 for label in labels]
                        compare_exiting = [float(compare_map["exiting"].get(label, 0)) if label in compare_map.index else 0 for label in labels]
                        ax.bar([position + width * 1.6 for position in positions], compare_boardings, color=PALETTE["amber"], alpha=0.88, width=width, label=f"{compare_name} Einstiege")
                        ax.bar([position + width * 2.7 for position in positions], compare_exiting, color=PALETTE["blue"], alpha=0.88, width=width, label=f"{compare_name} Ausstiege")
                elif has_compare:
                    width = 0.34
                    compare_map = compare_data.set_index("x_label")
                    compare_values = [float(compare_map["value"].get(label, 0)) if label in compare_map.index else 0 for label in labels]
                    ax.bar([position - width / 2 for position in positions], values, color=PALETTE["accent"], alpha=0.92, width=width, label=main_name)
                    ax.bar([position + width / 2 for position in positions], compare_values, color=PALETTE["teal"], alpha=0.92, width=width, label=compare_name)
                else:
                    colors = [PALETTE["accent"], PALETTE["teal"], PALETTE["amber"], PALETTE["blue"], "#a78bfa"]
                    colors = [colors[index % len(colors)] for index in positions]
                    ax.bar(positions, values, color=colors, alpha=0.92, width=0.68)
            if show_flow_pair or has_compare:
                legend = ax.legend(frameon=False, loc="upper right")
                for text in legend.get_texts():
                    text.set_color(PALETTE["text"])
            rotation = 35 if x_axis in {"Haltestelle", "Datum"} else 0
            ax.set_xticks(positions)
            ax.set_xticklabels(
                labels,
                rotation=rotation,
                ha="right" if rotation else "center",
            )


        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(PALETTE["border"])
        ax.spines["bottom"].set_color(PALETTE["border"])
        ax.grid(axis="y", color=PALETTE["border"], linewidth=1, alpha=0.72)
        ax.tick_params(axis="x", colors=PALETTE["muted"], labelsize=9)
        ax.tick_params(axis="y", colors=PALETTE["muted"], labelsize=10)
        ax.set_xlabel(x_axis, color=PALETTE["muted"])
        ax.set_ylabel(y_metric, color=PALETTE["muted"])
        ax.set_title(f"{y_metric} nach {x_axis}", color=PALETTE["text"], fontsize=13, fontweight="bold", loc="left")
        fig.tight_layout(pad=2)
        canvas = FigureCanvasTkAgg(fig, master=self.flex_host)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


    def _fill_station_panel(self, stations: pd.DataFrame) -> None:
        """
        Befüllt das Haltestellenpanel mit den wichtigsten Haltestellen der Auswahl.

        Die Methode reduziert die Daten auf die obersten Einträge und formatiert sie
        als kompakte Textliste mit Linie, Reihenfolge, Haltestellenname sowie Ein-
        und Ausstiegswerten.

        Parameter:
            stations (pd.DataFrame): Haltestellenbezogene aggregierte Daten.

        Rückgabewerte:
            None: Die Methode aktualisiert die Haltestellen-Textbox.

        Fehler/Sonderfälle:
            Ist keine Textbox vorhanden, wird die Methode beendet. Bei leerer
            Datenbasis wird eine Standardmeldung angezeigt.

        Projektkontext:
            Die Methode stellt eine textuelle Schnellübersicht der wichtigsten
            Haltestellen im aktuellen Analysefenster bereit.
        """
        if self.station_textbox is None:
            return
        top = stations.head(18).copy()
        lines = []
        for _, row in top.iterrows():
            name = str(row["station"])[:22]
            lines.append(
                f"L{int(row['line']):>3} {int(row['display_sequence']):>2}. {name:<22} "
                f"Ein {int(row['boardings']):>6} | Aus {int(row.get('exiting', 0)):>6}"
            )
        self.station_textbox.delete("1.0", "end")
        self.station_textbox.insert("1.0", "\n".join(lines) if lines else "Keine Haltestellenfolge verfügbar.")


    def _fill_insights(self, daily: pd.DataFrame, hourly: pd.DataFrame, stations: pd.DataFrame) -> None:
        """
        Befüllt den zentralen Insight-Bereich mit fachlich interpretierten Analyseaussagen.

        Die Methode kombiniert linienbezogene Nachfragewerte, Tages- und Stundenpeaks,
        Haltestellenprioritäten sowie optional Kontextfaktoren wie Vorlesungen,
        Events, Feiertage und Schulferien zu einem mehrabsätzigen Textblock.

        Parameter:
            daily (pd.DataFrame): Tagesbezogene Nachfrage- und Linieninformationen.
            hourly (pd.DataFrame): Stundenbezogene Nachfrageinformationen.
            stations (pd.DataFrame): Haltestellenbezogene aggregierte Daten.

        Rückgabewerte:
            None: Die Methode aktualisiert die Insight-Textbox.

        Fehler/Sonderfälle:
            Ist keine Insight-Textbox vorhanden, wird die Methode beendet. Die
            Berechnung setzt voraus, dass die übergebenen Datenrahmen nicht leer sind;
            sie wird typischerweise nur nach erfolgreicher Dashboard-Aggregation aufgerufen.

        Projektkontext:
            Die Methode erzeugt eine fachlich interpretierende Narrativebene über den
            Rohdaten und unterstützt damit die Ableitung verkehrlicher Aussagen aus
            der Analyseoberfläche.
        """
        if self.insight_textbox is None:
            return
        total_by_line = daily.groupby("line")["boardings"].sum().sort_values(ascending=False)
        daily_totals = self._daily_totals(daily)
        totals_only = daily_totals.set_index("date")["boardings"]
        hour_totals = hourly.groupby("hour")["boardings"].sum()
        top_station_row = stations.sort_values("boardings", ascending=False).iloc[0]
        counts = self._context_counts(daily_totals)


        lines = [
            f"Führende Linie: {self.repo.label_for_line(int(total_by_line.index[0]))} mit {int(total_by_line.iloc[0]):,} Einstiegen.".replace(",", "."),
            f"Spitzenlast: {totals_only.idxmax().isoformat()} war der stärkste Tag der Auswahl.",
            f"Tagesvolatilität: {(totals_only.std(ddof=0) / totals_only.mean()):.2f} als CV.",
            f"Prime Time: Die stärkste Nachfrage liegt um {int(hour_totals.idxmax()):02d}:00 Uhr.",
            f"Top Stop: {str(top_station_row['station'])} auf Linie {int(top_station_row['line'])}.",
        ]
        if 10 in self.selected_lines:
            lines.append("Linie 10 bleibt der dynamische Hubland-Korridor für adaptive Steuerung.")
        if 55 in self.selected_lines:
            lines.append("Linie 55 bringt einen eventgetriebenen Robustheitstest in die Auswahl.")
        if 27 in self.selected_lines:
            lines.append("Linie 27 liefert einen sauberen Pendlerkontrast.")


        if self.show_lectures.get() and "lecture_period_jmu" in daily_totals:
            lecture_delta = self._safe_percent_delta(
                daily_totals.loc[daily_totals["lecture_period_jmu"] > 0, "boardings"],
                daily_totals.loc[daily_totals["lecture_period_jmu"] == 0, "boardings"],
            )
            if lecture_delta:
                lines.append(f"Vorlesungen aktiv: {counts['lectures']} Tage im Zeitraum, Nachfrage im Schnitt {lecture_delta} gegenüber Nicht-Vorlesungstagen.")


        if self.show_events.get() and "event_hours" in daily_totals:
            event_delta = self._safe_percent_delta(
                daily_totals.loc[daily_totals["event_hours"] > 0, "boardings"],
                daily_totals.loc[daily_totals["event_hours"] == 0, "boardings"],
            )
            if event_delta:
                lines.append(f"Events aktiv: {counts['events']} Eventtage, Nachfrage im Schnitt {event_delta} gegenüber Tagen ohne Event.")
            event_days = daily_totals.loc[daily_totals["event_hours"] > 0].copy()
            if not event_days.empty:
                top_event = event_days.sort_values(
                    ["event_count", "total_event_duration_hours", "event_hours", "boardings"],
                    ascending=False,
                ).iloc[0]
                concert_days = int((event_days.get("concert_hours", 0) > 0).sum()) if "concert_hours" in event_days else 0
                lines.append(
                    f"Stärkster Eventtag: {top_event['date'].isoformat()} mit {int(top_event.get('event_count', 0))} Events, "
                    f"{float(top_event.get('total_event_duration_hours', 0)):.1f} Stunden Gesamtdauer"
                    + (f" und {int(top_event.get('concert_event_count', 0))} Konzert-Events." if "concert_event_count" in top_event else ".")
                )
                if concert_days:
                    lines.append(f"Davon {concert_days} Tage mit Konzertbezug, diese sind im Chart als Stern-Marker sichtbar.")


        if self.show_holidays.get() and "public_holiday" in daily_totals:
            holiday_delta = self._safe_percent_delta(
                daily_totals.loc[daily_totals["public_holiday"] > 0, "boardings"],
                daily_totals.loc[daily_totals["public_holiday"] == 0, "boardings"],
            )
            if holiday_delta:
                lines.append(f"Feiertage aktiv: {counts['holidays']} Feiertage, Nachfrage im Schnitt {holiday_delta} gegenüber normalen Tagen.")


        if self.show_school_holidays.get() and "school_holiday" in daily_totals:
            school_delta = self._safe_percent_delta(
                daily_totals.loc[daily_totals["school_holiday"] > 0, "boardings"],
                daily_totals.loc[daily_totals["school_holiday"] == 0, "boardings"],
            )
            if school_delta:
                lines.append(f"Ferien aktiv: {counts['school_holidays']} Ferientage, Nachfrage im Schnitt {school_delta} gegenüber Tagen ohne Ferien.")


        self.insight_textbox.delete("1.0", "end")
        self.insight_textbox.insert("1.0", "\n\n".join(lines))



    def run() -> None:
        """
        Startet die Anwendung mit initialisiertem Datenrepository und GUI.

        Die Methode prüft zunächst das Vorhandensein des Datenverzeichnisses,
        erstellt anschließend das Repository sowie das Hauptfenster und startet die
        Tkinter-Ereignisschleife.

        Parameter:
            Keine.

        Rückgabewerte:
            None: Die Methode startet die Anwendung dauerhaft bis zum Schließen des GUI-Fensters.

        Fehler/Sonderfälle:
            Existiert das konfigurierte Datenverzeichnis nicht, wird ein
            `FileNotFoundError` ausgelöst.

        Projektkontext:
            Die Methode bildet den formalen Einstiegspunkt der Anwendung und koppelt
            Datenzugriff, Dashboard-Instanz und GUI-Laufzeit zusammen.
        """
        if not DATA_DIR.exists():
            raise FileNotFoundError(f"Datenordner nicht gefunden: {DATA_DIR}")
        repo = TransitDataRepository(DATA_DIR)
        app = TransitDashboard(repo)
        app.mainloop()



    if __name__ == "__main__":
        
        run()