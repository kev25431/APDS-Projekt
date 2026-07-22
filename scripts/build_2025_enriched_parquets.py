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
    """
    Kapselt Verarbeitungsstatistiken für eine einzelne Quelldatei.

    Die Datenklasse sammelt technische und fachlich relevante Kennzahlen eines
    Verarbeitungslaufs, etwa Eingangs- und Ausgangszeilen, bereinigte Duplikate,
    korrigierte negative Werte und Informationen zum optional erzeugten
    Model-Ready-Export.

    Parameter:
        source_file (str): Name der Quelldatei.
        output_file (str): Name der erzeugten Ausgabedatei.
        rows_in (int): Anzahl Zeilen in der Quelldatei vor Filterung.
        rows_2025 (int): Anzahl Zeilen nach Jahresfilterung.
        rows_out (int): Anzahl Zeilen im bereinigten Ausgabedatensatz.
        duplicate_rows_removed (int): Anzahl entfernter Dubletten.
        negative_values_fixed (int): Anzahl korrigierter negativer Werte.
        missing_values_filled (int): Anzahl imputierter fehlender Werte.
        model_ready_file (str): Name der optional erzeugten ML-Datei.
        model_ready_rows (int): Anzahl Zeilen im Model-Ready-Datensatz.
        skipped (bool): Kennzeichen, ob die Datei übersprungen wurde.
        reason (str): Begründung für das Überspringen.

    Rückgabewerte:
        ProcessStats: Instanz mit allen relevanten Laufkennzahlen.

    Fehler/Sonderfälle:
        Bei übersprungenen Dateien bleiben viele Kennzahlen auf 0; `reason`
        dokumentiert den Grund.

    Projektkontext:
        Die Klasse strukturiert die Laufprotokollierung und bildet die Grundlage
        für das Manifest der gesamten ETL-Ausführung.
    """
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
    """
    Parst die Kommandozeilenargumente des ETL-Skripts.

    Die Funktion definiert Standardpfade für Rohdaten, Zusatzdaten,
    Ausgabeverzeichnisse und den optionalen Model-Ready-Export. Über `argparse`
    wird daraus eine nutzerfreundliche CLI mit Parametern und Hilfetexten
    aufgebaut. [web:178]

    Parameter:
        Keine.

    Rückgabewerte:
        argparse.Namespace: Geparste Kommandozeilenargumente.

    Fehler/Sonderfälle:
        Ungültige Argumente werden durch `argparse` mit einer Fehlermeldung und
        einem Programmabbruch behandelt.

    Projektkontext:
        Die Funktion ist der Einstiegspunkt für reproduzierbare und steuerbare
        Batch-Verarbeitung der WVV-Daten.
    """
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
    """
    Erzeugt einen täglichen Kalenderrahmen für ein vollständiges Jahr.

    Der resultierende DataFrame enthält für jeden Tag des Zieljahres zentrale
    Kalendermerkmale wie Wochentag, Wochenendindikator und Monat. Damit bildet
    er die Grundlage für das spätere Mergen weiterer externer Kontextdaten.

    Parameter:
        year (int): Zieljahr, für das der Datumsrahmen aufgebaut werden soll.

    Rückgabewerte:
        pd.DataFrame: Täglicher Kalenderrahmen mit Datums- und Basiszeitmerkmalen.

    Fehler/Sonderfälle:
        Keine explizite Fehlerbehandlung.

    Projektkontext:
        Die Funktion liefert das Grundgerüst für die spätere Kontextanreicherung
        aller Verkehrsdaten.
    """
    dates = pd.date_range(date(year, 1, 1), date(year, 12, 31), freq="D")
    frame = pd.DataFrame({"date": dates.date})
    frame["weekday"] = pd.to_datetime(frame["date"].astype(str)).dt.weekday
    frame["is_weekend"] = frame["weekday"].isin([5, 6]).astype("int8")
    frame["month"] = pd.to_datetime(frame["date"].astype(str)).dt.month.astype("int8")
    return frame



def read_csv_flexible(path: Path, **kwargs) -> pd.DataFrame:
    """
    Liest CSV-Dateien tolerant gegenüber unterschiedlichen Trennzeichen ein.

    Die Funktion versucht zunächst einen regulären CSV-Import und fällt bei
    Parserfehlern auf ein Semikolon als Trennzeichen zurück. Dadurch können
    heterogene Zusatzdatenquellen robuster verarbeitet werden.

    Parameter:
        path (Path): Pfad zur einzulesenden CSV-Datei.
        **kwargs: Weitere Argumente für `pandas.read_csv`.

    Rückgabewerte:
        pd.DataFrame: Eingelesener DataFrame oder ein leerer DataFrame, falls die
        Datei nicht existiert.

    Fehler/Sonderfälle:
        Existiert die Datei nicht, wird ein leerer DataFrame zurückgegeben. Nur
        `ParserError` löst den expliziten Fallback auf Semikolon aus.

    Projektkontext:
        Die Funktion erleichtert die Einbindung unterschiedlich formatierter
        Zusatzdateien in den ETL-Prozess.
    """
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.ParserError:
        return pd.read_csv(path, sep=";", **kwargs)



def ensure_date_column(frame: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    """
    Normalisiert eine Datums-Spalte auf Python-`date`-Werte.

    Die Funktion konvertiert die angegebene Spalte tolerant nach Datum und
    entfernt Zeilen, für die keine gültige Datumsinterpretation möglich ist.

    Parameter:
        frame (pd.DataFrame): Eingabedatenrahmen.
        column (str): Name der zu normalisierenden Datumsspalte.

    Rückgabewerte:
        pd.DataFrame: Kopie des DataFrames mit bereinigter Datumsspalte.

    Fehler/Sonderfälle:
        Ist der DataFrame leer oder die Spalte nicht vorhanden, wird der Input
        unverändert zurückgegeben.

    Projektkontext:
        Die Funktion stellt sicher, dass spätere Joins auf Tagesebene konsistent
        und ohne Formatkonflikte möglich sind.
    """
    if frame.empty or column not in frame.columns:
        return frame
    frame = frame.copy()
    frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.date
    return frame.dropna(subset=[column])



def build_daily_context(additional_dir: Path, year: int) -> pd.DataFrame:
    """
    Baut den täglichen Kontextdatensatz aus externen Zusatzquellen auf.

    Zusammengeführt werden gesetzliche Feiertage, Schulferien, Vorlesungszeiten,
    Events sowie verkaufsoffene Sonntage. Fehlende Informationen werden
    systematisch ergänzt, typisiert und in ein konsistentes Tagesformat
    überführt.

    Parameter:
        additional_dir (Path): Verzeichnis mit Zusatzdateien.
        year (int): Zieljahr der Kontextaufbereitung.

    Rückgabewerte:
        pd.DataFrame: Täglicher Kontextdatensatz.
        pd.DataFrame: Stundenbezogener Event-Kontext für spätere Merges.

    Fehler/Sonderfälle:
        Fehlende oder leere Zusatzdateien werden tolerant behandelt; die
        entsprechenden Merkmale werden dann mit 0 bzw. leeren Strings befüllt.

    Projektkontext:
        Die Funktion bündelt domänenspezifische Einflussfaktoren auf die
        Fahrgastnachfrage und macht sie für spätere Modellierung nutzbar.
    """
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
    """
    Führt tägliche Kontextdaten selektiv in einen Basisrahmen ein.

    Die Funktion merged nur die angeforderten und tatsächlich vorhandenen
    Spalten und stellt sicher, dass bei fehlender Zusatzquelle die erwarteten
    Zielspalten dennoch mit 0 existieren.

    Parameter:
        base (pd.DataFrame): Basisrahmen auf Tagesebene.
        other (pd.DataFrame): Einzumergender Kontextdatensatz.
        columns (list[str]): Erwartete Kontextspalten.

    Rückgabewerte:
        pd.DataFrame: Gemergter DataFrame mit den angeforderten Merkmalen.

    Fehler/Sonderfälle:
        Ist `other` leer, werden fehlende Zielspalten direkt im Basisrahmen mit 0 ergänzt.

    Projektkontext:
        Die Funktion standardisiert das Merging heterogener Tageskontexte und
        reduziert Boilerplate-Code in der Kontextaufbereitung.
    """
    if other.empty:
        for column in columns:
            if column not in base.columns:
                base[column] = 0
        return base
    available = ["date", *[column for column in columns if column in other.columns]]
    merged = base.merge(other[available], on="date", how="left")
    return merged



def merge_lecture_ranges(context: pd.DataFrame, lectures: pd.DataFrame) -> pd.DataFrame:
    """
    Überführt Vorlesungszeiträume in tägliche Indikatorvariablen.

    Die Funktion interpretiert Start- und Enddaten aus der Vorlesungsdatei und
    markiert im Kalenderrahmen die Tage, an denen Vorlesungsbetrieb für JMU
    und/oder THWS stattfindet.

    Parameter:
        context (pd.DataFrame): Täglicher Kalender- bzw. Kontextrahmen.
        lectures (pd.DataFrame): Datensatz mit Vorlesungszeiträumen.

    Rückgabewerte:
        pd.DataFrame: Kontextdatensatz mit ergänzten Vorlesungsindikatoren.

    Fehler/Sonderfälle:
        Fehlen erforderliche Spalten oder ist der Datensatz leer, wird
        `lecture_period_thws` bei Bedarf mit 0 ergänzt und ansonsten der Input
        weitgehend unverändert zurückgegeben.

    Projektkontext:
        Vorlesungszeiten sind ein fachlich relevanter Nachfragetreiber für den
        ÖPNV in einer Studierendenstadt und werden hier auf Tagesebene kodiert.
    """
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
    """
    Erzeugt täglichen und stündlichen Event-Kontext aus einer Eventdatei.

    Die Funktion verdichtet Veranstaltungen auf Tagesebene zu Kennzahlen wie
    Anzahl, Konzertanteil und Dauer und erstellt zusätzlich einen stündlichen
    Kontext, der aktive Event- und Konzertstunden markiert.

    Parameter:
        additional_dir (Path): Verzeichnis mit Eventdateien.
        year (int): Zieljahr.

    Rückgabewerte:
        tuple[pd.DataFrame, pd.DataFrame]: Tageskontext und Stundenkontext für Events.

    Fehler/Sonderfälle:
        Fehlen Eventdaten oder notwendige Spalten, werden leere bzw. mit 0
        belegte Fallback-Strukturen zurückgegeben. Events werden auf das Zieljahr
        zugeschnitten.

    Projektkontext:
        Events können kurzfristige Nachfrageimpulse erzeugen und werden deshalb
        sowohl tages- als auch stundenbezogen modellierbar gemacht.
    """
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
    """
    Erzeugt einen leeren, aber schema-kompatiblen Event-Kontext.

    Die Funktion ergänzt alle erwarteten Eventspalten mit 0, wenn keine
    verwertbaren Eventdaten vorliegen.

    Parameter:
        base (pd.DataFrame): Basisrahmen mit Datumsspalte.

    Rückgabewerte:
        pd.DataFrame: Event-Kontext ohne aktive Events.

    Fehler/Sonderfälle:
        Keine explizite Fehlerbehandlung.

    Projektkontext:
        Die Funktion stellt sicher, dass nachgelagerte Pipelines auch ohne
        Eventdaten ein stabiles Schema erhalten.
    """
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
    """
    Aggregiert stündliche Innenstadt-Fußgänger- und Temperaturdaten.

    Die Funktion liest Messdaten aus `dataAllStreets.csv`, filtert sie auf das
    Zieljahr und verdichtet sie je Datum und Stunde zu stadtbezogenen
    Bewegungs- und Wettermerkmalen.

    Parameter:
        additional_dir (Path): Verzeichnis mit Zusatzdaten.
        year (int): Zieljahr der Aggregation.

    Rückgabewerte:
        pd.DataFrame: Stündlicher Straßenkontext mit aggregierten Kennzahlen.

    Fehler/Sonderfälle:
        Fehlende Dateien, Lesefehler, leere Daten oder fehlende Pflichtspalten
        führen zu einem leeren DataFrame.

    Projektkontext:
        Die Funktion erweitert den Datensatz optional um externe urbane
        Aktivitätsindikatoren, die für Fahrgastaufkommen relevant sein können.
    """
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
    """
    Prüft schnell, ob eine Parquet-Datei das erwartete Mindestschema besitzt.

    Die Funktion nutzt `pyarrow.parquet.ParquetFile(...).schema.names`, um ohne
    vollständiges Einlesen zu kontrollieren, ob die notwendige Spalte
    `vehicle_station` vorhanden ist. [web:187]

    Parameter:
        path (Path): Pfad zur zu prüfenden Parquet-Datei.

    Rückgabewerte:
        bool: True bei vorhandenem Mindestschema, sonst False.

    Fehler/Sonderfälle:
        Jegliche Lesefehler oder ungültige Dateien führen zu False.

    Projektkontext:
        Die Funktion dient als frühe Qualitäts- und Formatprüfung, um
        ungeeignete Dateien effizient aus dem ETL-Prozess auszuschließen.
    """
    try:
        return REQUIRED_SCHEMA_COLUMN in pq.ParquetFile(path).schema.names
    except Exception:
        return False



def filter_2025(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Filtert einen Rohdatensatz auf das gewünschte Jahr.

    Die Funktion leitet zunächst eine konsolidierte Datumsspalte aus
    `main_date_day` oder ersatzweise `report_date` ab und behält anschließend
    nur Zeilen des Zieljahres bei.

    Parameter:
        frame (pd.DataFrame): Rohdatensatz.
        year (int): Zieljahr der Filterung.

    Rückgabewerte:
        pd.DataFrame: Auf das Zieljahr reduzierter DataFrame.

    Fehler/Sonderfälle:
        Nicht interpretierbare Datumswerte werden zu `NaT` bzw. `NaN` und beim
        Jahresfilter ausgeschlossen.

    Projektkontext:
        Die Funktion sorgt dafür, dass die weitere Verarbeitung strikt auf den
        fachlich vorgesehenen Archivzeitraum eingeschränkt bleibt.
    """
    frame = frame.copy()
    date_source = "main_date_day" if "main_date_day" in frame.columns else "report_date"
    frame["date"] = pd.to_datetime(frame[date_source], errors="coerce").dt.date
    return frame[frame["date"].map(lambda value: pd.notna(value) and value.year == year)].copy()



def add_hour_column(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Leitet Stunden- und Minutenmerkmale aus verfügbaren Zeitstempeln ab.

    Die Funktion durchsucht mehrere potenzielle Zeitspalten, übernimmt die erste
    brauchbare Quelle und berechnet daraus sowohl die Stunde als auch die
    Abfahrtsminute des Tages.

    Parameter:
        frame (pd.DataFrame): Eingabedatenrahmen mit Zeitstempeln.

    Rückgabewerte:
        pd.DataFrame: DataFrame mit ergänzten Zeitmerkmalen.

    Fehler/Sonderfälle:
        Wird keine brauchbare Zeitquelle gefunden, wird `hour` auf 0 gesetzt
        und `departure_minute_of_day` aus dieser Stunde abgeleitet. Die Minute
        des Tages wird auf den Bereich 0 bis 1439 begrenzt.

    Projektkontext:
        Die Funktion erzeugt zentrale zeitliche Features für spätere
        Aggregation, Kontextverknüpfung und ML-Modellierung.
    """
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
    """
    Bereinigt Rohdaten, entfernt Dubletten und imputiert fehlende Werte.

    Die Funktion ist das Herzstück der technischen Datenbereinigung. Sie
    entfernt doppelte Ereignisse, neutralisiert unplausible negative Werte,
    berechnet Fahrgaständerungen, imputiert Besetzungs- und
    Auslastungsinformationen und füllt fehlende numerische wie kategoriale
    Felder robust auf.

    Parameter:
        frame (pd.DataFrame): Zu bereinigender Rohdatensatz.

    Rückgabewerte:
        tuple[pd.DataFrame, int, int, int]: Bereinigter DataFrame, Anzahl
        entfernter Dubletten, Anzahl korrigierter negativer Werte und Anzahl
        gefüllter Fehlwerte.

    Fehler/Sonderfälle:
        Existiert `stop_event_id`, wird dedupliziert auf Ereignisebene,
        andernfalls auf vollständigen Zeilen. Nicht interpretierbare Werte werden
        zunächst in `NaN` überführt und dann imputiert.

    Projektkontext:
        Die Funktion erzeugt aus operativen Rohdaten eine stabile fachliche
        Datengrundlage für Analyse- und Modellierungszwecke.
    """
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
    """
    Imputiert die Abfahrtsbelegung entlang einer Fahrtfolge.

    Die Funktion nutzt, sofern verfügbar, die kumulierte Fahrgaständerung je
    Fahrt als fachlich motivierte Schätzung der Besetzung und propagiert
    bekannte Werte zusätzlich innerhalb derselben Fahrt per Vorwärts- und
    Rückwärtsfüllung.

    Parameter:
        frame (pd.DataFrame): Datenrahmen mit Fahrgaständerungen und ggf.
            Besetzungswerten.

    Rückgabewerte:
        pd.DataFrame: DataFrame mit imputierter `occupancy_departure`.

    Fehler/Sonderfälle:
        Fehlen die für die Kumulation nötigen Spalten, greift nur der
        Vorwärts-/Rückwärts-Fallback. Fehlende Restwerte werden auf 0 gesetzt.

    Projektkontext:
        Die Funktion rekonstruiert ein fachlich wichtiges Lastmerkmal, das für
        Kapazitätsbewertung und Modellierung relevant ist.
    """
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
    """
    Ergänzt einen Fahrtdatensatz um tägliche und stündliche Kontextmerkmale.

    Gemergt werden tägliche Kalender- und Eventmerkmale, stündliche
    Eventindikatoren sowie optional Straßen- und Temperaturdaten. Fehlende
    Kontextwerte werden anschließend systematisch auf 0 oder leere Strings gesetzt.

    Parameter:
        frame (pd.DataFrame): Bereinigter Fahrtdatensatz.
        daily_context (pd.DataFrame): Tageskontext.
        hourly_events (pd.DataFrame): Stundenkontext für Events.
        street_hourly (pd.DataFrame): Optionaler Straßenkontext.

    Rückgabewerte:
        pd.DataFrame: Kontextangereicherter Fahrtdatensatz.

    Fehler/Sonderfälle:
        Ist kein stündlicher Eventkontext vorhanden, werden `event_hour` und
        `concert_hour` direkt mit 0 ergänzt. Ein vorhandenes `weekday` wird bei
        Bedarf zu `weekday_name` umbenannt, um Konflikte mit numerischen
        Wochentagsmerkmalen zu vermeiden.

    Projektkontext:
        Die Funktion verbindet operative Fahrtdaten mit externen
        Einflussvariablen und macht daraus einen fachlich reichhaltigen Datensatz
        für Analyse und ML.
    """
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
    """
    Erzeugt den Zieldateinamen für den angereicherten Parquet-Export.

    Der Dateiname basiert auf dem Quellnamen und wird um einen standardisierten
    Suffix mit Jahr ergänzt.

    Parameter:
        source (Path): Quelldatei.
        output_dir (Path): Ausgabeverzeichnis.
        year (int): Jahr des Exports.

    Rückgabewerte:
        Path: Zielpfad der bereinigten Kontextdatei.

    Fehler/Sonderfälle:
        Keine explizite Fehlerbehandlung.

    Projektkontext:
        Die Funktion sorgt für konsistente und nachvollziehbare Benennung der
        ETL-Ausgabedateien.
    """
    stem = re.sub(r"\.parquet$", "", source.name)
    return output_dir / f"{stem}_clean_context_{year}.parquet"



def model_ready_output_name(source: Path, model_ready_dir: Path, year: int) -> Path:
    """
    Erzeugt den Zieldateinamen für den Model-Ready-Parquet-Export.

    Der Dateiname basiert auf dem Quellnamen und erhält einen Suffix, der den
    reduzierten, ML-tauglichen Charakter des Exports kennzeichnet.

    Parameter:
        source (Path): Quelldatei.
        model_ready_dir (Path): Zielverzeichnis für ML-Dateien.
        year (int): Jahr des Exports.

    Rückgabewerte:
        Path: Zielpfad der Model-Ready-Datei.

    Fehler/Sonderfälle:
        Keine explizite Fehlerbehandlung.

    Projektkontext:
        Die Funktion trennt fachlich den angereicherten Vollbestand vom
        nachgelagerten Modellierungsdatensatz.
    """
    stem = re.sub(r"\.parquet$", "", source.name)
    return model_ready_dir / f"{stem}_model_ready_{year}.parquet"



def normalize_key(value: object) -> str:
    """
    Normalisiert freie Textwerte zu einem stabilen technischen Schlüssel.

    Umlaute und Sonderzeichen werden vereinheitlicht, nicht-alphanumerische
    Zeichen durch Unterstriche ersetzt und Leer- bzw. Nullfälle robust auf
    `unknown` abgebildet.

    Parameter:
        value (object): Beliebiger Eingabewert, typischerweise ein Stationsname.

    Rückgabewerte:
        str: Normalisierter Schlüsselwert.

    Fehler/Sonderfälle:
        `None` oder leer wirkende Werte werden auf `unknown` abgebildet.

    Projektkontext:
        Die Funktion erzeugt robuste, systemverträgliche Schlüssel für
        kategoriale ML-Merkmale wie Haltestellen.
    """
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
    """
    Reduziert einen angereicherten Datensatz auf ein ML-taugliches Exportschema.

    Die Funktion ergänzt weitere Zeitmerkmale, erzeugt stabile Stationsschlüssel,
    wählt nur die für Modellierung vorgesehenen Spalten aus und setzt für alle
    Merkmale eine explizite, pipelinesichere Typisierung durch. `to_parquet(...)`
    kann den so erzeugten DataFrame anschließend effizient als Parquet-Datei
    persistieren. [web:176]

    Parameter:
        enriched (pd.DataFrame): Vollständig bereinigter und angereicherter Datensatz.
        year (int): Zieljahr; dient unter anderem als Fallback-Datum.

    Rückgabewerte:
        pd.DataFrame: Reduzierter, typsicherer und fehlwertfreier Model-Ready-Datensatz.

    Fehler/Sonderfälle:
        Es werden nur tatsächlich vorhandene Spalten exportiert. Am Ende wird
        bewusst eine starke Garantie erzwungen: Es verbleiben keine fehlenden
        Werte im Model-Ready-Export.

    Projektkontext:
        Die Funktion bildet den Übergang von allgemeiner Datenaufbereitung zur
        konsistenten ML-Datenbasis für Trainings- und Prognosepipelines.
    """
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


    # Starke Garantie fuer nachgelagerte ML-Pipelines: kein Missing Value im Model-Ready-Export.
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
    """
    Verarbeitet eine einzelne Parquet-Quelldatei vollständig durch die ETL-Pipeline.

    Die Funktion prüft zunächst, ob die Datei verarbeitet werden kann und muss,
    liest sie dann ein, filtert auf das Zieljahr, bereinigt und imputiert Werte,
    reichert Kontext an und schreibt anschließend den Voll- sowie optional den
    Model-Ready-Export als Parquet-Dateien. Parquet-Export über
    `DataFrame.to_parquet(...)` unterstützt dabei Engines wie `pyarrow` und
    Kompression wie `snappy`. [web:176]

    Parameter:
        source (Path): Quelldatei.
        output_dir (Path): Zielverzeichnis für den angereicherten Export.
        model_ready_dir (Path | None): Optionales Zielverzeichnis für den
            Model-Ready-Export.
        daily_context (pd.DataFrame): Tageskontext.
        hourly_events (pd.DataFrame): Stündlicher Eventkontext.
        street_hourly (pd.DataFrame): Optionaler Straßenkontext.
        year (int): Zieljahr.
        overwrite (bool): Steuert, ob bestehende Dateien überschrieben werden.

    Rückgabewerte:
        ProcessStats: Statistiken und Status zur Verarbeitung der Datei.

    Fehler/Sonderfälle:
        Existieren Zielartefakte bereits und `overwrite` ist False, wird die
        Datei übersprungen. Dateien ohne erforderliches Schema oder ohne Zeilen
        des Zieljahres werden ebenfalls übersprungen.

    Projektkontext:
        Die Funktion ist die operative Kerneinheit der dateibasierten
        Datenpipeline und verbindet alle Teilfunktionen zu einem vollständigen
        Verarbeitungsablauf.
    """
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
    """
    Liefert die zu verarbeitenden Kandidatendateien für ein Jahr.

    Der Dateiname dient hier als schneller Vorfilter, um den Laufzeitaufwand zu
    reduzieren. Die fachlich verbindliche Jahresfilterung erfolgt dennoch später
    zeilenbasiert innerhalb der Daten.

    Parameter:
        input_dir (Path): Eingabeverzeichnis mit Parquet-Dateien.
        year (int): Zieljahr.

    Rückgabewerte:
        Iterable[Path]: Sortierte Iterable passender Kandidatendateien.

    Fehler/Sonderfälle:
        Keine explizite Fehlerbehandlung.

    Projektkontext:
        Die Funktion reduziert den Suchraum des Batch-Laufs und verbessert die
        Effizienz der Vorverarbeitung.
    """
    # Dateinamenfilter reduziert die Laufzeit; die zeilenweise Filterung garantiert spaeter trotzdem das Zieljahr.
    return sorted(input_dir.glob(f"*{year}*.parquet"))



def main() -> None:
    """
    Führt den vollständigen Batch-Lauf der Datenaufbereitung aus.

    Die Funktion parst die CLI-Argumente, baut alle benötigten Kontextdaten auf,
    iteriert über passende Quelldateien, verarbeitet jede Datei einzeln und
    schreibt abschließend ein Manifest mit Laufstatistiken als JSON-Datei.

    Parameter:
        Keine.

    Rückgabewerte:
        None: Die Funktion steuert den ETL-Lauf und schreibt Artefakte auf das Dateisystem.

    Fehler/Sonderfälle:
        Werden keine passenden Eingabedateien gefunden, wird das Programm mit
        `SystemExit` abgebrochen. Das Manifest wird sowohl im Voll-Export als
        auch optional im Model-Ready-Verzeichnis abgelegt.

    Projektkontext:
        `main()` ist der Kommandozeilen-Einstiegspunkt der gesamten
        Aufbereitungspipeline und damit das Bindeglied zwischen Roharchiv,
        Feature Engineering und ML-Export.
    """
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