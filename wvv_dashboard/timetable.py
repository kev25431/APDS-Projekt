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

"""
Modul zur Erstellung und zum Vergleich von WVV- und adaptiven Fahrplantabellen.

Das Modul liest bereinigte Fahrplandaten ein, wählt für eine Linie und einen
Betriebstag den fachlich passendsten WVV-Fahrplan aus und erzeugt daraus eine
tabellarische Referenz. Optional kann zusätzlich ein prognosebasierter,
adaptiver Fahrplan konstruiert werden, dessen Fahrtenzahl pro Stunde durch die
Service-Policy bestimmt wird.

Projektkontext:
    timetable.py bildet die Brücke zwischen bereinigten WVV-Fahrplandaten,
    Nachfrageprognose und GUI-Darstellung. Es setzt die im Projekt beschriebene
    Logik um, nach der nicht die Route selbst neu optimiert wird, sondern die
    Anzahl und zeitliche Verteilung der Fahrten auf Basis historischer und
    prognostizierter Nachfrage angepasst wird.
"""

SERVICE_LABELS = {
    "weekday": "Montag bis Freitag",
    "saturday": "Samstag",
    "sunday_holiday": "Sonntag / Feiertag",
    "unknown": "Unbekannter Gültigkeitstyp",
}


def station_key(value: object) -> str:
    """
    Normalisiert einen Haltestellennamen zu einem robusten Schlüssel.

    Der Text wird ASCII-normalisiert, in Kleinbuchstaben überführt und von
    Sonderzeichen bereinigt. Zusätzlich werden Varianten von „Straße/Str.“
    vereinheitlicht, damit Haltestellen aus verschiedenen Datenquellen besser
    zusammengeführt werden können.

    Parameter:
        value (object): Ursprünglicher Haltestellenwert.

    Rückgabewerte:
        str: Normalisierter, nur aus Buchstaben und Ziffern bestehender Schlüssel.

    Projektkontext:
        Die Funktion ist zentral für das Matching zwischen Fahrplan-Longformat,
        Routentabellen und ggf. weiteren Datenquellen, deren Haltestellennamen
        leicht voneinander abweichen können.
    """
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("strasse", "str").replace("str.", "str")
    return re.sub(r"[^a-z0-9]+", "", text)


def format_minutes(value: float | int | None) -> str:
    """
    Formatiert eine Minutenangabe des Betriebstags als Uhrzeitstring.

    Der Wert wird auf ganze Minuten gerundet und modulo 24 Stunden dargestellt,
    sodass auch Werte außerhalb des Bereichs 0 bis 1439 in ein Uhrzeitformat
    überführt werden können.

    Parameter:
        value (float | int | None): Minutenwert seit Tagesbeginn.

    Rückgabewerte:
        str: Uhrzeit im Format ``HH:MM`` oder ``-`` bei fehlendem Wert.

    Fehler/Sonderfälle:
        ``None`` und NaN-Werte werden als ``-`` dargestellt.

    Projektkontext:
        Die Funktion wird für die tabellarische Ausgabe von WVV- und adaptiven
        Fahrplanzeiten in der GUI verwendet.
    """
    if value is None or pd.isna(value):
        return "-"
    minutes = int(round(float(value))) % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


@dataclass
class TimetableComparison:
    """
    Datencontainer für einen vollständigen Fahrplanvergleich.

    Das Objekt enthält sowohl Metadaten zum ausgewählten Fahrplan als auch die
    eigentlichen Vergleichstabellen für den WVV-Bestand und den adaptiven
    Fahrplan. Zusätzlich werden Route, Kurzsummary und ein Kennzeichen für die
    Nutzung von Prognosedaten mitgeführt.

    Projektkontext:
        TimetableComparison ist das zentrale Austauschformat zwischen
        TimetableRepository und GUI. Es erlaubt, alle für die Darstellung des
        Fahrplanvergleichs benötigten Informationen in einer Struktur zu bündeln.
    """

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
    """
    Repository für bereinigte Fahrplandaten und Fahrplanvergleiche.

    Die Klasse kapselt das Laden, Caching und die Auswahl von Fahrplandaten aus
    dem bereinigten Clean-Verzeichnis. Sie bietet Methoden, um verfügbare Linien
    zu ermitteln, WVV-Abfahrten stündlich zu zählen und vollständige Vergleiche
    zwischen WVV-Fahrplan und adaptiver Planung aufzubauen.

    Projektkontext:
        Das Repository bildet die fahrplanbezogene Datenzugriffsschicht des
        Projekts. Es wird insbesondere von der GUI verwendet, um für gewählte
        Linien und Tage einen fachlich plausiblen Vergleichsfahrplan zu erzeugen.
    """

    def __init__(self, clean_dir: Path = TIMETABLE_CLEAN_DIR) -> None:
        """
        Initialisiert das Repository mit einem Verzeichnis bereinigter Fahrplandaten.

        Parameter:
            clean_dir (Path): Verzeichnis mit ``fahrplan_long``- und
                ``fahrplan_routes``-Dateien im CSV- oder Parquet-Format.

        Projektkontext:
            Über das konfigurierbare Verzeichnis kann das Repository sowohl mit
            produktiven Fahrplandaten als auch mit Test- oder Alternativständen
            arbeiten.
        """
        self.clean_dir = clean_dir
        self._long: pd.DataFrame | None = None
        self._routes: pd.DataFrame | None = None
        self._holiday_dates: set[date] | None = None

    def available_lines(self) -> list[int]:
        """
        Liefert alle in den bereinigten Fahrplandaten verfügbaren Linien.

        Rückgabewerte:
            list[int]: Sortierte Liste verfügbarer Liniennummern.

        Fehler/Sonderfälle:
            Bei fehlenden oder leeren Fahrplandaten wird eine leere Liste
            zurückgegeben.

        Projektkontext:
            Die Methode speist Auswahlmenüs und Prüflogik in der GUI und macht
            sichtbar, für welche Linien überhaupt tabellarische Vergleichsdaten
            vorliegen.
        """
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
        """
        Zählt WVV-Abfahrten einer Linie je Stunde für einen gewählten Betriebstag.

        Zunächst wird der zur Tagesart passende Service-Key bestimmt. Anschließend
        werden die Fahrten der fachlich besten Planfassung ausgewählt und pro
        Startstunde gezählt. Grundlage ist jeweils die erste Abfahrtsminute einer
        Fahrt.

        Parameter:
            line (int): Ziel-Linie.
            selected_day (date): Gewählter Betriebstag.
            start_hour (int): Startstunde des Betrachtungsfensters.
            horizon_hours (int): Länge des Betrachtungsfensters in Stunden.

        Rückgabewerte:
            dict[int, int]: Abbildung Stunde -> Anzahl WVV-Abfahrten.

        Fehler/Sonderfälle:
            Falls keine passenden Fahrplandaten vorliegen, wird ein leeres
            Dictionary zurückgegeben.
            Berücksichtigt werden nur Fahrten, deren erste Abfahrt innerhalb des
            gewählten Stundenfensters liegt.

        Projektkontext:
            Die Methode dient unter anderem als Input für KPI-Vergleiche, bei
            denen reale WVV-Baseline-Fahrten pro Stunde benötigt werden.
        """
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
        """
        Lädt das bereinigte Fahrplan-Longformat lazy und mit Cache.

        Rückgabewerte:
            pd.DataFrame: Fahrplandaten im Longformat.

        Projektkontext:
            Das Longformat ist die Hauptdatenbasis für Fahrten, Minutenwerte und
            spätere Tabellenkonstruktion.
        """
        if self._long is None:
            self._long = self._read_clean_file("fahrplan_long")
        return self._long

    @property
    def routes(self) -> pd.DataFrame:
        """
        Lädt die bereinigten Routendaten lazy und mit Cache.

        Rückgabewerte:
            pd.DataFrame: Routendaten mit Haltestellenreihenfolge.

        Projektkontext:
            Die Routendaten werden genutzt, um Haltestellenfolgen für die
            Vergleichstabellen sauber und konsistent darzustellen.
        """
        if self._routes is None:
            self._routes = self._read_clean_file("fahrplan_routes")
        return self._routes

    def _read_clean_file(self, stem: str) -> pd.DataFrame:
        """
        Liest eine bereinigte Fahrplandatei aus Parquet oder CSV.

        Die Funktion bevorzugt Parquet und fällt bei Bedarf auf CSV zurück.
        Relevante Spalten werden typisiert und fehlende ``station_key``-Werte aus
        dem Haltestellennamen erzeugt.

        Parameter:
            stem (str): Dateistamm, z. B. ``fahrplan_long`` oder ``fahrplan_routes``.

        Rückgabewerte:
            pd.DataFrame: Eingelesene und grundlegend typisierte Fahrplandaten.

        Fehler/Sonderfälle:
            Existiert weder Parquet- noch CSV-Datei, wird ein leerer DataFrame
            zurückgegeben.

        Projektkontext:
            Die Funktion ist der zentrale Einlesepunkt für die aus
            ``clean_timetables.py`` hervorgehenden Clean-Dateien.
        """
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
        """
        Bestimmt den Service-Key für einen Kalendertag.

        Feiertage werden wie Sonntag/Feiertag behandelt. Ansonsten erfolgt die
        Einordnung nach Wochentag in Werktag, Samstag oder Sonntag/Feiertag.

        Parameter:
            selected_day (date): Zu klassifizierender Tag.

        Rückgabewerte:
            str: Service-Key für die Fahrplanauswahl.

        Projektkontext:
            Die Funktion steuert, welche Fahrplanvariante für einen Tag als
            fachlich passend angesehen wird.
        """
        if selected_day in self._public_holiday_dates():
            return "sunday_holiday"
        if selected_day.weekday() < 5:
            return "weekday"
        if selected_day.weekday() == 5:
            return "saturday"
        return "sunday_holiday"

    def _public_holiday_dates(self) -> set[date]:
        """
        Lädt und cached bayerische Feiertage aus der Zusatzdatenbasis.

        Rückgabewerte:
            set[date]: Menge aller als Feiertag markierten Kalendertage.

        Fehler/Sonderfälle:
            Bei fehlender oder nicht lesbarer Datei wird eine leere Menge
            zurückgegeben.

        Projektkontext:
            Feiertage beeinflussen direkt die Wahl des passenden Fahrplantyps und
            damit den späteren Vergleich zwischen WVV- und adaptivem Fahrplan.
        """
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
        """
        Erstellt einen vollständigen Vergleich zwischen WVV- und adaptivem Fahrplan.

        Die Methode wählt zunächst den für Linie, Service-Key und Zeitfenster besten
        WVV-Plan aus. Danach werden Route, WVV-Tabelle, Reisezeit-Offsets und eine
        adaptive Vergleichstabelle aufgebaut. Liegt ein Prognose-Schedule vor,
        fließt er in die Bestimmung der Fahrtenanzahl je Stunde ein.

        Parameter:
            line (int): Ziel-Linie.
            selected_day (date): Gewählter Betriebstag.
            start_hour (int): Startstunde des Betrachtungsfensters.
            horizon_hours (int): Länge des Betrachtungsfensters in Stunden.
            prediction_schedule (pd.DataFrame | None): Optionaler Prognose-Schedule
                für adaptive Fahrten.
            max_trips (int): Obergrenze der in Tabellen dargestellten Fahrten.

        Rückgabewerte:
            TimetableComparison: Vollständiges Vergleichsobjekt.

        Fehler/Sonderfälle:
            Falls keine bereinigten Fahrplandaten oder keine passende Tabelle
            vorhanden sind, wird ein inhaltsleeres Vergleichsobjekt mit Hinweistext
            zurückgegeben.

        Projektkontext:
            Dies ist die zentrale Orchestrierungsmethode des Moduls und der
            wichtigste Einstiegspunkt für die Fahrplanansicht in der GUI.
        """
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
        """
        Erzeugt ein leeres Vergleichsobjekt mit Hinweistext.

        Parameter:
            line (int): Ziel-Linie.
            selected_day (date): Bezugstag.
            message (str): Fehl- oder Hinweistext.

        Rückgabewerte:
            TimetableComparison: Minimal befülltes Vergleichsobjekt.

        Projektkontext:
            Die Methode sorgt dafür, dass die GUI auch bei fehlenden Daten eine
            stabile und erklärbare Antwortstruktur erhält.
        """
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
        """
        Wählt die fachlich beste Fahrplantabelle für Linie, Service-Typ und Zeitfenster.

        Die Auswahl erfolgt mehrstufig: Zunächst werden passende Service-Typen
        priorisiert, anschließend Tabellen nach Deckung des Zeitfensters,
        Aktualität, Anzahl Haltestellen, Anzahl Fahrten und Zeilenumfang bewertet.
        Tabellen derselben Planfassung mit identischer Route werden zusammengeführt.

        Parameter:
            line (int): Ziel-Linie.
            service_key (str): Bevorzugter Service-Key.
            start_hour (int): Startstunde des betrachteten Fensters.
            horizon_hours (int): Länge des Fensters in Stunden.

        Rückgabewerte:
            pd.DataFrame: Ausgewählte und ggf. zusammengeführte Tabellenzeilen.

        Fehler/Sonderfälle:
            Wenn keine geeignete Tabelle gefunden wird, wird ein leerer DataFrame
            zurückgegeben.
            Die Kandidatenreihenfolge enthält Fallbacks wie ``unknown`` und weitere
            Service-Typen, um auch bei unvollständigen Daten einen nutzbaren Plan
            zu finden.

        Projektkontext:
            Diese Auswahlregel setzt die in der Projektbeschreibung formulierte
            Logik um, nach der nicht blind irgendein Fahrplan verwendet wird,
            sondern der fachlich plausibelste Plan für den gewünschten Tag.
        """
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
        """
        Reduziert einen Dateinamen auf einen planbezogenen Schlüssel.

        Zusätze wie Tabellen- oder Sheet-Suffixe werden entfernt, sodass mehrere
        CSV-Extrakte derselben Planfassung als zusammengehörig erkannt werden.

        Parameter:
            source_file (str): Ursprünglicher Quelldateiname.

        Rückgabewerte:
            str: Planbezogener Schlüssel.

        Projektkontext:
            Die Methode ist wichtig, um Fahrplantabellen derselben Quelle
            zusammenzuführen, wenn sie aus mehreren Blättern oder Teildateien
            stammen.
        """
        return re.sub(r"__(table|sheet)_[^/\\]+\.csv$", "", str(source_file))

    def _best_plan_key(self, frame: pd.DataFrame) -> str:
        """
        Bestimmt den besten Plan-Schlüssel innerhalb eines Tabellenbestands.

        Bewertet werden Aktualität, Anzahl Fahrten, Umfang der Zeilen und Anzahl
        beteiligter Tabellen.

        Parameter:
            frame (pd.DataFrame): Kandidatenbestand.

        Rückgabewerte:
            str: Plan-Schlüssel der besten Planfassung oder leerer String.

        Fehler/Sonderfälle:
            Bei leerem oder nicht auswertbarem Bestand wird ein leerer String
            zurückgegeben.

        Projektkontext:
            Diese Hilfsfunktion unterstützt die konsistente Auswahl einer
            zusammenhängenden Planfassung für Linienauswertungen.
        """
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
        """
        Erzeugt eine Signatur der Haltestellenfolge einer Tabelle.

        Grundlage ist die sortierte und deduplizierte Reihenfolge der
        ``station_key``-Werte.

        Parameter:
            table (pd.DataFrame): Tabellenbestand einer Fahrplanquelle.

        Rückgabewerte:
            tuple[str, ...]: Unveränderliche Routensignatur.

        Projektkontext:
            Die Signatur erlaubt es, Tabellen derselben Planfassung nur dann
            zusammenzuführen, wenn sie tatsächlich dieselbe Route beschreiben.
        """
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
        """
        Erzeugt ein lesbares Quellenlabel für die GUI.

        Bei einer einzelnen Quelle wird der Dateiname ausgegeben. Bei mehreren
        zusammengeführten Tabellen wird der Planname samt Anzahl der Tabellen
        dargestellt.

        Parameter:
            selected (pd.DataFrame): Ausgewählte Tabellenzeilen.

        Rückgabewerte:
            str: Quellenlabel für die Ausgabe.

        Projektkontext:
            Das Label macht transparent, auf welcher Planquelle der Vergleich
            basiert.
        """
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
        """
        Ermittelt die Haltestellenfolge für die ausgewählte Fahrplantabelle.

        Wenn passende Routendaten vorliegen, werden diese bevorzugt verwendet.
        Andernfalls wird die Route direkt aus den ausgewählten Longdaten rekonstruiert.

        Parameter:
            line (int): Ziel-Linie.
            source_file (str): Quellenbezeichnung.
            table_no (int): Tabellennummer.
            service_key (str): Service-Key der Tabelle.
            selected (pd.DataFrame): Ausgewählte Longdaten.

        Rückgabewerte:
            pd.DataFrame: Route mit Haltestellenfolge, Schlüsseln und Markern.

        Fehler/Sonderfälle:
            Fehlende Routendaten werden durch Rekonstruktion aus dem Longformat
            abgefangen.

        Projektkontext:
            Die Route wird benötigt, um sowohl WVV- als auch adaptive Fahrten
            entlang derselben Haltestellenfolge abzubilden.
        """
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
        """
        Baut die WVV-Vergleichstabelle sowie Reisezeit-Offsets und Abfahrtsliste.

        Ausgewählt werden zunächst die ersten Abfahrtszeiten pro Fahrt, wobei
        Dubletten mit gleicher gerundeter Startminute zugunsten der besser
        belegten Fahrt entfernt werden. Danach werden die Fahrten auf das
        gewünschte Zeitfenster begrenzt und als Matrix über die Route aufgebaut.

        Parameter:
            selected (pd.DataFrame): Ausgewählte Fahrplan-Langdaten.
            route (pd.DataFrame): Haltestellenfolge der Linie.
            start_hour (int): Startstunde des Betrachtungsfensters.
            horizon_hours (int): Länge des Fensters in Stunden.
            max_trips (int): Maximale Anzahl darzustellender Fahrten.

        Rückgabewerte:
            tuple[pd.DataFrame, dict[str, float], list[dict[str, object]]]:
                WVV-Tabelle, stationenbezogene Reisezeit-Offsets und Liste der
                berücksichtigten Abfahrten.

        Fehler/Sonderfälle:
            Bei leeren Eingaben werden leere Strukturen zurückgegeben.
            Wenn innerhalb des Zeitfensters keine Fahrt liegt, werden ersatzweise
            die ersten verfügbaren Fahrten verwendet.

        Projektkontext:
            Die Methode setzt die im Projekt beschriebene Logik des originalen
            WVV-Fahrplans um, auf dessen realen Abfahrtslagen und Reisezeiten der
            Vergleich aufbaut.
        """
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
        """
        Ermittelt typische Reisezeit-Offsets je Haltestelle.

        Für jede Fahrt wird zunächst die erste Abfahrtsminute bestimmt. Danach
        wird je Haltestelle der Median des Offsets zur ersten Abfahrt berechnet.
        Die resultierenden Offsets werden entlang der Route monotonisiert, damit
        keine zeitlichen Rücksprünge entstehen.

        Parameter:
            selected (pd.DataFrame): Fahrplan-Langdaten.
            route (pd.DataFrame): Haltestellenfolge.
            departures (list[dict[str, object]]): Verwendete Abfahrtsliste.

        Rückgabewerte:
            dict[str, float]: Abbildung station_key -> typischer Minutenoffset.

        Fehler/Sonderfälle:
            Bei fehlenden Daten wird ein leeres Dictionary zurückgegeben.
            Nicht vorhandene Offsets werden entlang der Route durch den letzten
            bekannten Wert ersetzt.

        Projektkontext:
            Die Offsets sind entscheidend dafür, dass adaptive Fahrten dieselbe
            reale Fahrzeitstruktur wie der WVV-Fahrplan beibehalten.
        """
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
        """
        Baut aus Route und Abfahrtsliste eine tabellarische Fahrplanmatrix.

        Für jede berücksichtigte Fahrt wird eine Spalte erzeugt. Die Werte je
        Haltestelle stammen aus den echten Fahrplanzeiten des ausgewählten
        WVV-Fahrplans.

        Parameter:
            selected (pd.DataFrame): Fahrplan-Langdaten.
            route (pd.DataFrame): Haltestellenfolge.
            departures (list[dict[str, object]]): Fahrten, die als Spalten
                dargestellt werden sollen.

        Rückgabewerte:
            pd.DataFrame: Matrixförmige Fahrplantabelle.

        Projektkontext:
            Die Methode erstellt die Referenztabelle, die in der GUI dem
            adaptiven Fahrplan gegenübergestellt wird.
        """
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
        """
        Baut eine adaptive Fahrplantabelle für das betrachtete Zeitfenster.

        Die Fahrtenzahl wird stundenweise anhand des Prognose-Schedules und der
        Service-Policy bestimmt. Anschließend werden bestehende WVV-Fahrten
        übernommen, reduziert oder durch synthetische Abfahrten ergänzt. Die
        Zeiten an den Folgestationen werden über die zuvor berechneten Offsets
        fortgeschrieben.

        Parameter:
            line (int): Ziel-Linie.
            route (pd.DataFrame): Haltestellenfolge.
            offsets (dict[str, float]): Reisezeit-Offsets je Haltestelle.
            wvv_departures (list[dict[str, object]]): WVV-Abfahrten als Ausgangsbasis.
            prediction_schedule (pd.DataFrame | None): Optionaler Prognose-Schedule.
            selected_day (date): Gewählter Betriebstag.
            start_hour (int): Startstunde des Betrachtungsfensters.
            horizon_hours (int): Länge des Fensters in Stunden.
            max_trips (int): Maximale Anzahl darzustellender Fahrten.

        Rückgabewerte:
            tuple[pd.DataFrame, list[dict[str, object]]]:
                Adaptive Tabelle und Liste der erzeugten/übernommenen Abfahrten.

        Fehler/Sonderfälle:
            Bei leerer Route wird keine Tabelle erzeugt.
            Neue Fahrten außerhalb des Service-Spans werden nur zugelassen, wenn
            der Service-Spannentest positiv ausfällt oder bereits echte Fahrten
            in der Stunde existieren.

        Projektkontext:
            Diese Methode setzt die fachliche Kernidee der adaptiven
            Fahrplanerstellung um: gleiche Route und Reisezeiten, aber angepasste
            Fahrtenzahl und zeitliche Verteilung.
        """
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
        """
        Bestimmt die stündliche Service-Spanne aus vorhandenen Abfahrten.

        Parameter:
            departures (list[dict[str, object]]): Liste bekannter Abfahrten.

        Rückgabewerte:
            tuple[int, int] | None: Früheste und späteste Stunde oder ``None``.

        Fehler/Sonderfälle:
            Bei leerer Abfahrtsliste wird ``None`` zurückgegeben.

        Projektkontext:
            Die Service-Spanne dient als Schutz davor, adaptive Zusatzfahrten
            außerhalb des üblichen Betriebsfensters unkontrolliert zu erzeugen.
        """
        if not departures:
            return None
        minutes = [int(float(item["minute"])) for item in departures]
        return min(minutes) // 60, max(minutes) // 60

    def _hour_inside_service_span(self, hour: int, service_hours: tuple[int, int] | None) -> bool:
        """
        Prüft, ob eine Stunde innerhalb der bekannten Service-Spanne liegt.

        Die Methode berücksichtigt auch Spannen, die über Mitternacht laufen.

        Parameter:
            hour (int): Zu prüfende Stunde.
            service_hours (tuple[int, int] | None): Früheste und späteste Stunde.

        Rückgabewerte:
            bool: ``True`` bei Zugehörigkeit zur Service-Spanne.

        Fehler/Sonderfälle:
            Liegt keine Service-Spanne vor, wird ``True`` zurückgegeben, um keine
            unnötige Sperre für adaptive Fahrten zu setzen.

        Projektkontext:
            Die Logik ist relevant, damit adaptive Fahrten nur in plausiblen
            Betriebszeiten ergänzt werden.
        """
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
        """
        Indexiert Prognose-Schedule-Zeilen nach Stunde für einen gewählten Tag.

        Parameter:
            prediction_schedule (pd.DataFrame | None): Prognosedaten.
            selected_day (date): Zieltag.
            start_hour (int): Startstunde des Fensters.
            horizon_hours (int): Länge des Fensters.

        Rückgabewerte:
            dict[int, pd.Series]: Abbildung Stunde -> Schedule-Zeile.

        Fehler/Sonderfälle:
            Fehlende oder leere Prognosedaten führen zu einem leeren Dictionary.
            Berücksichtigt werden nur Zeilen des gewählten Tages und des
            betrachteten Stundenfensters.

        Projektkontext:
            Die Methode stellt die stündliche Prognose in der Form bereit, in der
            sie für die adaptive Fahrtenentscheidung benötigt wird.
        """
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
        """
        Bestimmt die adaptive Ziel-Fahrtenzahl für eine Stunde.

        Falls keine Prognosezeile vorliegt, wird die bestehende WVV-Fahrtenzahl
        beibehalten. Andernfalls wird die Nachfrage an die zentrale
        Service-Policy übergeben. Primär wird ``predicted_boardings`` genutzt;
        falls dieser Wert fehlt, wird aus ``predicted_demand`` heuristisch ein
        Einsteigeranteil von 55 Prozent angenommen.

        Parameter:
            schedule_row (pd.Series | None): Prognosezeile der Stunde.
            existing_count (int): Anzahl vorhandener WVV-Fahrten.
            allow_new_service (bool): Kennzeichen, ob neue Fahrten zulässig sind.
            line (int | None): Liniennummer.
            hour (int | None): Zielstunde.

        Rückgabewerte:
            int: Empfohlene Fahrtenzahl für die Stunde.

        Fehler/Sonderfälle:
            Wenn keine Prognose vorliegt, wird mindestens der bestehende Bestand
            zurückgegeben.
            Wenn keine neue Bedienung erlaubt ist und bislang keine Fahrt
            existiert, wird 0 zurückgegeben.

        Projektkontext:
            Die Methode verdichtet Prognose, Bestandsangebot, Kapazität und
            Service-Policy zu einer konkreten Stundenentscheidung.
        """
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
        """
        Wandelt einen Wert robust in ``float`` um.

        Parameter:
            value (object): Zu konvertierender Wert.
            default (float): Ersatzwert bei Fehler oder Nicht-Endlichkeit.

        Rückgabewerte:
            float: Gültiger Fließkommawert.

        Fehler/Sonderfälle:
            Nicht numerische oder nicht endliche Werte werden durch ``default``
            ersetzt.

        Projektkontext:
            Die Methode schützt die adaptive Stundenlogik vor unvollständigen oder
            fehlerhaften Prognosedaten.
        """
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
        """
        Erzeugt die Abfahrten einer Stunde für den adaptiven Fahrplan.

        Die Logik folgt drei Fällen:
        1. Zielanzahl entspricht dem Bestand -> Übernahme.
        2. Zielanzahl kleiner als Bestand -> gleichmäßige Auswahl bestehender Fahrten.
        3. Zielanzahl größer als Bestand -> Ergänzung synthetischer Fahrten mit
           möglichst gleichmäßiger Verteilung innerhalb der Stunde.

        Parameter:
            hour_start (int): Minutenwert des Stundenbeginns.
            existing (list[dict[str, object]]): Vorhandene WVV-Fahrten der Stunde.
            target (int): Gewünschte Zielanzahl.

        Rückgabewerte:
            list[dict[str, object]]: Fahrtenliste der Stunde.

        Fehler/Sonderfälle:
            Die Zielanzahl wird auf den Bereich 0 bis 12 begrenzt.
            Bei Reduktion auf genau eine Fahrt wird die mittlere bestehende Fahrt
            gewählt.
            Neue synthetische Fahrten erhalten ein ``trip_key``-Präfix
            ``adaptive:`` und das Kennzeichen ``synthetic=True``.

        Projektkontext:
            Diese Methode setzt die konkrete Taktlogik des adaptiven Fahrplans um,
            ohne die Route selbst zu verändern.
        """
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
        """
        Erzeugt einen erklärenden Kurztext zum Vergleichsergebnis.

        Der Text unterscheidet zwischen reinem Fallback ohne Prognosemodell,
        Angebotsausweitung, Angebotseinsparung und unveränderter Fahrtenzahl.

        Parameter:
            wvv_departures (list[dict[str, object]]): WVV-Abfahrten.
            adaptive_departures (list[dict[str, object]]): Adaptive Abfahrten.
            prediction_schedule (pd.DataFrame | None): Verwendeter Prognose-Schedule.

        Rückgabewerte:
            str: Kurzinterpretation des Vergleichs.

        Projektkontext:
            Der Hinweistext wird in der GUI verwendet, um dem Nutzer die Wirkung
            der adaptiven Planung in wenigen Sätzen zu erklären.
        """
        if prediction_schedule is None or prediction_schedule.empty:
            return "Kein trainiertes Prognosemodell geladen: adaptiver Plan nutzt WVV-Bestand als Fallback."
        delta = len(adaptive_departures) - len(wvv_departures)
        if delta > 0:
            return f"Adaptive Planung würde {delta} zusätzliche Fahrten über den Betriebstag anbieten."
        if delta < 0:
            return f"Adaptive Planung würde {abs(delta)} Fahrten über den Betriebstag einsparen."
        return "Adaptive Planung hält die Fahrtenzahl, verteilt sie aber prognosebasiert über den Betriebstag."