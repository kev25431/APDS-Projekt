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
    """
    Lädt die Trainingsbasis für die Nachfrageprognose.

    Der Loader priorisiert die bereinigten und mit Kontextmerkmalen
    angereicherten 2025-Parquet-Dateien. Falls diese nicht verfügbar sind,
    wird auf den allgemeinen Datenbestand des Dashboards zurückgegriffen.

    Parameter:
        repo (Any): Datenrepository der Anwendung mit Zugriff auf Linienbereiche
            und ggf. tägliche Kontextdaten.
        enriched_dir (Path): Verzeichnis der bevorzugten 2025-Enriched-Dateien.

    Rückgabewerte:
        TrainingDataLoader: Initialisierte Ladeinstanz.

    Fehler/Sonderfälle:
        Fehlende Verzeichnisse oder fehlerhafte Einzeldateien führen nicht zum
        Abbruch des gesamten Ladevorgangs. Stattdessen werden verfügbare Daten
        genutzt oder ein leerer Trainingscontainer zurückgegeben.

    Projektkontext:
        Die Klasse verbindet Datenvorverarbeitung und Modelltraining, indem sie
        der Prediction-Pipeline eine fachlich priorisierte Datenbasis liefert.
        Damit wird die in der Projektbeschreibung vorgesehene Nutzung des
        2025-Archivs als primäre Lernquelle umgesetzt.
    """

    def __init__(self, repo: Any, enriched_dir: Path = ENRICHED_TRAINING_DIR) -> None:
        """
        Initialisiert den Datenlader mit Repository und Zielverzeichnis.

        Parameter:
            repo (Any): Datenquelle der Anwendung.
            enriched_dir (Path): Speicherort der bereinigten Trainings-Parquets.

        Rückgabewerte:
            None: Die Methode setzt ausschließlich Objektzustand.

        Fehler/Sonderfälle:
            Keine unmittelbare Fehlerbehandlung erforderlich; Verfügbarkeit wird
            erst beim eigentlichen Ladevorgang geprüft.

        Projektkontext:
            Der Loader wird vom Vorhersageservice und von Trainingsskripten
            gemeinsam verwendet, um Datenkonsistenz sicherzustellen.
        """
        self.repo = repo
        self.enriched_dir = enriched_dir

    def load(self, lines: list[int], start: date, end: date) -> TrainingData:
        """
        Lädt Trainingsdaten für eine oder mehrere Linien.

        Zuerst wird versucht, passende Enriched-2025-Dateien zu laden. Nur wenn
        dort keine verwertbaren Daten gefunden werden, erfolgt der Fallback auf
        den allgemeinen Repository-Zugriff für den angegebenen Zeitraum.

        Parameter:
            lines (list[int]): Zu ladende Liniennummern.
            start (date): Startdatum des gewünschten Zeitraums.
            end (date): Enddatum des gewünschten Zeitraums.

        Rückgabewerte:
            TrainingData: Geladene Rohdaten inklusive Herkunftsangabe und
            Datumsabdeckung.

        Fehler/Sonderfälle:
            Bei vollständig fehlenden Daten wird ein leerer Trainingscontainer
            mit entsprechender Quellenbeschreibung zurückgegeben.

        Projektkontext:
            Die Methode setzt die fachliche Priorisierung der Projektübersicht
            direkt um: bevorzugt wird das bereinigte 2025-Archiv, weil dieses
            bereits für die Nachfrageprognose vorbereitet wurde.
        """
        enriched = self._load_enriched_2025(lines)
        if not enriched.empty:
            return TrainingData(
                raw=enriched,
                source="full_api_data_enriched_2025",
                date_span=(min(enriched["date"]), max(enriched["date"])),
            )

        return self._load_dashboard_repository(lines, start, end)

    def _load_dashboard_repository(self, lines: list[int], start: date, end: date) -> TrainingData:
        """
        Lädt Trainingsdaten direkt aus dem Datenrepository der Anwendung.

        Für jede angeforderte Linie wird der passende Zeitbereich geladen und
        anschließend zu einem gemeinsamen Rohdatensatz zusammengeführt. Zudem
        wird ein normalisierter Haltestellenschlüssel erzeugt.

        Parameter:
            lines (list[int]): Zu ladende Linien.
            start (date): Beginn des Zeitraums.
            end (date): Ende des Zeitraums.

        Rückgabewerte:
            TrainingData: Rohdaten aus dem App-Datenordner oder ein leerer
            Container bei fehlenden Ergebnissen.

        Fehler/Sonderfälle:
            Leere Teilergebnisse einzelner Linien werden ignoriert. Sind alle
            Ergebnisse leer, wird ein leerer TrainingData-Container erzeugt.

        Projektkontext:
            Dieser Pfad dient als technischer Fallback, falls die bevorzugten
            2025-Enriched-Dateien nicht vorliegen oder nicht verwendbar sind.
        """
        frames = [self.repo.load_line_range(line, start, end) for line in lines]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return TrainingData(pd.DataFrame(), "Keine Trainingsdaten", None)
        raw = pd.concat(frames, ignore_index=True).copy()
        raw["station_key"] = raw["station"].map(normalize_station_name)
        return TrainingData(raw, "App-Datenordner", (start, end))

    def _enriched_files_for_lines(self, lines: list[int]) -> list[Path]:
        """
        Ermittelt passende Enriched-Parquet-Dateien für gegebene Linien.

        Die Auswahl basiert auf einem Dateinamenschema, das die Liniennummer in
        der Datei kodiert. Nur Dateien im erwarteten Format werden berücksichtigt.

        Parameter:
            lines (list[int]): Gewünschte Liniennummern.

        Rückgabewerte:
            list[Path]: Gefundene Dateipfade im Enriched-Verzeichnis.

        Fehler/Sonderfälle:
            Existiert das Verzeichnis nicht, wird eine leere Liste geliefert.
            Dateien mit nicht passendem Namensschema werden ignoriert.

        Projektkontext:
            Die Methode koppelt die Trainingslogik bewusst an die erzeugten
            2025-Bestände aus der Datenvorverarbeitung und stellt damit den
            Übergang von Preprocessing zu Modelltraining her.
        """
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
        """
        Liest eine einzelne Enriched-Parquet-Datei robust ein.

        Die Methode versucht zunächst, nur die für das Modell relevanten Spalten
        zu laden. Falls das aufgrund eines abweichenden Schemas fehlschlägt, wird
        die vollständige Datei gelesen und fehlende Pflichtspalten werden ergänzt.

        Parameter:
            path (Path): Dateipfad zur Enriched-Parquet-Datei.

        Rückgabewerte:
            pd.DataFrame: Eingelesener und auf die benötigten Spalten
            projizierter Datenrahmen.

        Fehler/Sonderfälle:
            Fehlende Modellspalten werden mit 0 ergänzt, damit nachgelagerte
            Schritte ein vollständiges Schema vorfinden.

        Projektkontext:
            Diese robuste Leselogik ist wichtig, weil Vorverarbeitungsstände
            einzelner Dateien variieren können, das Training aber dennoch mit
            einer einheitlichen Struktur arbeiten muss.
        """
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
        """
        Lädt und bereinigt die bevorzugten Enriched-2025-Trainingsdaten.

        Mehrere Linien-Dateien werden zusammengeführt, auf das Jahr 2025
        eingeschränkt, typisiert und um normalisierte Haltestellenschlüssel
        ergänzt. Zudem werden Event- und Konzertstunden harmonisiert und
        Dubletten entfernt.

        Parameter:
            lines (list[int]): Gewünschte Liniennummern.

        Rückgabewerte:
            pd.DataFrame: Bereinigter Rohdatensatz für das Modelltraining.

        Fehler/Sonderfälle:
            Fehlerhafte Einzeldateien werden übersprungen. Fehlen verwertbare
            Daten vollständig, wird ein leerer DataFrame zurückgegeben. Fehlende
            Datum- oder Stundeninformationen werden – soweit möglich – aus
            Alternativspalten rekonstruiert.

        Projektkontext:
            Diese Methode liefert die bevorzugte Lernbasis für das Ensemble aus
            Random Forest und Graph-Light-Modell und setzt damit die fachliche
            Schwerpunktsetzung auf das bereinigte 2025-Archiv um.
        """
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