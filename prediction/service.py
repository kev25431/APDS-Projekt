from __future__ import annotations


import json
import math
import pickle
from datetime import date, timedelta
from typing import Any


import numpy as np
import pandas as pd


try:
    from sklearn.ensemble import RandomForestRegressor
except Exception:  # pragma: no cover - optional dependency fallback
    RandomForestRegressor = None


from .config import (
    CONTEXT_COLUMNS,
    INCREMENTAL_RANDOM_FOREST_TREES,
    INITIAL_RANDOM_FOREST_TREES,
    MODEL_DIR,
    OUTPUT_DIR,
    TARGET_UTILIZATION,
    VEHICLE_TYPE_CAPACITY_FALLBACK,
)
from .data import TrainingDataLoader
from .models import GraphMessagePassingRegressor, GroupFallbackRegressor
from .types import PredictionResult



class DemandPredictionService:
    """
    Orchestriert Training, Laden und Nutzung des Prognose-Ensembles.

    Die Klasse bündelt die gesamte Prognoselogik des Projekts. Sie verbindet
    die Datenaufbereitung mit zwei Modellkomponenten – einem Random-Forest-
    Regressor und einem graphbasierten Regressor – und kombiniert beide zu
    einem gewichteten Ensemble. Zusätzlich erzeugt sie aus Vorhersagen
    operative Ableitungen wie einen adaptiven Fahrplan.

    Parameter:
        repo (Any): Datenrepository des Projekts mit Zugriff auf Rohdaten und
            Kontextinformationen.

    Rückgabewerte:
        Keine direkten Rückgabewerte auf Klassenebene.

    Fehler/Sonderfälle:
        Ist scikit-learn nicht verfügbar, wird statt des Random Forest ein
        gruppenbasierter Fallback-Regressor verwendet.

    Projektkontext:
        Die Klasse ist der zentrale Service für die Prediction-Funktionalität
        und bildet die Brücke zwischen historischen Verkehrsdaten, ML-Modellen
        und operativer Entscheidungsunterstützung.
    """

    """Prediction orchestrator for RF + graph-light ensemble.


    Training is intentionally based on the cleaned 2025 archive when available.
    Predictions can then be requested for 2026 dates through the GUI.
    """


    def __init__(self, repo: Any) -> None:
        """
        Initialisiert den Prognoseservice mit Standardzuständen und Modellobjekten.

        Die Methode legt interne Datenstrukturen für Trainingsdaten, Merkmale,
        Stationszuordnungen, Modellgewichte und Metriken an und instanziiert
        bereits das graphbasierte Regressionsmodell.

        Parameter:
            repo (Any): Repository mit Zugriff auf Linien-, Fahrgast- und
                Kontextdaten.

        Rückgabewerte:
            None: Die Methode initialisiert den Objektzustand.

        Fehler/Sonderfälle:
            Keine explizite Fehlerbehandlung; das Repository muss die vom Service
            erwarteten Schnittstellen bereitstellen.

        Projektkontext:
            Die Initialisierung definiert den Ausgangszustand des gesamten
            Prediction-Subsystems und bereitet sowohl Training als auch spätere
            Inferenz vor.
        """
        self.repo = repo
        self.data_loader = TrainingDataLoader(repo)
        self.trained = False
        self.feature_columns: list[str] = []
        self.station_id_map: dict[str, int] = {}
        self.station_catalog = pd.DataFrame()
        self.raw_train = pd.DataFrame()
        self.train_frame = pd.DataFrame()
        self.trained_lines: set[int] = set()
        self.vehicle_capacity_by_type = VEHICLE_TYPE_CAPACITY_FALLBACK.copy()
        self.rf_model: Any = None
        self.graph_model = GraphMessagePassingRegressor()
        self.weights = {"random_forest": 0.5, "gnn": 0.5}
        self.metrics = {"rf_mae": math.nan, "gnn_mae": math.nan, "ensemble_mae": math.nan}
        self.training_source = ""
        self.training_date_span: tuple[date, date] | None = None
        self.last_training_mode = "initial"


    def fit(
        self,
        lines: list[int],
        start: date,
        end: date,
        incremental: bool = True,
        additional_trees: int = INCREMENTAL_RANDOM_FOREST_TREES,
    ) -> PredictionResult:
        """
        Trainiert oder erweitert das Ensemble für die angegebenen Linien.

        Die Methode lädt zunächst Trainingsdaten, erzeugt daraus einen
        Merkmalsdatenrahmen, baut Stations- und Feature-Mappings auf und trainiert
        anschließend Random Forest und Graph-Modell. Danach werden beide Modelle
        auf einem Validierungsfenster bewertet und über MAE-basierte Gewichte zu
        einem Ensemble kombiniert.

        Parameter:
            lines (list[int]): Zu trainierende Linien.
            start (date): Beginn des Trainingszeitraums.
            end (date): Ende des Trainingszeitraums.
            incremental (bool): Steuert, ob ein vorhandenes Modell für diese
                Linien inkrementell erweitert werden soll.
            additional_trees (int): Anzahl zusätzlich hinzuzufügender Bäume beim
                inkrementellen Random-Forest-Training.

        Rückgabewerte:
            PredictionResult: Ergebnisobjekt mit Metriken, Gewichten und
            Trainingsstatusmeldung.

        Fehler/Sonderfälle:
            Bei zu wenig Daten oder weniger als zwei eindeutigen Trainingstagen
            wird kein nutzbares Modell aufgebaut. Fehlen frühere Feature- oder
            Stationsmappings, werden sie neu erzeugt.

        Projektkontext:
            Die Methode ist der zentrale Trainingseinstieg des Systems und
            erzeugt die Modellbasis für spätere Prognosen und adaptive
            Fahrplanempfehlungen.
        """
        existing_loaded = self.load_for_lines(lines) if incremental else False
        previous_station_id_map = dict(self.station_id_map) if existing_loaded else {}
        previous_feature_columns = list(self.feature_columns) if existing_loaded else []


        training_data = self.data_loader.load(lines, start, end)
        raw = training_data.raw
        self.training_source = training_data.source
        self.training_date_span = training_data.date_span


        frame = self._build_feature_frame(raw)
        if frame.empty or len(frame["date"].unique()) < 2:
            self.trained = False
            self.trained_lines = set()
            return PredictionResult(pd.DataFrame(), self.metrics, self.weights, "Zu wenig Daten für Training.")


        self.raw_train = raw
        self.train_frame = frame
        self.vehicle_capacity_by_type = self._estimate_vehicle_capacities(raw)
        self.station_catalog = self._build_station_catalog(frame)
        self.station_id_map = self._station_id_map_for_frame(frame, previous_station_id_map)
        frame["station_id"] = frame["station_key"].map(self.station_id_map).fillna(-1).astype(int)
        current_feature_columns = self._feature_columns(frame)
        if existing_loaded and previous_feature_columns and all(column in frame.columns for column in previous_feature_columns):
            self.feature_columns = previous_feature_columns
        else:
            self.feature_columns = current_feature_columns
        for column in self.feature_columns:
            if column not in frame:
                frame[column] = 0


        train_frame, val_frame = self._time_split(frame)
        self.last_training_mode = "incremental" if existing_loaded else "initial"
        self.rf_model = self._fit_random_forest(
            train_frame,
            incremental=existing_loaded,
            additional_trees=additional_trees,
        )
        self.graph_model = GraphMessagePassingRegressor()
        self.graph_model.fit(train_frame, raw[raw["date"].isin(set(train_frame["date"]))])


        rf_pred = self._predict_rf(val_frame)
        gnn_pred = self.graph_model.predict(val_frame)
        y_val = val_frame[["boardings", "exiting"]].to_numpy(dtype=float)
        rf_mae = self._mae(y_val, rf_pred)
        gnn_mae = self._mae(y_val, gnn_pred)
        self.weights = self._weights_from_mae(rf_mae, gnn_mae)
        ensemble_pred = self.weights["random_forest"] * rf_pred + self.weights["gnn"] * gnn_pred
        ensemble_mae = self._mae(y_val, ensemble_pred)
        self.metrics = {"rf_mae": rf_mae, "gnn_mae": gnn_mae, "ensemble_mae": ensemble_mae}
        self.trained = True
        self.trained_lines = {int(line) for line in lines}
        self._save_model()
        self._write_metrics()
        return PredictionResult(pd.DataFrame(), self.metrics, self.weights, self._training_message())


    def load_for_lines(self, lines: list[int]) -> bool:
        """
        Lädt ein bereits persistiertes Modell für eine bestimmte Linienmenge.

        Die Methode prüft zunächst das Vorhandensein und die Struktur der
        gespeicherten Modelldatei. Danach werden Merkmalslisten,
        Stationsabbildungen, Trainingsdaten, Modellobjekte sowie Metadaten in
        den aktuellen Servicezustand übernommen.

        Parameter:
            lines (list[int]): Linienmenge, für die ein gespeichertes Modell
                gesucht wird.

        Rückgabewerte:
            bool: True, wenn ein kompatibles Modell erfolgreich geladen wurde,
            sonst False.

        Fehler/Sonderfälle:
            Nicht vorhandene Dateien, beschädigte Pickles, Versionskonflikte oder
            unpassende Linienmengen führen zu False. Ein Modell gilt nur dann als
            geladen, wenn ein RF-Modell und Feature-Spalten verfügbar sind.

        Projektkontext:
            Die Methode ermöglicht persistente Wiederverwendung zuvor trainierter
            Modelle und reduziert Trainingsaufwand beim erneuten Start der Anwendung.
        """
        path = self._model_path(lines)
        if not path.exists():
            return False
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
        except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError):
            return False


        if payload.get("version") != 1:
            return False
        trained_lines = {int(line) for line in payload.get("trained_lines", [])}
        if trained_lines != {int(line) for line in lines}:
            return False


        self.feature_columns = list(payload.get("feature_columns", []))
        self.station_id_map = dict(payload.get("station_id_map", {}))
        self.station_catalog = payload.get("station_catalog", pd.DataFrame())
        self.raw_train = payload.get("raw_train", pd.DataFrame())
        self.train_frame = payload.get("train_frame", pd.DataFrame())
        self.trained_lines = trained_lines
        self.vehicle_capacity_by_type = dict(payload.get("vehicle_capacity_by_type", VEHICLE_TYPE_CAPACITY_FALLBACK.copy()))
        self.rf_model = payload.get("rf_model")
        self.graph_model = payload.get("graph_model", GraphMessagePassingRegressor())
        self.weights = dict(payload.get("weights", {"random_forest": 0.5, "gnn": 0.5}))
        self.metrics = dict(payload.get("metrics", {"rf_mae": math.nan, "gnn_mae": math.nan, "ensemble_mae": math.nan}))
        self.training_source = str(payload.get("training_source", "Gespeichertes Modell"))
        self.last_training_mode = str(payload.get("last_training_mode", "loaded"))
        span = payload.get("training_date_span")
        if isinstance(span, list | tuple) and len(span) == 2:
            self.training_date_span = (pd.Timestamp(span[0]).date(), pd.Timestamp(span[1]).date())
        else:
            self.training_date_span = None
        self.trained = self.rf_model is not None and bool(self.feature_columns)
        return self.trained


    def predict(self, line: int, selected_day: date, hour: int) -> PredictionResult:
        """
        Berechnet eine punktuelle Prognose für eine Linie zu einem bestimmten Tag und einer Stunde.

        Die Methode erzeugt zunächst einen Prognose-Datenrahmen für alle
        relevanten Haltestellen der Linie, wendet darauf das Ensemble an und
        sortiert die Ergebnisse anschließend nach prognostizierter Stärke.

        Parameter:
            line (int): Ziel-Linie der Vorhersage.
            selected_day (date): Prognosetag.
            hour (int): Prognosestunde.

        Rückgabewerte:
            PredictionResult: Ergebnisobjekt mit Vorhersagedaten, Metriken,
            Gewichten und Statusmeldung.

        Fehler/Sonderfälle:
            Ist kein trainiertes Modell vorhanden oder existieren für die Linie
            keine Haltestellen im Trainingssatz, wird ein leeres Ergebnisobjekt
            mit Statusmeldung zurückgegeben.

        Projektkontext:
            Die Methode stellt die einfache, stichtagsbezogene Vorhersagefunktion
            des Services bereit und dient als Grundlage für GUI-gestützte
            Einzelprognosen.
        """
        if not self.trained:
            return PredictionResult(pd.DataFrame(), self.metrics, self.weights, "Modell ist noch nicht trainiert.")


        frame = self._build_prediction_frame(line, selected_day, hour)
        if frame.empty:
            return PredictionResult(pd.DataFrame(), self.metrics, self.weights, "Keine Haltestellen für diese Linie im Trainingssatz.")


        result = self._predict_frame(frame)
        result = result.sort_values(["pred_total", "pred_boardings"], ascending=False).reset_index(drop=True)
        self._write_predictions(result)
        return PredictionResult(result, self.metrics, self.weights, "Prediction berechnet.")


    def predict_short_term(
        self,
        line: int,
        selected_day: date,
        start_hour: int,
        horizon_hours: int = 4,
        station_keys: set[str] | None = None,
        cost_per_bus_hour: float | None = None,
    ) -> PredictionResult:
        """
        Berechnet eine kurzfristige Mehrstundenprognose und optional einen adaptiven Fahrplan.

        Die Methode erzeugt für einen begrenzten Prognosehorizont mehrere
        stundenbezogene Eingaberahmen, führt darauf das Ensemble aus und ergänzt
        das Ergebnis je nach Modus um Abschnittskapazitäten oder adaptive
        Fahrplanempfehlungen.

        Parameter:
            line (int): Ziel-Linie der Vorhersage.
            selected_day (date): Starttag des Prognosefensters.
            start_hour (int): Startstunde des Prognosefensters.
            horizon_hours (int): Länge des Prognosehorizonts in Stunden.
            station_keys (set[str] | None): Optionale Einschränkung auf bestimmte
                Haltestellenabschnitte.
            cost_per_bus_hour (float | None): Optionaler Busstundenkostensatz für
                die adaptive Fahrplanlogik.

        Rückgabewerte:
            PredictionResult: Ergebnisobjekt mit Vorhersagen, Metriken,
            Ensemblegewichten, Statusmeldung und optionalem Fahrplan.

        Fehler/Sonderfälle:
            Der Horizont wird auf mindestens 1 und höchstens 12 Stunden begrenzt.
            Sind keine passenden Haltestellen verfügbar, wird ein leeres Ergebnis
            zurückgegeben. Bei gesetzten `station_keys` wird kein adaptiver
            Fahrplan, sondern eine abschnittsbezogene Kapazitätsergänzung erzeugt.

        Projektkontext:
            Diese Methode bildet den operativ wichtigsten Prognosepfad des
            Projekts, da sie Mehrstundenprognosen direkt in nutzbare
            Dispositionshinweise überführt.
        """
        if not self.trained:
            return PredictionResult(pd.DataFrame(), self.metrics, self.weights, "Modell ist noch nicht trainiert.")


        frames: list[pd.DataFrame] = []
        horizon_hours = max(1, min(int(horizon_hours), 12))
        for offset in range(horizon_hours):
            absolute_hour = int(start_hour) + offset
            day = selected_day + timedelta(days=absolute_hour // 24)
            hour = absolute_hour % 24
            frame = self._build_prediction_frame(line, day, hour)
            if station_keys:
                frame = frame[frame["station_key"].isin(station_keys)].copy()
            if not frame.empty:
                frame["horizon_step"] = offset + 1
                frames.append(frame)
        if not frames:
            return PredictionResult(pd.DataFrame(), self.metrics, self.weights, "Keine Haltestellen für diese Linie im Trainingssatz.")


        frame = pd.concat(frames, ignore_index=True)
        result = self._predict_frame(frame, include_horizon=True)
        result = result.sort_values(
            ["date", "hour", "pred_total", "pred_boardings"],
            ascending=[True, True, False, False],
        ).reset_index(drop=True)
        if station_keys:
            self._add_section_capacity(result, line)
        schedule = pd.DataFrame() if station_keys else self._build_adaptive_schedule(result, line, cost_per_bus_hour=cost_per_bus_hour)
        self._write_predictions(result)
        self._write_schedule(schedule)
        return PredictionResult(result, self.metrics, self.weights, "Kurzfrist-Prognose und adaptiver Fahrplan berechnet.", schedule)


    def _training_message(self) -> str:
        """
        Erzeugt eine fachlich lesbare Statusmeldung zum abgeschlossenen Training.

        Die Meldung enthält Trainingsquelle, optional den verwendeten Datumsbereich,
        die Information, ob das Modell neu oder inkrementell trainiert wurde, sowie
        die Anzahl der Random-Forest-Bäume.

        Parameter:
            Keine.

        Rückgabewerte:
            str: Formulierte Trainingsmeldung für GUI oder Ergebnisobjekte.

        Fehler/Sonderfälle:
            Liegt kein Trainingszeitraum vor, wird eine verkürzte Meldung ohne
            Datumsintervall erzeugt.

        Projektkontext:
            Die Methode macht Trainingsherkunft und Modellzustand transparent und
            unterstützt die Nachvollziehbarkeit für Entwickler und Stakeholder.
        """
        mode = "inkrementell erweitert" if self.last_training_mode == "incremental" else "neu trainiert"
        tree_count = int(getattr(self.rf_model, "n_estimators", 0) or 0)
        tree_text = f" | RF-Bäume: {tree_count}" if tree_count else ""
        if self.training_date_span is None:
            return f"Training abgeschlossen: {self.training_source} für Prediction 2026 ({mode}{tree_text})."
        start, end = self.training_date_span
        return (
            f"Training abgeschlossen: {self.training_source} "
            f"({start.isoformat()} bis {end.isoformat()}) für Prediction 2026 ({mode}{tree_text})."
        )


    def _model_path(self, lines: list[int]):
        """
        Ermittelt den Dateipfad des persistenten Modells für eine Linienmenge.

        Die Methode bildet aus den sortierten Linien einen stabilen Schlüssel und
        verwendet diesen zur Generierung des Pickle-Dateinamens.

        Parameter:
            lines (list[int]): Linienmenge des Modells.

        Rückgabewerte:
            Path: Zielpfad der Modelldatei.

        Fehler/Sonderfälle:
            Keine explizite Fehlerbehandlung.

        Projektkontext:
            Die Methode sorgt für eine reproduzierbare Ablagestruktur der
            linienbezogenen Modellpersistenz.
        """
        line_key = "_".join(str(int(line)) for line in sorted(lines))
        return MODEL_DIR / f"wvv_prediction_lines_{line_key}.pkl"


    def _save_model(self) -> None:
        """
        Persistiert das aktuell trainierte Modell mitsamt Metadaten auf Dateisystemebene.

        Die Methode schreibt Modellobjekte, Mappings, Trainingsdaten,
        Feature-Definitionen, Metriken und Metadaten in ein Pickle-Payload und
        ersetzt die Zieldatei atomar über eine temporäre Datei.

        Parameter:
            Keine.

        Rückgabewerte:
            None: Die Methode speichert Modellartefakte auf dem Dateisystem.

        Fehler/Sonderfälle:
            Ist kein trainiertes Modell vorhanden, wird nichts gespeichert.
            Datei- oder Pickle-Fehler werden tolerant behandelt; temporäre Dateien
            werden nach Möglichkeit wieder entfernt.

        Projektkontext:
            Die Methode ermöglicht Wiederverwendbarkeit und inkrementelle
            Erweiterung von Modellen über Anwendungssitzungen hinweg.
        """
        if not self.trained:
            return
        path = self._model_path(sorted(self.trained_lines))
        payload = {
            "version": 1,
            "trained_lines": sorted(self.trained_lines),
            "feature_columns": self.feature_columns,
            "station_id_map": self.station_id_map,
            "station_catalog": self.station_catalog,
            "raw_train": self.raw_train,
            "train_frame": self.train_frame,
            "vehicle_capacity_by_type": self.vehicle_capacity_by_type,
            "rf_model": self.rf_model,
            "graph_model": self.graph_model,
            "weights": self.weights,
            "metrics": self.metrics,
            "training_source": self.training_source,
            "last_training_mode": self.last_training_mode,
            "rf_n_estimators": int(getattr(self.rf_model, "n_estimators", 0) or 0),
            "training_date_span": (
                None
                if self.training_date_span is None
                else [self.training_date_span[0].isoformat(), self.training_date_span[1].isoformat()]
            ),
        }
        tmp_path = path.with_suffix(".tmp")
        try:
            with tmp_path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(path)
        except (OSError, pickle.PickleError):
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


    def _predict_frame(self, frame: pd.DataFrame, include_horizon: bool = False) -> pd.DataFrame:
        """
        Wendet das Ensemble auf einen vorbereiteten Eingabedatenrahmen an.

        Die Methode berechnet Vorhersagen beider Teilmodelle, kombiniert diese
        über die aktuellen Ensemblegewichte und erzeugt daraus einen
        Ergebnisdatenrahmen mit Einzel- und Gesamtprognosen. `numpy.clip`
        begrenzt Werte dabei auf ein zulässiges Intervall; hier werden negative
        Vorhersagen auf 0 abgeschnitten. [web:165]

        Parameter:
            frame (pd.DataFrame): Vorbereiteter Eingabedatenrahmen für die Inferenz.
            include_horizon (bool): Steuert, ob die Spalte `horizon_step` in das
                Ergebnis übernommen wird.

        Rückgabewerte:
            pd.DataFrame: Ergebnisdatenrahmen mit RF-, GNN- und Ensembleprognosen.

        Fehler/Sonderfälle:
            Enthält der Eingabedatenrahmen die Horizontspalte nicht, wird sie auch
            bei gesetztem Flag nicht übernommen.

        Projektkontext:
            Die Methode bündelt die eigentliche Inferenzlogik des Ensembles und
            erzeugt die standardisierte Ergebnisstruktur für weitere Auswertung.
        """
        rf_pred = self._predict_rf(frame)
        gnn_pred = self.graph_model.predict(frame)
        ensemble = self.weights["random_forest"] * rf_pred + self.weights["gnn"] * gnn_pred
        columns = ["line", "station", "station_key", "station_number", "hour", "date"]
        if include_horizon and "horizon_step" in frame.columns:
            columns.append("horizon_step")
        result = frame[columns].copy()
        result["rf_boardings"] = np.clip(rf_pred[:, 0], 0, None)
        result["rf_exiting"] = np.clip(rf_pred[:, 1], 0, None)
        result["gnn_boardings"] = np.clip(gnn_pred[:, 0], 0, None)
        result["gnn_exiting"] = np.clip(gnn_pred[:, 1], 0, None)
        result["pred_boardings"] = np.clip(ensemble[:, 0], 0, None)
        result["pred_exiting"] = np.clip(ensemble[:, 1], 0, None)
        result["pred_total"] = result["pred_boardings"] + result["pred_exiting"]
        return result


    def _build_feature_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Erzeugt aus Rohdaten einen modellgeeigneten Merkmalsdatenrahmen.

        Die Methode aggregiert Rohdaten auf Ebene von Datum, Linie, Haltestelle
        und Stunde und berechnet daraus Zielgrößen sowie erklärende Merkmale wie
        Fahrtenzahl, Peak-Last, durchschnittliche Auslastung und Kalendermerkmale.

        Parameter:
            raw (pd.DataFrame): Bereinigte Rohdaten des Trainingszeitraums.

        Rückgabewerte:
            pd.DataFrame: Aggregierter und mit Kontext angereicherter
            Merkmalsdatenrahmen.

        Fehler/Sonderfälle:
            Bei leerem Input wird ein leerer Datenrahmen zurückgegeben. Fehlende
            Kontextmerkmale werden später über `_merge_context` ergänzt.

        Projektkontext:
            Die Methode bildet den zentralen Transformationsschritt vom operativen
            Ereignisdatensatz hin zu einem lernbaren ML-Feature-Set.
        """
        if raw.empty:
            return pd.DataFrame()
        aggregations = {
            "boardings": ("passenger_boarding", "sum"),
            "exiting": ("passenger_exiting", "sum"),
            "journeys": ("journey", "nunique"),
            "peak_load": ("occupancy_departure", "max"),
            "avg_utilization": ("vehicle_utilization", "mean"),
        }
        for column in CONTEXT_COLUMNS:
            if column in raw.columns:
                aggregations[column] = (column, "max")
        frame = (
            raw.groupby(["date", "line", "station_number", "station", "station_key", "hour"], as_index=False)
            .agg(**aggregations)
            .sort_values(["date", "line", "station_number", "hour"])
        )
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        frame["weekday"] = pd.to_datetime(frame["date"].astype(str)).dt.weekday
        frame["month"] = pd.to_datetime(frame["date"].astype(str)).dt.month
        frame["is_weekend"] = frame["weekday"].isin([5, 6]).astype(int)
        frame = self._merge_context(frame)
        return frame.fillna(0)


    def _merge_context(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Ergänzt fehlende Kontextmerkmale in einem Merkmalsdatenrahmen.

        Die Methode fügt aus einem täglichen Kontextdatensatz diejenigen Spalten
        hinzu, die im Eingabedatenrahmen noch fehlen, und setzt nicht verfügbare
        Kontextmerkmale standardmäßig auf 0.

        Parameter:
            frame (pd.DataFrame): Eingabedatenrahmen mit bereits vorhandenen
                Merkmals- und Zielspalten.

        Rückgabewerte:
            pd.DataFrame: Datenrahmen mit vollständig ergänzteten Kontextmerkmalen.

        Fehler/Sonderfälle:
            Ist kein Kontextdatensatz im Repository verfügbar oder fehlen dort die
            benötigten Spalten, werden die Merkmale mit 0 ergänzt.

        Projektkontext:
            Die Methode stellt sicher, dass Training und Inferenz auf einer
            stabilen, konsistenten Merkmalsmenge basieren.
        """
        merged = frame.copy()
        missing_columns = [column for column in CONTEXT_COLUMNS if column not in merged.columns]
        if missing_columns and not self.repo.context_daily.empty:
            context_columns = ["date"] + [column for column in missing_columns if column in self.repo.context_daily.columns]
            if len(context_columns) > 1:
                merged = merged.merge(self.repo.context_daily[context_columns], on="date", how="left")
        for column in CONTEXT_COLUMNS:
            if column not in merged:
                merged[column] = 0
        return merged.fillna({column: 0 for column in CONTEXT_COLUMNS})


    def _build_station_catalog(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Erzeugt einen Katalog aller im Trainingssatz vorkommenden Haltestellen.

        Der Katalog fasst pro Linie und Stationsschlüssel die zugehörige
        Haltestellenbezeichnung und eine repräsentative Stationsnummer zusammen.

        Parameter:
            frame (pd.DataFrame): Trainingsdatenrahmen mit Stationsinformationen.

        Rückgabewerte:
            pd.DataFrame: Sortierter Haltestellenkatalog je Linie.

        Fehler/Sonderfälle:
            Keine explizite Fehlerbehandlung.

        Projektkontext:
            Der Haltestellenkatalog dient der späteren Konstruktion von
            Prognoseeingaben für bestimmte Linien und Tage.
        """
        return (
            frame.groupby(["line", "station_key", "station"], as_index=False)
            .agg(station_number=("station_number", "median"))
            .sort_values(["line", "station_number", "station"])
        )


    def _station_id_map_for_frame(self, frame: pd.DataFrame, previous_map: dict[str, int] | None = None) -> dict[str, int]:
        """
        Baut eine stabile numerische ID-Abbildung für Stationsschlüssel auf.

        Die Methode übernimmt vorhandene IDs, sofern verfügbar, und ergänzt
        neue Stationen fortlaufend. Dadurch bleiben numerische Stations-IDs über
        inkrementelle Trainingsläufe hinweg möglichst konsistent.

        Parameter:
            frame (pd.DataFrame): Datenrahmen mit Stationsschlüsseln.
            previous_map (dict[str, int] | None): Optional vorhandene frühere
                Abbildung.

        Rückgabewerte:
            dict[str, int]: Vollständige Abbildung von Stationsschlüssel auf ID.

        Fehler/Sonderfälle:
            Fehlt eine frühere Abbildung, wird bei 0 bzw. der ersten freien ID
            begonnen.

        Projektkontext:
            Die Methode erzeugt ein modellgeeignetes numerisches Stationsmerkmal
            und stabilisiert die Repräsentation bei inkrementellem Training.
        """
        station_map = dict(previous_map or {})
        next_id = max(station_map.values(), default=-1) + 1
        for key in sorted(str(value) for value in frame["station_key"].dropna().unique()):
            if key not in station_map:
                station_map[key] = next_id
                next_id += 1
        return station_map


    def _feature_columns(self, frame: pd.DataFrame) -> list[str]:
        """
        Bestimmt die tatsächlich verwendeten Feature-Spalten des Modells.

        Die Methode startet mit einer fachlich vordefinierten Basismenge und
        ergänzt nur diejenigen Kontextspalten, die im aktuellen Datenrahmen
        tatsächlich vorhanden sind.

        Parameter:
            frame (pd.DataFrame): Eingabedatenrahmen mit potenziellen Features.

        Rückgabewerte:
            list[str]: Liste der im Modell verwendeten Feature-Spalten.

        Fehler/Sonderfälle:
            Fehlende Basisspalten werden nicht künstlich ergänzt, sondern nur
            berücksichtigt, wenn sie im Frame vorhanden sind.

        Projektkontext:
            Die Methode definiert den Merkmalsraum, auf dessen Basis das
            Ensemble trainiert und später angewendet wird.
        """
        base = [
            "line",
            "station_number",
            "station_id",
            "hour",
            "weekday",
            "month",
            "is_weekend",
            "journeys",
            "peak_load",
            "avg_utilization",
        ]
        return [column for column in base + CONTEXT_COLUMNS if column in frame]


    def _time_split(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Teilt den Merkmalsdatenrahmen in zeitlich geordnetes Trainings- und Validierungsfenster.

        Die Methode verwendet primär einen datumsbasierten 80/20-Split, damit
        die Validierung zeitlich nachgelagert zum Training erfolgt und so eine
        realistischere Prognosesituation nachbildet.

        Parameter:
            frame (pd.DataFrame): Vollständiger Trainingsdatenrahmen.

        Rückgabewerte:
            tuple[pd.DataFrame, pd.DataFrame]: Trainings- und Validierungsframe.

        Fehler/Sonderfälle:
            Entsteht durch den Datumssplit ein leerer Teil, wird ersatzweise ein
            positionsbasierter Fallback-Split verwendet.

        Projektkontext:
            Die Methode schafft die Grundlage für eine einfache zeitliche
            Modellbewertung ohne Informationsleckage zwischen Training und Test.
        """
        unique_dates = sorted(frame["date"].unique())
        split_index = max(1, int(len(unique_dates) * 0.8))
        if split_index >= len(unique_dates):
            split_index = len(unique_dates) - 1
        train_dates = set(unique_dates[:split_index])
        train_frame = frame[frame["date"].isin(train_dates)].copy()
        val_frame = frame[~frame["date"].isin(train_dates)].copy()
        if train_frame.empty or val_frame.empty:
            midpoint = max(1, len(frame) - max(10, len(frame) // 5))
            train_frame = frame.iloc[:midpoint].copy()
            val_frame = frame.iloc[midpoint:].copy()
        return train_frame, val_frame


    def _fit_random_forest(
        self,
        train_frame: pd.DataFrame,
        incremental: bool = False,
        additional_trees: int = INCREMENTAL_RANDOM_FOREST_TREES,
    ) -> Any:
        """
        Trainiert den Random-Forest-Teil des Ensembles oder einen Fallback-Regressor.

        Die Methode verwendet bei verfügbarer scikit-learn-Installation einen
        `RandomForestRegressor`. Ist inkrementelles Training aktiv und bereits
        ein kompatibles Modell vorhanden, wird `warm_start=True` genutzt, um
        zusätzliche Bäume zum bestehenden Wald hinzuzufügen statt komplett neu
        zu trainieren. [web:161]

        Parameter:
            train_frame (pd.DataFrame): Trainingsdatenrahmen.
            incremental (bool): Steuert, ob ein vorhandenes RF-Modell erweitert
                werden soll.
            additional_trees (int): Zahl zusätzlich hinzuzufügender Bäume.

        Rückgabewerte:
            Any: Trainiertes Random-Forest-Modell oder Fallback-Regressor.

        Fehler/Sonderfälle:
            Ist scikit-learn nicht verfügbar, wird ein Gruppen-Fallback-Modell
            verwendet. Bei inkrementellem Training wird nur dann erweitert, wenn
            das bestehende Modell tatsächlich ein `RandomForestRegressor` ist.

        Projektkontext:
            Die Methode implementiert die tabellarische Kernmodellierung des
            Ensembles und unterstützt sowohl Initial- als auch Folge-Trainingsläufe.
        """
        if RandomForestRegressor is None:
            fallback = GroupFallbackRegressor()
            fallback.fit(train_frame)
            return fallback
        if incremental and isinstance(self.rf_model, RandomForestRegressor):
            model = self.rf_model
            current_trees = int(getattr(model, "n_estimators", INITIAL_RANDOM_FOREST_TREES))
            model.set_params(
                warm_start=True,
                n_estimators=current_trees + max(1, int(additional_trees)),
            )
        else:
            model = RandomForestRegressor(
                n_estimators=INITIAL_RANDOM_FOREST_TREES,
                max_depth=16,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=1,
                warm_start=True,
            )
        model.fit(
            train_frame[self.feature_columns].to_numpy(dtype=float),
            train_frame[["boardings", "exiting"]].to_numpy(dtype=float),
        )
        return model


    def _predict_rf(self, frame: pd.DataFrame) -> np.ndarray:
        """
        Berechnet Vorhersagen des Random-Forest-Zweigs oder des Fallback-Modells.

        Die Methode transformiert den Eingabedatenrahmen in ein numerisches
        Feature-Array und gibt für jede Zeile prognostizierte Ein- und
        Ausstiegswerte zurück.

        Parameter:
            frame (pd.DataFrame): Eingabedatenrahmen für die Vorhersage.

        Rückgabewerte:
            np.ndarray: Zweidimensionales Array mit Vorhersagen für Boardings
            und Exiting.

        Fehler/Sonderfälle:
            Bei leerem Frame wird ein leeres Array mit passender Form
            zurückgegeben. Für den Fallback-Regressor wird dessen eigene
            DataFrame-basierte Schnittstelle verwendet.

        Projektkontext:
            Die Methode kapselt die tabellarische Inferenzlogik des RF-Zweigs
            innerhalb des Ensemble-Services.
        """
        if frame.empty:
            return np.empty((0, 2), dtype=float)
        if isinstance(self.rf_model, GroupFallbackRegressor):
            return self.rf_model.predict(frame)
        return np.asarray(self.rf_model.predict(frame[self.feature_columns].to_numpy(dtype=float)), dtype=float)


    def _build_prediction_frame(self, line: int, selected_day: date, hour: int) -> pd.DataFrame:
        """
        Baut einen Inferenzdatenrahmen für eine Linie, einen Tag und eine Stunde.

        Die Methode verwendet den im Training erzeugten Haltestellenkatalog, um
        für alle bekannten Haltestellen einer Linie einen synthetischen
        Eingabedatenrahmen für den gewünschten Prognosezeitpunkt zu erzeugen.

        Parameter:
            line (int): Ziel-Linie.
            selected_day (date): Prognosetag.
            hour (int): Prognosestunde.

        Rückgabewerte:
            pd.DataFrame: Modellgeeigneter Eingabedatenrahmen für die Inferenz.

        Fehler/Sonderfälle:
            Ist für die Linie kein Katalog vorhanden, wird ein leerer DataFrame
            zurückgegeben. Fehlende Features werden mit 0 ergänzt.

        Projektkontext:
            Die Methode erzeugt die Brücke zwischen Trainingskatalog und
            zukünftiger Prognoseanfrage, ohne dass reale Beobachtungsdaten für
            den Zieltag vorliegen müssen.
        """
        catalog = self.station_catalog[self.station_catalog["line"] == int(line)].copy()
        if catalog.empty:
            return pd.DataFrame()
        catalog["date"] = selected_day
        catalog["hour"] = int(hour)
        catalog["weekday"] = pd.Timestamp(selected_day).weekday()
        catalog["month"] = pd.Timestamp(selected_day).month
        catalog["is_weekend"] = int(catalog["weekday"].iloc[0] in [5, 6])
        catalog["station_id"] = catalog["station_key"].map(self.station_id_map).fillna(-1).astype(int)
        catalog["journeys"] = 1
        catalog["peak_load"] = 0
        catalog["avg_utilization"] = 0
        catalog = self._merge_context(catalog)
        for column in self.feature_columns:
            if column not in catalog:
                catalog[column] = 0
        return catalog.fillna(0)


    def _mae(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Berechnet den mittleren absoluten Fehler zwischen Ist- und Prognosewerten.

        Vor der Fehlerberechnung werden negative Prognosewerte auf 0 begrenzt,
        damit unrealistische negative Fahrgastzahlen die Metrik nicht verfälschen.

        Parameter:
            actual (np.ndarray): Tatsächliche Zielwerte.
            predicted (np.ndarray): Vorhergesagte Zielwerte.

        Rückgabewerte:
            float: Mittlerer absoluter Fehler.

        Fehler/Sonderfälle:
            Bei leeren Arrays wird `math.nan` zurückgegeben.

        Projektkontext:
            Die Methode liefert die zentrale Gütemetrik zur Bewertung der beiden
            Teilmodelle und zur Bestimmung der Ensemblegewichte.
        """
        if actual.size == 0 or predicted.size == 0:
            return math.nan
        return float(np.mean(np.abs(actual - np.clip(predicted, 0, None))))


    def _weights_from_mae(self, rf_mae: float, gnn_mae: float) -> dict[str, float]:
        """
        Leitet Ensemblegewichte invers aus den Modellfehlern ab.

        Je kleiner der Fehler eines Teilmodells, desto höher sein Gewicht im
        Ensemble. Dazu werden inverse Fehlerwerte gebildet und normiert.

        Parameter:
            rf_mae (float): MAE des Random-Forest-Modells.
            gnn_mae (float): MAE des Graph-Modells.

        Rückgabewerte:
            dict[str, float]: Gewichte für Random Forest und GNN.

        Fehler/Sonderfälle:
            Sind die Eingabefehler nicht endlich, wird eine neutrale 50/50-
            Gewichtung zurückgegeben. Ein kleiner Offset von 0.001 verhindert
            Division durch 0.

        Projektkontext:
            Die Methode realisiert die adaptive Kopplung beider Modellwelten zu
            einem datengetrieben gewichteten Ensemble.
        """
        if not math.isfinite(rf_mae) or not math.isfinite(gnn_mae):
            return {"random_forest": 0.5, "gnn": 0.5}
        rf_score = 1.0 / (rf_mae + 0.001)
        gnn_score = 1.0 / (gnn_mae + 0.001)
        total = rf_score + gnn_score
        return {"random_forest": rf_score / total, "gnn": gnn_score / total}


    def _estimate_vehicle_capacities(self, raw: pd.DataFrame) -> dict[str, float]:
        """
        Schätzt typische Fahrzeugkapazitäten aus historischen Besetzungs- und Auslastungsdaten.

        Die Methode leitet aus beobachteter Belegung und Fahrzeugauslastung eine
        implizite Kapazität pro Fahrzeugtyp ab und verwendet den Median dieser
        Schätzungen als robuste Repräsentation.

        Parameter:
            raw (pd.DataFrame): Historische Rohdaten mit Fahrzeugtyp, Belegung und
                Auslastung.

        Rückgabewerte:
            dict[str, float]: Geschätzte Kapazitäten je Fahrzeugtyp.

        Fehler/Sonderfälle:
            Fehlen notwendige Spalten oder ist der Datensatz leer, werden nur die
            konfigurierten Fallback-Werte verwendet. Schätzungen außerhalb eines
            plausiblen Bereichs von 25 bis 260 Plätzen werden verworfen.

        Projektkontext:
            Die Methode liefert eine fahrzeugtypspezifische Kapazitätsbasis für
            Auslastungs- und Fahrplanempfehlungen im operativen Teil des Systems.
        """
        capacities = VEHICLE_TYPE_CAPACITY_FALLBACK.copy()
        required = {"vehicle_type", "occupancy_departure", "vehicle_utilization"}
        if raw.empty or not required.issubset(raw.columns):
            return capacities
        observed = raw[list(required)].copy()
        observed["vehicle_type"] = observed["vehicle_type"].fillna("Unknown").astype(str)
        occupancy = pd.to_numeric(observed["occupancy_departure"], errors="coerce")
        utilization = pd.to_numeric(observed["vehicle_utilization"], errors="coerce")
        estimated_capacity = occupancy / utilization.where(utilization > 0.05)
        observed["estimated_capacity"] = estimated_capacity.where(estimated_capacity.between(25, 260))
        medians = observed.groupby("vehicle_type")["estimated_capacity"].median().dropna()
        for vehicle_type, value in medians.items():
            capacities[str(vehicle_type)] = float(value)
        return capacities


    def _journey_capacity_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Erzeugt einen kapazitätsorientierten Fahrtenrahmen auf Journey-Ebene.

        Die Methode reduziert Rohdaten auf eindeutige Fahrten je Datum und Stunde
        und ergänzt dafür eine Fahrzeugkapazität auf Basis des Fahrzeugtyps.

        Parameter:
            raw (pd.DataFrame): Historische Rohdaten.

        Rückgabewerte:
            pd.DataFrame: Aggregierter Journey-Rahmen mit Fahrzeugtyp und Kapazität.

        Fehler/Sonderfälle:
            Bei leerem Input wird ein leerer, aber schema-kompatibler DataFrame
            zurückgegeben. Fehlt die Spalte `vehicle_type`, wird „Unknown“
            verwendet.

        Projektkontext:
            Die Methode schafft die Basis zur Ableitung historischer
            Kapazitätsangebote pro Stunde und damit für adaptive Fahrpläne.
        """
        if raw.empty:
            return pd.DataFrame(columns=["date", "hour", "journey", "vehicle_type", "capacity"])
        frame = raw.copy()
        if "vehicle_type" not in frame.columns:
            frame["vehicle_type"] = "Unknown"
        frame["vehicle_type"] = frame["vehicle_type"].fillna("Unknown").astype(str)
        frame["capacity"] = frame["vehicle_type"].map(self.vehicle_capacity_by_type).fillna(
            VEHICLE_TYPE_CAPACITY_FALLBACK["Unknown"]
        )
        return (
            frame.groupby(["date", "hour", "journey"], as_index=False)
            .agg(vehicle_type=("vehicle_type", "first"), capacity=("capacity", "first"))
        )


    def _hour_capacity_context(self, line: int, hour: int) -> tuple[int, float, float]:
        """
        Bestimmt historische Kapazitätskontexte für eine Linie und Stunde.

        Die Methode berechnet aus historischen Fahrten die typische Anzahl an
        Kursen, die durchschnittliche Fahrzeugkapazität und die daraus
        resultierende angebotene Gesamtkapazität für eine Stunde.

        Parameter:
            line (int): Ziel-Linie.
            hour (int): Zielstunde.

        Rückgabewerte:
            tuple[int, float, float]: Anzahl typischer Kurse, mittlere
            Fahrzeugkapazität und angebotene Gesamtkapazität.

        Fehler/Sonderfälle:
            Liegen keine historischen Fahrten vor, werden robuste Fallback-Werte
            verwendet. Fehlen Daten exakt für die gewünschte Stunde, wird auf
            globale Mittelwerte innerhalb der Linie zurückgegriffen.

        Projektkontext:
            Die Methode liefert Referenzgrößen für Abschnittsauslastung und
            adaptive Fahrplanempfehlungen.
        """
        raw = self.raw_train[self.raw_train["line"] == int(line)].copy()
        journey_capacity = self._journey_capacity_frame(raw)
        if journey_capacity.empty:
            return 1, VEHICLE_TYPE_CAPACITY_FALLBACK["Unknown"], VEHICLE_TYPE_CAPACITY_FALLBACK["Unknown"]
        hour_capacity = journey_capacity[journey_capacity["hour"] == int(hour)]
        if hour_capacity.empty:
            bus_runs = max(1, int(round(journey_capacity.groupby(["date", "hour"])["journey"].nunique().mean())))
            avg_capacity = float(journey_capacity["capacity"].mean())
        else:
            hourly = hour_capacity.groupby(["date", "hour"], as_index=False).agg(
                bus_runs=("journey", "nunique"),
                avg_vehicle_capacity=("capacity", "mean"),
            )
            bus_runs = max(1, int(round(hourly["bus_runs"].mean())))
            avg_capacity = float(hourly["avg_vehicle_capacity"].mean())
        offered_capacity = bus_runs * max(avg_capacity, 1.0)
        return bus_runs, max(avg_capacity, 1.0), offered_capacity


    def _section_utilization_profile(self, line: int, hour: int, station_keys: set[str]) -> float | None:
        """
        Leitet ein historisches Auslastungsprofil für einen Haltestellenabschnitt ab.

        Die Methode sucht in den Trainingsdaten nach vergleichbaren Beobachtungen
        derselben Linie, Stunde und Haltestellenmenge und verwendet das obere
        Quartil der beobachteten Auslastung als robusten Profilwert.

        Parameter:
            line (int): Ziel-Linie.
            hour (int): Zielstunde.
            station_keys (set[str]): Menge relevanter Haltestellen eines Abschnitts.

        Rückgabewerte:
            float | None: Profilbasierte Abschnittsauslastung oder None, falls
            keine belastbare Schätzung möglich ist.

        Fehler/Sonderfälle:
            Fehlen Trainingsdaten oder die Spalte `avg_utilization`, wird None
            geliefert. Gibt es keine stundenexakte Teilmenge, wird auf alle
            passenden Stationsdaten der Linie zurückgegriffen.

        Projektkontext:
            Die Methode ermöglicht eine fachlich plausiblere Abschnittsbewertung
            als reine Nachfrage/Kapazitäts-Quotienten.
        """
        if self.train_frame.empty or not station_keys or "avg_utilization" not in self.train_frame.columns:
            return None
        profile = self.train_frame[
            (self.train_frame["line"] == int(line))
            & (self.train_frame["hour"] == int(hour))
            & (self.train_frame["station_key"].isin(station_keys))
        ].copy()
        if profile.empty:
            profile = self.train_frame[
                (self.train_frame["line"] == int(line))
                & (self.train_frame["station_key"].isin(station_keys))
            ].copy()
        if profile.empty:
            return None
        utilization = pd.to_numeric(profile["avg_utilization"], errors="coerce").dropna()
        utilization = utilization[utilization >= 0]
        if utilization.empty:
            return None
        return float(utilization.quantile(0.75))


    def _add_section_capacity(self, result: pd.DataFrame, line: int) -> None:
        """
        Ergänzt abschnittsbezogene Kapazitäts- und Auslastungskennzahlen in Vorhersageergebnissen.

        Die Methode berechnet je Datum und Stunde historische Kapazitätskontexte,
        aggregierten Abschnittsdemand sowie eine profil- oder fallbackbasierte
        Abschnittsauslastung und schreibt diese Werte in den Ergebnisrahmen.

        Parameter:
            result (pd.DataFrame): Vorhersageergebnisse, die erweitert werden sollen.
            line (int): Ziel-Linie.

        Rückgabewerte:
            None: Die Methode erweitert den DataFrame in-place.

        Fehler/Sonderfälle:
            Bei leerem Ergebnisrahmen erfolgt keine Aktion. Kann kein
            historisches Auslastungsprofil bestimmt werden, wird eine
            Nachfrage/Kapazitäts-Schätzung als Fallback verwendet.

        Projektkontext:
            Die Methode macht kurzfristige Prognosen für Abschnittsanalysen und
            haltestellenbezogene operative Bewertungen nutzbar.
        """
        if result.empty:
            return
        for (_, hour), index in result.groupby(["date", "hour"]).groups.items():
            bus_runs, avg_capacity, offered_capacity = self._hour_capacity_context(line, int(hour))
            section_demand = float(result.loc[index, "pred_total"].sum())
            station_keys = set(result.loc[index, "station_key"].astype(str))
            utilization_profile = self._section_utilization_profile(line, int(hour), station_keys)
            fallback_utilization = section_demand / max(offered_capacity, 1.0)
            result.loc[index, "section_bus_runs"] = bus_runs
            result.loc[index, "avg_vehicle_capacity"] = avg_capacity
            result.loc[index, "section_offered_capacity"] = offered_capacity
            result.loc[index, "section_predicted_utilization"] = (
                utilization_profile if utilization_profile is not None else fallback_utilization
            )
            result.loc[index, "section_predicted_demand"] = section_demand


    def _build_adaptive_schedule(
        self,
        predictions: pd.DataFrame,
        line: int,
        cost_per_bus_hour: float | None = None,
    ) -> pd.DataFrame:
        """
        Leitet aus Prognosen einen adaptiven Fahrplanvorschlag ab.

        Die Methode aggregiert zunächst die prognostizierte Nachfrage je Stunde
        und stellt ihr historische Nachfragemuster, Kurszahlen und Kapazitäten
        gegenüber. Daraus werden empfohlene Kurse, erwartete Auslastung,
        potenziell vermeidbare Leerkurse sowie kostenbezogene Kennzahlen
        abgeleitet.

        Parameter:
            predictions (pd.DataFrame): Vorhersagedaten des Prognosemodells.
            line (int): Ziel-Linie.
            cost_per_bus_hour (float | None): Optionaler Busstundenkostensatz
                zur Sensitivitätssteuerung der Empfehlung.

        Rückgabewerte:
            pd.DataFrame: Adaptiver Fahrplan mit stundenweisen Maßnahmen und
            Begründungskennzahlen.

        Fehler/Sonderfälle:
            Bei leerem Vorhersagerahmen wird ein leerer DataFrame zurückgegeben.
            Fehlen historische Daten, werden robuste globale Fallback-Werte
            verwendet. Empfohlene Kurse werden zwischen 1 und einer aus der
            Historie abgeleiteten Flottengrenze begrenzt.

        Projektkontext:
            Die Methode transformiert reine Nachfrageprognosen in operative
            Handlungsempfehlungen und ist damit zentral für den praktischen
            Nutzen des Systems.
        """
        if predictions.empty:
            return pd.DataFrame()


        predicted_hourly = (
            predictions.groupby(["date", "line", "hour"], as_index=False)
            .agg(
                predicted_demand=("pred_total", "sum"),
                predicted_boardings=("pred_boardings", "sum"),
                predicted_exiting=("pred_exiting", "sum"),
            )
        )
        historical = self.train_frame[self.train_frame["line"] == int(line)].copy()
        raw = self.raw_train[self.raw_train["line"] == int(line)].copy()
        journey_capacity = self._journey_capacity_frame(raw)
        if not journey_capacity.empty:
            historical_runs = (
                journey_capacity.groupby(["date", "hour"], as_index=False)
                .agg(
                    bus_runs=("journey", "nunique"),
                    offered_capacity=("capacity", "sum"),
                    avg_vehicle_capacity=("capacity", "mean"),
                )
            )
        else:
            historical_runs = pd.DataFrame(columns=["date", "hour", "bus_runs", "offered_capacity", "avg_vehicle_capacity"])


        if not historical.empty:
            historical_hourly = (
                historical.groupby(["date", "hour"], as_index=False)
                .agg(historical_demand=("boardings", "sum"))
                .merge(historical_runs, on=["date", "hour"], how="left")
            )
        else:
            historical_hourly = historical_runs.copy()
            historical_hourly["historical_demand"] = 0.0


        if historical_hourly.empty:
            global_demand_p75 = max(float(predicted_hourly["predicted_demand"].median()), 1.0)
            global_runs_avg = 1.0
            global_capacity_avg = VEHICLE_TYPE_CAPACITY_FALLBACK["Unknown"]
            fleet_limit = 1
        else:
            global_demand_p75 = max(float(historical_hourly["historical_demand"].quantile(0.75)), 1.0)
            global_runs_avg = max(float(historical_hourly["bus_runs"].fillna(1).mean()), 1.0)
            global_capacity_avg = max(
                float(historical_hourly["avg_vehicle_capacity"].fillna(VEHICLE_TYPE_CAPACITY_FALLBACK["Unknown"]).mean()),
                1.0,
            )
            fleet_limit = max(1, int(math.ceil(historical_hourly["bus_runs"].fillna(1).quantile(0.95))))


        bus_hourly_cost = 230.0 if cost_per_bus_hour is None else max(0.0, float(cost_per_bus_hour))
        cost_pressure = max(0.65, min(1.65, bus_hourly_cost / 230.0))
        target_utilization = max(0.74, min(0.92, TARGET_UTILIZATION + (cost_pressure - 1.0) * 0.08))
        productive_share = max(0.28, min(0.52, 0.34 + (cost_pressure - 1.0) * 0.10))


        rows: list[dict[str, object]] = []
        for _, row in predicted_hourly.iterrows():
            hour_predictions = predictions[
                (predictions["date"] == row["date"])
                & (predictions["hour"] == row["hour"])
            ].sort_values("pred_total", ascending=False)
            hotspots = ", ".join(str(value)[:18] for value in hour_predictions["station"].head(3).tolist())
            total_demand = float(row["predicted_demand"])
            boardings = float(row.get("predicted_boardings", total_demand))
            exiting = float(row.get("predicted_exiting", 0.0))
            demand = boardings if boardings > 0 else total_demand
            hour_history = historical_hourly[historical_hourly["hour"] == int(row["hour"])]
            if hour_history.empty:
                demand_base = global_demand_p75
                base_runs = global_runs_avg
                avg_capacity = global_capacity_avg
            else:
                demand_base = max(float(hour_history["historical_demand"].quantile(0.75)), 1.0)
                base_runs = max(float(hour_history["bus_runs"].fillna(global_runs_avg).mean()), 1.0)
                avg_capacity = max(float(hour_history["avg_vehicle_capacity"].fillna(global_capacity_avg).mean()), 1.0)


            capacity_required_runs = int(math.ceil(demand / max(avg_capacity * target_utilization, 1.0)))
            profile_required_runs = int(math.ceil(base_runs * max(demand / demand_base, 0.45)))
            required_runs = max(capacity_required_runs, profile_required_runs)
            recommended_runs = min(max(1, required_runs), fleet_limit)
            baseline_runs = int(round(base_runs))


            productive_load = avg_capacity * productive_share
            while recommended_runs > 1 and demand / max(recommended_runs, 1) < productive_load:
                recommended_runs -= 1


            offered_capacity = recommended_runs * avg_capacity
            predicted_utilization = demand / max(offered_capacity, 1.0)
            avoidable_empty_runs = max(0, baseline_runs - recommended_runs)
            if recommended_runs > baseline_runs:
                action = "Verstärken"
            elif recommended_runs < baseline_runs and demand < demand_base * 0.65:
                action = "Ausdünnen"
            else:
                action = "Halten"
            if required_runs > fleet_limit:
                action = "Priorisieren"


            rows.append(
                {
                    "date": row["date"],
                    "line": int(line),
                    "hour": int(row["hour"]),
                    "predicted_demand": total_demand,
                    "predicted_boardings": boardings,
                    "predicted_exiting": exiting,
                    "baseline_runs": baseline_runs,
                    "recommended_runs": recommended_runs,
                    "fleet_limit": fleet_limit,
                    "avg_vehicle_capacity": avg_capacity,
                    "offered_capacity": offered_capacity,
                    "predicted_utilization": predicted_utilization,
                    "avoidable_empty_runs": avoidable_empty_runs,
                    "cost_per_bus_hour": bus_hourly_cost,
                    "baseline_cost": baseline_runs * bus_hourly_cost,
                    "adaptive_cost": recommended_runs * bus_hourly_cost,
                    "cost_weight": cost_pressure,
                    "action": action,
                    "hotspots": hotspots,
                }
            )
        return pd.DataFrame(rows)


    def _write_metrics(self) -> None:
        """
        Schreibt Modellmetriken und Trainingsmetadaten als JSON-Datei.

        Die Methode erzeugt ein serialisierbares Payload mit Modellgüte,
        Ensemblegewichten, verwendeten Features und Trainingsmetadaten und
        speichert es als `metrics.json`.

        Parameter:
            Keine.

        Rückgabewerte:
            None: Die Methode schreibt eine JSON-Datei in das Ausgabeverzeichnis.

        Fehler/Sonderfälle:
            Keine explizite Fehlerbehandlung; Dateischreibfehler würden nach
            außen propagieren.

        Projektkontext:
            Die Methode unterstützt Transparenz, Nachvollziehbarkeit und mögliche
            Weiterverarbeitung der Modellbewertung außerhalb der GUI.
        """
        payload = {
            "metrics": self.metrics,
            "weights": self.weights,
            "feature_columns": self.feature_columns,
            "training_source": self.training_source,
            "training_mode": self.last_training_mode,
            "rf_n_estimators": int(getattr(self.rf_model, "n_estimators", 0) or 0),
            "model_file": str(self._model_path(sorted(self.trained_lines))) if self.trained_lines else "",
            "training_date_span": (
                None
                if self.training_date_span is None
                else [self.training_date_span[0].isoformat(), self.training_date_span[1].isoformat()]
            ),
        }
        (OUTPUT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


    def _write_predictions(self, frame: pd.DataFrame) -> None:
        """
        Persistiert Vorhersageergebnisse als CSV-Datei.

        Die Methode schreibt den übergebenen Ergebnisrahmen in standardisierter
        Form nach `predictions.csv`. Pandas `to_csv(...)` exportiert dabei einen
        DataFrame als CSV-Datei. [web:173]

        Parameter:
            frame (pd.DataFrame): Zu speichernde Vorhersagedaten.

        Rückgabewerte:
            None: Die Methode schreibt eine CSV-Datei.

        Fehler/Sonderfälle:
            Keine explizite Fehlerbehandlung.

        Projektkontext:
            Die Methode ermöglicht die externe Weiterverwendung von Prognosen,
            etwa für Debugging, Reporting oder weitere Analysepipelines.
        """
        frame.to_csv(OUTPUT_DIR / "predictions.csv", index=False, encoding="utf-8")


    def _write_schedule(self, frame: pd.DataFrame) -> None:
        """
        Persistiert adaptive Fahrplanvorschläge als CSV-Datei.

        Die Methode schreibt den übergebenen Fahrplanrahmen nach
        `adaptive_schedule.csv`, damit die Ergebnisse außerhalb der GUI verfügbar
        bleiben.

        Parameter:
            frame (pd.DataFrame): Zu speichernder adaptiver Fahrplan.

        Rückgabewerte:
            None: Die Methode schreibt eine CSV-Datei.

        Fehler/Sonderfälle:
            Auch leere DataFrames werden in der aktuellen Logik exportiert.

        Projektkontext:
            Die Methode macht operative Handlungsempfehlungen exportierbar und
            damit für weitere Auswertung oder Abstimmung nutzbar.
        """
        frame.to_csv(OUTPUT_DIR / "adaptive_schedule.csv", index=False, encoding="utf-8")