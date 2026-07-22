from __future__ import annotations

"""
Gradient-Boosting-Modell zur kostenbewussten Optimierung von Fahrplankursen.

Dieses Modul ergänzt die Nachfrageprognose des RF/GNN-Stacks um eine
nachgelagerte Optimierungsebene. Während die Vorhersagemodelle den erwarteten
Fahrgastandrang pro Linie und Stunde schätzen, lernt dieses Modell eine
zielnahe Anzahl sinnvoller Kurse, bei der Nachfrageabdeckung, Überlastung,
Unterauslastung und betriebliche Kosten gemeinsam berücksichtigt werden.

Die Persistenz des Modells erfolgt als Pickle-Datei, ergänzt um Metriken im
JSON-Format. Dadurch kann das Boosting-Modell iterativ weitertrainiert und
später in der adaptiven Fahrplanerzeugung wiederverwendet werden.
"""

import json
import math
import pickle
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import GradientBoostingRegressor
except Exception: # pragma: no cover - optional dependency fallback
    GradientBoostingRegressor = None

from .config import CONTEXT_COLUMNS, MODEL_DIR, OUTPUT_DIR, VEHICLE_TYPE_CAPACITY_FALLBACK
from .data import TrainingDataLoader

BOOSTING_MODEL_PATH = MODEL_DIR / "wvv_timetable_gradient_boosting.pkl"
BOOSTING_METRICS_PATH = OUTPUT_DIR / "timetable_gradient_boosting_metrics.json"


@dataclass
class BoostingTrainingResult:
    """
    Ergebniscontainer für ein Training des Fahrplan-Boosting-Modells.

    Die Struktur fasst die für Monitoring, CLI-Skripte und spätere
    Weiterverarbeitung relevanten Kennzahlen eines Trainingslaufs zusammen.
    Dazu gehören insbesondere die Datenmenge, die trainierten Linien, der
    mittlere absolute Fehler der Kursvorhersage sowie der verwendete
    Kostenparameter pro Busstunde.
    """

    rows: int
    lines: list[int]
    mae_runs: float
    n_estimators: int
    cost_per_bus_hour: float
    message: str


class _EmptyRepo:
    """
    Minimaler Repository-Ersatz für nicht interaktive Trainingskontexte.

    Diese Klasse erlaubt die Initialisierung des Optimierers auch dann, wenn
    kein vollständiges Datenrepository aus der GUI vorhanden ist. Sie stellt
    nur die kleinste benötigte Schnittstelle bereit, damit der
    ``TrainingDataLoader`` in Skript- und Batch-Szenarien instanziiert werden
    kann.
    """

    context_daily = pd.DataFrame()

    def load_line_range(self, *_args: Any, **_kwargs: Any) -> pd.DataFrame:
        """
        Liefert in der Fallback-Variante absichtlich keine Liniendaten.

        Die Methode dient ausschließlich dazu, die erwartete Repository-
        Schnittstelle technisch zu erfüllen. Reale Trainingsdaten werden in
        diesem Fall über andere Pfade bereitgestellt.
        """
        return pd.DataFrame()


class TimetableGradientBoostingOptimizer:
    """
    Kostenbewusster Optimierer für adaptive Kursanzahlen pro Linie und Stunde.

    Das Modell lernt aus historischen Fahrgast- und Fahrplandaten, wie viele
    Fahrten in einem Zeitfenster sinnvoll sind. Die Zielfunktion ist nicht
    direkt beobachtet, sondern wird aus einer heuristischen fachlichen
    Bewertungslogik abgeleitet: Nicht bediente Nachfrage, ineffiziente
    Überkapazität, vollständige Angebotslücken und Betriebskosten werden
    gegeneinander abgewogen.

    Projektkontext:
        Diese Komponente ist die Planungsstufe oberhalb der reinen
        Nachfrageprognose. Sie übersetzt prognostizierte Nachfrage in
        handlungsrelevante Angebotsentscheidungen und verbindet damit
        Vorhersagemodell und adaptiven Fahrplan.
    """

    def __init__(self, repo: Any | None = None) -> None:
        """
        Initialisiert den Optimierer samt Datenzugriff und Modellzustand.

        Wird kein echtes Repository übergeben, verwendet die Klasse einen
        minimalistischen Fallback. Standardmäßig startet der Optimierer mit
        einem Kostenansatz von 230 EUR pro Busstunde und ohne geladenes Modell.
        """
        self.repo = repo or _EmptyRepo()
        self.data_loader = TrainingDataLoader(self.repo)
        self.model: Any = None
        self.feature_columns: list[str] = []
        self.lines: list[int] = []
        self.cost_per_bus_hour = 230.0
        self.metrics: dict[str, float] = {"mae_runs": math.nan}
        self.n_estimators = 0

    def fit(
        self,
        lines: list[int],
        start: date,
        end: date,
        *,
        cost_per_bus_hour: float = 230.0,
        estimators_per_iteration: int = 40,
        warm_start: bool = True,
    ) -> BoostingTrainingResult:
        """
        Trainiert oder erweitert das Gradient-Boosting-Modell auf Stundenprofilen.

        Zunächst werden Rohdaten geladen und auf Linie/Datum/Stunde aggregiert.
        Anschließend wird für jedes Stundenprofil eine Zielgröße ``target_runs``
        berechnet, die eine fachlich sinnvolle Kursanzahl repräsentiert. Auf
        dieser künstlich konstruierten Zielvariable wird danach ein
        ``GradientBoostingRegressor`` trainiert.

        Parameter:
            lines (list[int]): Zu trainierende Linien.
            start (date): Beginn des historischen Trainingszeitraums.
            end (date): Ende des historischen Trainingszeitraums.
            cost_per_bus_hour (float): Monetärer Kostenparameter, der direkt in
                die Zielheuristik einfließt und damit das Modellverhalten
                beeinflusst.
            estimators_per_iteration (int): Anzahl neuer Bäume pro
                Trainingsiteration.
            warm_start (bool): Steuert, ob ein vorhandenes Modell geladen und
                inkrementell erweitert werden soll.

        Rückgabewerte:
            BoostingTrainingResult: Zusammenfassung des Trainingslaufs.

        Fehler/Sonderfälle:
            Bei zu wenigen Stundenprofilen wird bewusst kein Modell trainiert,
            da die aus der Heuristik abgeleitete Zielgröße sonst nicht robust
            genug wäre.
        """
        self.cost_per_bus_hour = max(0.0, float(cost_per_bus_hour))
        existing_loaded = self.load() if warm_start else False
        training = self.data_loader.load(lines, start, end)
        hourly = self._build_hourly_training_frame(training.raw)
        if hourly.empty or len(hourly) < 20:
            return BoostingTrainingResult(
                rows=0,
                lines=lines,
                mae_runs=math.nan,
                n_estimators=0,
                cost_per_bus_hour=self.cost_per_bus_hour,
                message="Zu wenig Daten fuer Gradient-Boosting-Fahrplantraining.",
            )

        hourly["target_runs"] = hourly.apply(self._optimal_runs_for_row, axis=1)
        self.feature_columns = self._feature_columns(hourly)
        for column in self.feature_columns:
            hourly[column] = pd.to_numeric(hourly[column], errors="coerce").fillna(0)

        train_frame, val_frame = self._time_split(hourly)
        self.model = self._fit_model(
            train_frame,
            existing_loaded=existing_loaded,
            estimators_per_iteration=estimators_per_iteration,
        )

        predicted = np.clip(self.model.predict(val_frame[self.feature_columns].to_numpy(dtype=float)), 0, None)
        actual = val_frame["target_runs"].to_numpy(dtype=float)
        mae = float(np.mean(np.abs(predicted - actual))) if len(actual) else math.nan
        self.metrics = {"mae_runs": mae}
        self.lines = sorted(int(line) for line in lines)
        self.n_estimators = int(getattr(self.model, "n_estimators", 0) or 0)
        self._save()
        self._write_metrics(rows=len(hourly))
        return BoostingTrainingResult(
            rows=len(hourly),
            lines=self.lines,
            mae_runs=mae,
            n_estimators=self.n_estimators,
            cost_per_bus_hour=self.cost_per_bus_hour,
            message=(
                f"Gradient Boosting trainiert: {len(hourly)} Stundenprofile, "
                f"MAE {mae:.2f} Kurse, {self.n_estimators} Trees."
            ),
        )

    def load(self) -> bool:
        """
        Lädt ein zuvor persistiertes Boosting-Modell samt Metadaten.

        Gespeichert werden neben dem eigentlichen Regressor auch die
        verwendeten Merkmale, trainierten Linien, Kostenparameter und die
        zuletzt berechneten Qualitätsmetriken. Die Versionsprüfung verhindert,
        dass inkompatible Modellstände stillschweigend übernommen werden.
        """
        if not BOOSTING_MODEL_PATH.exists():
            return False
        try:
            with BOOSTING_MODEL_PATH.open("rb") as handle:
                payload = pickle.load(handle)
        except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError):
            return False
        if payload.get("version") != 1:
            return False
        self.model = payload.get("model")
        self.feature_columns = list(payload.get("feature_columns", []))
        self.lines = [int(line) for line in payload.get("lines", [])]
        self.cost_per_bus_hour = float(payload.get("cost_per_bus_hour", 230.0))
        self.metrics = dict(payload.get("metrics", {"mae_runs": math.nan}))
        self.n_estimators = int(payload.get("n_estimators", 0) or 0)
        return self.model is not None and bool(self.feature_columns)

    def predict_target_runs(self, schedule: pd.DataFrame) -> pd.Series:
        """
        Schätzt für einen Fahrplanentwurf die empfohlene Kursanzahl.

        Die Methode transformiert einen bereits vorhandenen Stundenfahrplan in
        das Merkmalsformat des Boosting-Modells. Falls reale Nachfragewerte
        nicht direkt als ``demand`` vorliegen, werden alternativ
        ``predicted_boardings`` oder ``predicted_demand`` verwendet. So kann
        das Modell auf Prognoseergebnissen des RF/GNN-Stacks aufsetzen.
        """
        if self.model is None and not self.load():
            return pd.Series(dtype=float)
        frame = schedule.copy()
        if frame.empty:
            return pd.Series(dtype=float)
        if "predicted_boardings" in frame:
            frame["demand"] = pd.to_numeric(frame["predicted_boardings"], errors="coerce").fillna(0)
        elif "predicted_demand" in frame:
            frame["demand"] = pd.to_numeric(frame["predicted_demand"], errors="coerce").fillna(0)
        else:
            frame["demand"] = 0
        frame["weekday"] = pd.to_datetime(frame["date"].astype(str), errors="coerce").dt.weekday.fillna(0).astype(int)
        frame["month"] = pd.to_datetime(frame["date"].astype(str), errors="coerce").dt.month.fillna(1).astype(int)
        frame["is_weekend"] = frame["weekday"].isin([5, 6]).astype(int)
        frame["baseline_runs"] = pd.to_numeric(frame.get("baseline_runs", 0), errors="coerce").fillna(0)
        frame["avg_vehicle_capacity"] = pd.to_numeric(frame.get("avg_vehicle_capacity", 90), errors="coerce").fillna(90)
        frame["cost_per_bus_hour"] = self.cost_per_bus_hour
        for column in self.feature_columns:
            if column not in frame:
                frame[column] = 0
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
        prediction = self.model.predict(frame[self.feature_columns].to_numpy(dtype=float))
        return pd.Series(np.clip(prediction, 0, None), index=frame.index)

    def _build_hourly_training_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Verdichtet Rohdaten zu stündlichen Trainingsprofilen pro Linie.

        Aggregiert werden Nachfrage, Aussteiger, bisheriges Angebotsniveau,
        mittlere Fahrzeugkapazität, Spitzenlast sowie Auslastungskennzahlen.
        Zusätzlich werden die in der Projektübersicht beschriebenen
        Kontextmerkmale wie Ferien-, Vorlesungs- und Eventinformationen auf
        Stundenebene übernommen.

        Fehler/Sonderfälle:
            Fehlende Fahrzeugtypen werden über feste Standardkapazitäten aus
            der Konfiguration ersetzt. Das verhindert, dass der Optimierer bei
            unvollständigen Flottendaten keine Angebotsabschätzung mehr
            vornehmen kann.
        """
        if raw.empty:
            return pd.DataFrame()
        frame = raw.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
        frame = frame.dropna(subset=["date", "line", "hour"])
        if frame.empty:
            return pd.DataFrame()
        for column in ["line", "hour", "passenger_boarding", "passenger_exiting", "occupancy_departure", "vehicle_utilization"]:
            if column not in frame:
                frame[column] = 0
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
        if "vehicle_type" not in frame:
            frame["vehicle_type"] = "Unknown"
        frame["vehicle_type"] = frame["vehicle_type"].fillna("Unknown").astype(str)
        frame["capacity"] = frame["vehicle_type"].map(VEHICLE_TYPE_CAPACITY_FALLBACK).fillna(
            VEHICLE_TYPE_CAPACITY_FALLBACK["Unknown"]
        )

        aggregations = {
            "demand": ("passenger_boarding", "sum"),
            "exiting": ("passenger_exiting", "sum"),
            "baseline_runs": ("journey", "nunique"),
            "avg_vehicle_capacity": ("capacity", "mean"),
            "peak_load": ("occupancy_departure", "max"),
            "avg_utilization": ("vehicle_utilization", "mean"),
        }
        for column in CONTEXT_COLUMNS:
            if column in frame:
                aggregations[column] = (column, "max")
        hourly = frame.groupby(["date", "line", "hour"], as_index=False).agg(**aggregations)
        hourly["weekday"] = pd.to_datetime(hourly["date"].astype(str)).dt.weekday
        hourly["month"] = pd.to_datetime(hourly["date"].astype(str)).dt.month
        hourly["is_weekend"] = hourly["weekday"].isin([5, 6]).astype(int)
        hourly["cost_per_bus_hour"] = self.cost_per_bus_hour
        for column in CONTEXT_COLUMNS:
            if column not in hourly:
                hourly[column] = 0
        return hourly.fillna(0)

    def _feature_columns(self, frame: pd.DataFrame) -> list[str]:
        """
        Definiert den Merkmalsraum des Boosting-Modells.

        Neben den offensichtlichen Nachfrage- und Angebotsgrößen werden auch
        Kalender- und Kontextsignale genutzt, damit der Optimierer etwa auf
        Vorlesungsbetrieb, Feiertage oder Eventlagen unterschiedlich reagieren
        kann.
        """
        base = [
            "line",
            "hour",
            "weekday",
            "month",
            "is_weekend",
            "demand",
            "exiting",
            "baseline_runs",
            "avg_vehicle_capacity",
            "peak_load",
            "avg_utilization",
            "cost_per_bus_hour",
        ]
        return [column for column in base + CONTEXT_COLUMNS if column in frame]

    def _optimal_runs_for_row(self, row: pd.Series) -> int:
        """
        Leitet aus einem Stundenprofil eine pseudo-optimale Kursanzahl ab.

        Diese Funktion ist der fachliche Kern des Moduls. Sie berechnet keine
        direkt beobachtete Zielvariable, sondern erzeugt sie heuristisch aus
        Nachfrage, Fahrzeugkapazität, vorhandenem Basisangebot und Kosten.

        Die Bewertungslogik enthält mehrere bewusst gesetzte Schwellenwerte:
        - minimale Kapazität von 35 Plätzen zur Vermeidung unrealistischer Werte,
        - Obergrenze der Suchmenge auf maximal 16 Kurse,
        - starke Strafe für nicht bediente Nachfrage,
        - moderate Strafe für Überangebot,
        - explizite Zusatzstrafe für vollständige Angebotslücken bei relevanter Nachfrage,
        - Komfortbonus bei hoher Abdeckung.

        Projektkontext:
            Die auf diese Weise erzeugte Zielgröße übersetzt die fachliche
            Optimierungsabsicht des Projekts in ein überwachtes Lernproblem.
            Das Gradient Boosting approximiert anschließend diese Regelbasis.
        """
        demand = max(0.0, float(row.get("demand", 0.0) or 0.0))
        capacity = max(35.0, float(row.get("avg_vehicle_capacity", 90.0) or 90.0))
        baseline = max(0, int(round(float(row.get("baseline_runs", 0.0) or 0.0))))
        if demand <= 0:
            return 1 if baseline > 0 else 0
        max_runs = max(1, min(16, max(baseline + 4, int(math.ceil(demand / (capacity * 0.62))) + 3)))
        cost_weight = max(0.4, min(2.5, self.cost_per_bus_hour / 230.0))
        best_runs = 0
        best_score = math.inf
        for runs in range(0, max_runs + 1):
            offered = runs * capacity
            served = min(demand, offered)
            unserved = max(0.0, demand - offered)
            underload = max(0.0, offered - demand)
            coverage = served / max(demand, 1.0)
            overload_penalty = unserved * 9.0
            underload_penalty = underload * 0.22 * cost_weight
            cost_penalty = runs * self.cost_per_bus_hour * 0.12
            service_gap_penalty = 180.0 if runs == 0 and demand >= capacity * 0.35 else 0.0
            comfort_bonus = coverage * 80.0
            score = overload_penalty + underload_penalty + cost_penalty + service_gap_penalty - comfort_bonus
            if score < best_score:
                best_score = score
                best_runs = runs
        if baseline > 0 and best_runs == 0 and demand >= capacity * 0.18:
            best_runs = 1
        return int(best_runs)

    def _time_split(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Trennt die Stundenprofile chronologisch in Trainings- und Validierungsmenge.

        Der Split erfolgt datumsbasiert mit rund 82 % Trainingsanteil. Damit
        wird eine realistischere Bewertung erreicht als bei zufälligem Mischen,
        weil spätere Zeiträume als approximative Zukunft behandelt werden.
        """
        dates = sorted(frame["date"].unique())
        split = max(1, int(len(dates) * 0.82))
        if split >= len(dates):
            split = len(dates) - 1
        train_dates = set(dates[:split])
        train = frame[frame["date"].isin(train_dates)].copy()
        val = frame[~frame["date"].isin(train_dates)].copy()
        if train.empty or val.empty:
            midpoint = max(1, len(frame) - max(10, len(frame) // 5))
            train = frame.iloc[:midpoint].copy()
            val = frame.iloc[midpoint:].copy()
        return train, val

    def _fit_model(self, train_frame: pd.DataFrame, *, existing_loaded: bool, estimators_per_iteration: int) -> Any:
        """
        Trainiert den eigentlichen ``GradientBoostingRegressor``.

        Bei vorhandenem Modellstand wird per ``warm_start`` inkrementell
        weitertrainiert, indem zusätzliche Bäume ergänzt werden. Andernfalls
        wird ein neues Modell mit konservativen Hyperparametern initialisiert,
        die auf robuste Generalisierung statt maximale Modellkomplexität zielen.
        """
        if GradientBoostingRegressor is None:
            raise RuntimeError("scikit-learn GradientBoostingRegressor ist nicht verfuegbar.")
        if existing_loaded and isinstance(self.model, GradientBoostingRegressor):
            model = self.model
            current = int(getattr(model, "n_estimators", 80) or 80)
            model.set_params(warm_start=True, n_estimators=current + max(1, int(estimators_per_iteration)))
        else:
            model = GradientBoostingRegressor(
                n_estimators=max(20, int(estimators_per_iteration)),
                learning_rate=0.055,
                max_depth=3,
                min_samples_leaf=8,
                subsample=0.88,
                random_state=44,
                warm_start=True,
            )

        model.fit(
            train_frame[self.feature_columns].to_numpy(dtype=float),
            train_frame["target_runs"].to_numpy(dtype=float),
        )
        return model

    def _save(self) -> None:
        """
        Persistiert Modell und Metadaten atomar auf Dateisystemebene.

        Die Zwischenspeicherung in einer temporären Datei reduziert das Risiko
        eines korrupten Modellartefakts bei abgebrochenen Schreiboperationen.
        """
        payload = {
            "version": 1,
            "model": self.model,
            "feature_columns": self.feature_columns,
            "lines": self.lines,
            "cost_per_bus_hour": self.cost_per_bus_hour,
            "metrics": self.metrics,
            "n_estimators": self.n_estimators,
        }
        tmp = BOOSTING_MODEL_PATH.with_suffix(".tmp")
        with tmp.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(BOOSTING_MODEL_PATH)

    def _write_metrics(self, *, rows: int) -> None:
        """
        Schreibt Trainingsmetriken als lesbares JSON für Monitoring und GUI.

        Neben dem Fehlermaß werden auch Datenumfang, Kostenannahme,
        Baumanzahl und trainierte Linien abgelegt, damit Modellstände später
        fachlich nachvollzogen werden können.
        """
        payload = {
            "rows": rows,
            "lines": self.lines,
            "cost_per_bus_hour": self.cost_per_bus_hour,
            "metrics": self.metrics,
            "n_estimators": self.n_estimators,
        }
        BOOSTING_METRICS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")