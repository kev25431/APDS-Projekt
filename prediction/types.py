from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd


@dataclass
class PredictionResult:
    """
    Standardisierter Rückgabewert für Vorhersage- und Trainingsoperationen.

    Die Klasse kapselt tabellarische Vorhersagen, Qualitätsmetriken,
    Ensemble-Gewichte, Statusmeldungen sowie optional einen bereits aus
    der Nachfrageprognose abgeleiteten adaptiven Stundenfahrplan.

    Parameter:
        predictions (pd.DataFrame): Tabelle mit modellierten Vorhersagen,
            typischerweise pro Haltestelle, Linie, Datum und Stunde.
        metrics (dict[str, float]): Gütemaße der trainierten Modelle,
            insbesondere MAE-Werte für Random Forest, Graph-Modell und Ensemble.
        weights (dict[str, float]): Gewichte der Ensemble-Komponenten, die aus
            der historischen Validierung abgeleitet wurden.
        message (str): Fachlich verständliche Statusmeldung zum Ergebnis.
        schedule (pd.DataFrame): Optional erzeugter adaptiver Fahrplan auf
            Stundenebene oder als weiterverarbeitete Planungstabelle.

    Rückgabewerte:
        PredictionResult: Instanz mit allen relevanten Ergebnisdaten.

    Fehler/Sonderfälle:
        Leere DataFrames sind zulässig und werden verwendet, wenn Training oder
        Vorhersage mangels Daten, Modellzustand oder Linienbezug nicht möglich ist.

    Projektkontext:
        Der Container sorgt dafür, dass GUI, Skripte und spätere Dokumentation
        auf ein einheitliches Ergebnisformat zugreifen können. Dadurch werden
        technische Modellinformationen und fachliche Planungsergebnisse gemeinsam
        transportiert.
    """

    predictions: pd.DataFrame
    metrics: dict[str, float]
    weights: dict[str, float]
    message: str
    schedule: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class TrainingData:
    """
    Container für geladene Trainingsdaten inklusive Herkunftsnachweis.

    Die Klasse beschreibt nicht nur die eigentlichen Rohdaten, sondern auch die
    Datenquelle und den zeitlichen Abdeckungsbereich. Dadurch kann der Service
    transparent dokumentieren, ob mit bevorzugten 2025-Enriched-Daten oder mit
    einem Fallback aus dem App-Datenbestand gearbeitet wurde.

    Parameter:
        raw (pd.DataFrame): Geladene Rohdaten für das Training.
        source (str): Bezeichnung der tatsächlich genutzten Datenquelle.
        date_span (tuple[date, date] | None): Frühestes und spätestes Datum der
            Trainingsbasis oder None bei fehlenden Daten.

    Rückgabewerte:
        TrainingData: Strukturierte Trainingsbasis für nachgelagerte Modelle.

    Fehler/Sonderfälle:
        Bei fehlenden Trainingsdaten kann raw leer und date_span None sein.
        Diese Konstellation wird von aufrufenden Komponenten explizit geprüft.

    Projektkontext:
        Die Klasse trennt Datenbeschaffung und Modelltraining klar voneinander
        und macht die Datenherkunft für wissenschaftliche Nachvollziehbarkeit
        sowie GUI-Statusmeldungen sichtbar.
    """

    raw: pd.DataFrame
    source: str
    date_span: tuple[date, date] | None