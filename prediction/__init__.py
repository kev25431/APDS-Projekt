"""
Öffentliche Exporte des Prediction-Pakets.

Dieses Modul bündelt die zentralen Einstiegspunkte des Prediction-Teilprojekts,
damit andere Komponenten – insbesondere GUI, Skripte und Trainingsroutinen –
die wichtigsten Klassen und Hilfsfunktionen über einen stabilen Paketimport
verwenden können.

Projektkontext:
    Das Paket stellt die Nachfrageprognose für Ein- und Aussteiger bereit.
    Exportiert werden der zentrale Service zur Vorhersage, der standardisierte
    Ergebniscontainer sowie die Normalisierung von Haltestellennamen als
    gemeinsame Hilfsfunktion für Daten- und Modelllogik.
"""
from .service import DemandPredictionService
from .types import PredictionResult
from .utils import normalize_station_name

__all__ = ["DemandPredictionService", "PredictionResult", "normalize_station_name"]