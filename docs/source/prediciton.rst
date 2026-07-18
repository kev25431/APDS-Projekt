Prediciton
==========

Dieser Ordner enthält die Vorhersage- und Optimierungslogik.

- ``/prediction/__init__.py``: Export-Datei für das Paket. Sie macht die Hauptklassen und Hilfsfunktionen aus dem Vorhersagepaket über den Paketnamen verfügbar.
- ``/prediction/config.py``: Zentrale Konfiguration für das Vorhersage-Modul. Enthält Pfade zu Modell- und Output-Ordnern, Kontextspalten, Fahrzeugkapazitäten und Trainingsparameter.
- ``/prediction/data.py``: Lädt Trainingsdaten. Bevorzugt die bereinigten 2025-Parquet-Dateien aus dem enriched-Ordner. Falls diese fehlen, greift sie auf die Dashboard-Daten im Repository zurück.
- ``/prediction/models.py``: Enthält die eigentlichen Vorhersagemodelle. Dazu gehören ein Fallback-Regressor für den Fall, dass scikit-learn nicht verfügbar ist, sowie ein graph-light Modell, das stationäre Profile mit Nachbarinformationen kombiniert.
- ``/prediction/service.py``: Zentrale Service-Klasse ``DemandPredictionService``. Sie trainiert Modelle, speichert sie, erzeugt Vorhersagen für Linien, Tage und Stunden und baut aus diesen Vorhersagen einen adaptiven Fahrplan.
- ``/prediction/types.py``: Datentypen für die Vorhersage-Pipeline. Enthält ``PredictionResult`` für die Ausgabe eines Vorhersage-Laufs und ``TrainingData`` für die Trainingsdatenstruktur.
- ``/prediction/utils.py``: Kleine Hilfsfunktion zur Normalisierung von Stationsnamen.
- ``/prediction/boosting.py``: Gradient-Boosting-Optimierer für die Anzahl von Fahrten pro Stunde. Er lernt aus historischen Daten, wie viele Fahrten in Abhängigkeit von Nachfrage, Kapazität und Kosten sinnvoll sind.
- ``/prediction/models/``: Ordner für gespeicherte Modell-Dateien, zum Beispiel trainierte Pickle-Dateien.
- ``/prediction/outputs/``: Ordner für Ausgabe-Dateien wie Vorhersagen, adaptive Fahrpläne und Metriken.


Logik Random-Forest-Modell
-----------------------------
Das RandomForestRegressor-Modell wird in ``/prediction/models.py`` hat folgende


Graph-Light-Modell
------------------
Das zweite Vorhersagemodell ist kein tiefes neuronales Graph Neural Network, sondern ein leichtgewichtiges, graphbasiertes Nachbarschaftsmodell. Die Implementierung befindet sich in `prediction/models.py`.

Grundidee:

Eine Haltestelle wird nicht isoliert betrachtet. Das Modell ermittelt anhand realer Fahrten, welche Haltestellen direkt aufeinanderfolgen. Aus diesen Verbindungen entsteht ein Nachbarschaftsgraph.

Für jede Haltestelle berechnet das Modell historische Mittelwerte auf mehreren Aggregationsebenen:

1. **Exaktes Profil**
   Mittelwert je `line`, `station_key`, `hour` und `weekday`

2. **Stundenprofil**
   Mittelwert je `line`, `station_key` und `hour`

3. **Haltestellenprofil**
   Mittelwert je `line` und `station_key`

4. **Globaler Mittelwert**
   Wird verwendet, wenn auf den vorherigen Ebenen kein historischer Wert verfügbar ist.

Zunächst wird der historische Mittelwert der jeweiligen Haltestelle bestimmt:


own = historischer Mittelwert der Haltestelle


Anschließend werden die benachbarten Haltestellen derselben Linie berücksichtigt:


neighbor_mean = Mittelwert der historischen Werte der Nachbarhaltestellen


Sind Nachbarwerte vorhanden, wird die Vorhersage als gewichtete Kombination aus dem eigenen historischen Wert und dem Mittelwert der Nachbarhaltestellen berechnet:


GNN_prediction = 0.72 × own + 0.28 × neighbor_mean


Sind keine Nachbarwerte verfügbar, entspricht die Vorhersage dem eigenen historischen Wert:


GNN_prediction = own


Das Modell erzeugt zwei Vorhersagewerte:

* `pred_gnn_boardings`: prognostizierte Anzahl der Einstiege
* `pred_gnn_exiting`: prognostizierte Anzahl der Ausstiege

Modellgewichtung
-----------------
Nach dem Training wird nach Datum getrennt:

Trainingsdaten: erste 80 Prozent der Tage
Validierung:   letzte 20 Prozent der Tage

Für Random Forest und Graph-Light wird der MAE berechnet:

MAE = mean(abs(actual - predicted))

Dann werden die Gewichte automatisch gesetzt:

rf_score  = 1 / (rf_mae  + 0.001)
gnn_score = 1 / (gnn_mae + 0.001)

Summe:

total_score = rf_score + gnn_score

Gewichte:

weight_rf  = rf_score  / total_score
weight_gnn = gnn_score / total_score

Das bessere Modell bekommt also automatisch mehr Gewicht.