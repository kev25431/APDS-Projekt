Nachfrageprognose
=================

Das Python-Paket ``prediction`` prognostiziert die Nachfrage je Linie, Haltestelle und
Stunde. Die Ausgabe enthält erwartete Ein- und Ausstiege. Daraus leitet der
Service zusätzlich eine stündliche Empfehlung für die Zahl der Fahrten ab.

Die konkrete Vergleichstabelle aus WVV- und adaptivem Fahrplan entsteht danach
im Dashboard. Sie verwendet die Regeln aus ``wvv_dashboard/service_policy.py``
und die vorhandenen WVV-Fahrplandaten. Das Gradient-Boosting-Modell in
``prediction/boosting.py`` ist ein zusätzlicher, separat trainierbarer Optimierer;
es erzeugt derzeit nicht die Fahrplantabelle der GUI.

Datenbasis
----------

``TrainingDataLoader`` lädt bevorzugt bereinigte Parquet-Dateien aus
``downloads/wvv-pjs-2026/full_api_data_enriched_2025``. Berücksichtigt werden
Dateien, deren Namen dem folgenden Schema entsprechen:

.. code-block:: text

   data_2025-MM-DD_2025-MM-DD_line_<Linie>_clean_context_2025.parquet

Wenn dort keine passende Datei vorhanden ist, lädt der Service die Daten über
das Dashboard-Repository aus ``downloads/wvv-pjs-2026/full_api_data``.

Für das Training werden insbesondere folgende Rohspalten verwendet:

* ``date``, ``line``, ``station``, ``station_number`` und ``hour``
* ``journey``
* ``passenger_boarding`` und ``passenger_exiting``
* ``occupancy_departure`` und ``vehicle_utilization``
* ``vehicle_type``

Der Schlüssel ``station_key`` wird durch die Normalisierung des
Haltestellennamens erzeugt.

Kontextmerkmale
~~~~~~~~~~~~~~~

Soweit sie in den Daten oder im täglichen Kontext des Repositories vorhanden
sind, verwendet die Pipeline diese Kontextspalten:

* ``lecture_period_jmu`` und ``lecture_period_thws``
* ``public_holiday``, ``nationwide`` und ``school_holiday``
* ``event_hours``, ``concert_hours``, ``event_count`` und
  ``concert_event_count``
* ``total_event_duration_hours`` und ``max_event_duration_hours``
* ``event_day``, ``concert_day``, ``event_hour`` und ``concert_hour``
* ``verkaufsoffener_sonntag``

Fehlende Kontextspalten werden mit ``0`` belegt. In den angereicherten Daten
wird ``event_hours`` gegebenenfalls durch ``event_hour`` und ``concert_hours``
durch ``concert_hour`` ergänzt.

Aufbereitung der Trainingsdaten
-------------------------------

``DemandPredictionService._build_feature_frame`` gruppiert die Rohdaten nach

.. code-block:: text

   date, line, station_number, station, station_key, hour

und berechnet pro Gruppe:

.. list-table:: Aggregierte Trainingswerte
   :header-rows: 1
   :widths: 30 70

   * - Ausgabespalte
     - Berechnung
   * - ``boardings``
     - Summe von ``passenger_boarding``
   * - ``exiting``
     - Summe von ``passenger_exiting``
   * - ``journeys``
     - Anzahl eindeutiger Werte in ``journey``
   * - ``peak_load``
     - Maximum von ``occupancy_departure``
   * - ``avg_utilization``
     - Mittelwert von ``vehicle_utilization``
   * - Kontextspalten
     - Jeweiliges Maximum innerhalb der Gruppe

Zusätzlich entstehen ``weekday`` (Montag ``0`` bis Sonntag ``6``), ``month`` (``1-12``)
und ``is_weekend``. Jede Haltestelle erhält außerdem über ``station_key`` eine
interne numerische ``station_id``.

Die Modellmatrix verwendet diese Basisfeatures:

.. code-block:: text

   line, station_number, station_id, hour, weekday, month, is_weekend,
   journeys, peak_load, avg_utilization

Dazu kommen alle verfügbaren Kontextmerkmale. Zielmatrix des Trainings ist
``[boardings, exiting]``; beide Werte werden somit gemeinsam vorhergesagt.

Random Forest
-------------

Wenn scikit-learn verfügbar ist, trainiert der Service einen
``RandomForestRegressor`` mit folgenden Parametern:

.. list-table:: Random-Forest-Konfiguration
   :header-rows: 1

   * - Parameter
     - Wert
   * - ``n_estimators`` beim ersten Training
     - ``90``
   * - zusätzliche Bäume beim inkrementellen Training
     - standardmäßig ``30``
   * - ``max_depth``
     - ``16``
   * - ``min_samples_leaf``
     - ``3``
   * - ``random_state``
     - ``42``
   * - ``n_jobs``
     - ``1``
   * - ``warm_start``
     - ``True``

Beim inkrementellen Training wird ein passendes gespeichertes Modell geladen
und seine Baumzahl erhöht. Ohne scikit-learn verwendet der Service stattdessen
``GroupFallbackRegressor``. Dieses Fallback arbeitet mit historischen
Mittelwerten je Linie, Haltestelle, Stunde und Wochentag, anschließend je Linie
und Haltestelle und zuletzt mit einem globalen Mittelwert.

Graph-Light-Modell
------------------

``GraphMessagePassingRegressor`` ist ein leichtes Nachbarschaftsmodell und kein
tiefes neuronales Graph Neural Network. Beim Training werden aufeinanderfolgende
Haltestellen einer realen Fahrt als Nachbarn gespeichert. Zusätzlich bildet das
Modell historische Mittelwerte in dieser Reihenfolge:

#. Linie, Haltestelle, Stunde und Wochentag
#. Linie, Haltestelle und Stunde
#. Linie und Haltestelle
#. globaler Mittelwert

Für eine Haltestelle wird zunächst das bestmögliche eigene Profil ``own``
ermittelt. Sind Nachbarn bekannt, wird deren Mittelwert ``neighbor_mean``
einbezogen:

.. code-block:: text

   graph_prediction = 0.72 * own + 0.28 * neighbor_mean

Ohne bekannte Nachbarn entspricht die Graph-Light-Prognose dem eigenen Profil.
Auch dieses Modell liefert jeweils einen Wert für Ein- und Ausstiege.

Zeitliche Validierung und Ensemble
----------------------------------

Die vorhandenen Tage werden chronologisch sortiert. Die ersten ``80 %`` der
Tage bilden den Trainingssatz, die letzten ``20 %`` den Validierungssatz. Nur
wenn dieser datumsbasierte Split leere Teilmengen erzeugt, greift eine
zeilenbasierte Ersatzaufteilung.

Für Random Forest und Graph-Light wird der mittlere absolute Fehler berechnet:

.. code-block:: text

   MAE = mean(abs(actual - max(predicted, 0)))

Ein kleinerer MAE bedeutet einen kleineren durchschnittlichen absoluten Fehler
auf den zurückgehaltenen historischen Daten. Aus beiden Fehlern entstehen die
Ensemble-Gewichte:

.. code-block:: text

   rf_score  = 1 / (rf_mae  + 0.001)
   gnn_score = 1 / (gnn_mae + 0.001)

   weight_rf  = rf_score  / (rf_score + gnn_score)
   weight_gnn = gnn_score / (rf_score + gnn_score)

Falls einer der Fehler nicht endlich ist, werden beide Gewichte auf ``0.5``
gesetzt. Die finale Prognose lautet getrennt nach Zielwert:

.. code-block:: text

   pred_boardings = weight_rf * rf_boardings + weight_gnn * gnn_boardings
   pred_exiting   = weight_rf * rf_exiting   + weight_gnn * gnn_exiting
   pred_total     = pred_boardings + pred_exiting

Negative Modellwerte werden in der Ausgabe auf ``0`` begrenzt. Neben ``rf_mae``
und ``gnn_mae`` wird auch ``ensemble_mae`` gespeichert. Diese Werte sind ein
historisches Backtesting-Ergebnis^, sie zeigen also wir gut das Modell auf zurückgehaltene Daten funktioniert hat; sie garantieren nicht die Genauigkeit einer
Prognose für einen zukünftigen Tag.

Prognosen für zukünftige Zeitpunkte
-----------------------------------

Für eine gewählte Linie baut der Service aus dem Trainingskatalog je bekannter
Haltestelle eine Zeile. Datum, Stunde, Wochentag, Monat und Wochenendkennzeichen
stammen aus dem gewählten Zeitpunkt. Da für diesen Zukunftszeitpunkt keine
gemessenen Betriebswerte vorliegen, setzt der Service:

.. code-block:: text

   journeys       = 1
   peak_load      = 0
   avg_utilization = 0

Danach werden die Kontextmerkmale ergänzt und beide Modelle ausgeführt.
``predict`` berechnet eine einzelne Stunde. ``predict_short_term`` verarbeitet
einen Horizont von mindestens einer und höchstens zwölf Stunden; ein Übergang
über Mitternacht wird auf den Folgetag übertragen. Optional kann die Ausgabe auf
ausgewählte ``station_keys`` beschränkt werden.

Stündliche Fahrtenempfehlung im Prediction-Service
--------------------------------------------------

Ohne Haltestellenfilter aggregiert ``predict_short_term`` die Prognosen je Datum,
Linie und Stunde. ``predicted_boardings`` ist die primäre Nachfragegröße für die
Fahrtenzahl; nur wenn sie ``0`` ist, wird ``predicted_demand`` verwendet.

Historische Größen
~~~~~~~~~~~~~~~~~~

Je Stunde werden historische Einstiege, eindeutige Fahrten und eine mittlere
Fahrzeugkapazität bestimmt. Zunächst gelten diese Kapazitäts-Fallbacks:

.. list-table:: Fahrzeugkapazitäten
   :header-rows: 1

   * - Fahrzeugtyp
     - Kapazität
   * - ``SKOM``
     - ``80``
   * - ``GKOM``
     - ``140``
   * - ``GTE``
     - ``180``
   * - ``GTN``
     - ``170``
   * - ``Unknown``
     - ``90``

Wenn sich aus ``occupancy_departure / vehicle_utilization`` Werte zwischen
``25`` und ``260`` ableiten lassen, ersetzt der Median je Fahrzeugtyp den
jeweiligen Fallback.

Für Stunden ohne eigene Historie verwendet der Service das 75-%-Quantil der
historischen Nachfrage, die durchschnittliche Fahrtenzahl und die
durchschnittliche Fahrzeugkapazität. Das aufgerundete 95-%-Quantil der
historischen Fahrtenzahl bildet das ``fleet_limit``.

Kosten- und Auslastungsparameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standardmäßig kostet eine Busstunde ``230 EUR``. Der Kostenfaktor und die daraus
abgeleiteten Grenzen werden so berechnet:

.. code-block:: text

   cost_pressure = clamp(bus_hourly_cost / 230, 0.65, 1.65)
   target_utilization = clamp(0.82 + (cost_pressure - 1) * 0.08,
                              0.74, 0.92)
   productive_share = clamp(0.34 + (cost_pressure - 1) * 0.10,
                            0.28, 0.52)

Die erforderliche Fahrtenzahl kombiniert Kapazität und historisches Profil:

.. code-block:: text

   capacity_required_runs = ceil(demand / (avg_capacity * target_utilization))
   profile_required_runs = ceil(base_runs * max(demand / demand_base, 0.45))
   required_runs = max(capacity_required_runs, profile_required_runs)
   recommended_runs = clamp(required_runs, 1, fleet_limit)

Solange mehr als eine Fahrt empfohlen wird und die Nachfrage je Fahrt unter
``avg_capacity * productive_share`` liegt, reduziert der Service die Empfehlung
um eine Fahrt. 

Die Aktion lautet abhängig vom Ergebnis ``Verstärken``,
``Ausdünnen``, ``Halten`` oder bei Überschreitung des Flottenlimits
``Priorisieren``.

Diese Service-Ausgabe ist eine erste stündliche Empfehlung. Beim Aufbau der
konkreten GUI-Fahrplantabelle wird die Zahl erneut gegen den tatsächlichen
WVV-Bestand und die linienabhängige Service-Policy geprüft. Details stehen unter
:doc:`dashboard`.

Optionaler Gradient-Boosting-Optimierer
---------------------------------------

``TimetableGradientBoostingOptimizer`` in ``prediction/boosting.py`` lernt eine
Zielzahl von Fahrten je Linie und Stunde. Seine Features sind:

.. code-block:: text

   line, hour, weekday, month, is_weekend, demand, exiting,
   baseline_runs, avg_vehicle_capacity, peak_load, avg_utilization,
   cost_per_bus_hour

Dazu kommen die Kontextmerkmale. Der zeitliche Split verwendet ``82 %`` der
Tage zum Training und ``18 %`` zur Validierung. Der
``GradientBoostingRegressor`` wird mit folgenden Parametern angelegt:

.. code-block:: text

   n_estimators     = max(20, estimators_per_iteration)
   learning_rate    = 0.055
   max_depth        = 3
   min_samples_leaf = 8
   subsample        = 0.88
   random_state     = 44
   warm_start       = True

Für jede historische Stundenzeile sucht der Optimierer zunächst regelbasiert
die Fahrtenzahl mit dem kleinsten Score. Berücksichtigt werden nicht bediente
Nachfrage, Überangebot, Busstundenkosten, eine Versorgungslücke bei null Fahrten
und ein Komfortbonus für abgedeckte Nachfrage. Das trainierte Modell approximiert
anschließend diese Zielwerte. Als Qualitätswert wird der MAE der Fahrtenzahl auf
dem Validierungssatz gespeichert.

Persistierte Dateien
--------------------

``prediction/models`` enthält die Pickle-Dateien der Modelle:

* ``wvv_prediction_lines_<Linien>.pkl`` für das Ensemble
* ``wvv_timetable_gradient_boosting.pkl`` für den optionalen Optimierer

``prediction/outputs`` enthält Laufzeit- und Ergebnisdateien, unter anderem:

* ``metrics.json``
* ``predictions.csv``
* ``adaptive_schedule.csv``
* ``timetable_gradient_boosting_metrics.json``

Gespeicherte Ensemble-Modelle werden nur geladen, wenn Modellversion und exakt
angeforderte Linienmenge übereinstimmen.

Modulübersicht
--------------

* ``prediction/config.py``: Pfade, Kontextspalten, Kapazitäts-Fallbacks und
  Trainingskonstanten
* ``prediction/data.py``: Laden und Normalisieren der Trainingsdaten
* ``prediction/models.py``: Fallback- und Graph-Light-Regressor
* ``prediction/service.py``: Training, Validierung, Vorhersage, Persistenz und
  erste stündliche Fahrtenempfehlung
* ``prediction/boosting.py``: optionaler Gradient-Boosting-Optimierer
* ``prediction/types.py``: ``PredictionResult`` und ``TrainingData``
* ``prediction/utils.py``: Normalisierung von Haltestellennamen
* ``prediction/__init__.py``: öffentliche Exporte des Pakets

Grenzen der Aussagekraft
------------------------

Die Prognose extrapoliert aus historischen Daten und Kontextmerkmalen. Sie ist
kein Live-Fahrgastzähler. Der adaptive Fahrplan ist ein rechnerischer Vorschlag
und berücksichtigt keine vollständige Fahrer- und Fahrzeugumlaufplanung,
Pausen, Depotfahrten, Betriebsstörungen oder die tatsächliche Verfügbarkeit von
Fahrzeugen. Die GUI verändert auf Basis des vorhandenen WVV-Fahrplans die Zahl
und zeitliche Verteilung der Fahrten; sie optimiert weder die Route noch die
Fahrzeit zwischen den Haltestellen frei.
