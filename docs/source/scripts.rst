Aufbereitungs- und Trainingsskripte
===================================

Der Ordner ``scripts`` enthält die Kommandozeilenprogramme für Datenaufbereitung,
Fahrplanbereinigung und Modelltraining. Alle folgenden Befehle werden aus dem
Projektstamm ausgeführt.

2025-Daten aufbereiten
----------------------

``scripts/build_2025_enriched_parquets.py`` liest WVV-Parquet-Dateien,
beschränkt sie auf ein Jahr, bereinigt Messwerte und ergänzt Kalender- und
Kontextdaten.

Ein Lauf mit den Standardpfaden:

.. code-block:: console

   python3 scripts/build_2025_enriched_parquets.py

Standardverzeichnisse
~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Ein- und Ausgaben
   :header-rows: 1

   * - Zweck
     - Pfad
   * - Rohdaten
     - ``downloads/wvv-pjs-2026/full_api_data``
   * - Zusatzdaten
     - ``downloads/wvv-pjs-2026/Additional Data``
   * - Angereicherte Ausgabe
     - ``downloads/wvv-pjs-2026/full_api_data_enriched_2025``
   * - Reduzierte ML-Ausgabe
     - ``downloads/wvv-pjs-2026/model_ready_2025``

Das Skript verarbeitet nur Kandidatendateien, deren Name das gewählte Jahr
enthält und deren Parquet-Schema die Spalte ``vehicle_station`` enthält. Die
Zeilen werden zusätzlich anhand von ``main_date_day`` oder ersatzweise
``report_date`` auf das gewählte Jahr gefiltert.

Bereinigung
~~~~~~~~~~~

Die Aufbereitung führt unter anderem folgende belegte Schritte aus:

* Duplikate werden über ``stop_event_id`` oder, falls die Spalte fehlt, über die
  vollständige Zeile entfernt.
* Negative Werte in definierten nichtnegativen Messspalten werden zunächst als
  fehlend behandelt.
* Ein- und Ausstiege werden numerisch konvertiert, mit ``0`` aufgefüllt und auf
  Ganzzahlen gerundet.
* ``passenger_change`` wird als Einsteiger minus Aussteiger neu berechnet.
* Fehlende ``occupancy_departure``-Werte werden soweit möglich über die
  kumulierte Fahrgaständerung innerhalb einer Fahrt sowie Vorwärts-/Rückwärts-
  Auffüllung ergänzt.
* ``vehicle_utilization`` wird auf höchstens ``2.0`` begrenzt und mit Medianen
  aufgefüllt.
* Weitere numerische Werte werden zunächst linienweise und anschließend global
  über Mediane ergänzt.
* Fehlende Textwerte erhalten, mit Ausnahme des Namens eines verkaufsoffenen
  Sonntags, den Wert ``Unknown``.

Die Stunde wird aus der ersten verwendbaren Spalte in dieser Reihenfolge
abgeleitet: ``departure_plan_station``, ``departure_journey``,
``departure_plan_journey``, ``report_date``. Daraus entsteht auch
``departure_minute_of_day``.

Kontextdaten
~~~~~~~~~~~~

Der tägliche Kontext kann folgende Dateien aus dem Zusatzdatenordner einlesen:

* ``bavarian_public_holidays_daily.csv``
* ``bavarian_school_holidays_daily.csv``
* ``lectures_daily.csv``
* ``lectures.csv``
* ``events.csv``
* ``verkaufsoffene_sonntage.csv``

Aus Events entstehen sowohl tägliche Merkmale als auch ``event_hour`` und
``concert_hour`` je Stunde. Mit ``--include-streets`` werden zusätzlich
stündliche Fußgänger- und Temperaturwerte aus ``dataAllStreets.csv`` aggregiert.
Ohne diese Option werden keine Straßenmerkmale eingelesen.

Ausgaben und Optionen
~~~~~~~~~~~~~~~~~~~~~

Angereicherte Dateien erhalten den Suffix
``_clean_context_<Jahr>.parquet``. Die standardmäßig zusätzlich erzeugten
reduzierten Dateien heißen ``_model_ready_<Jahr>.parquet``. Beide werden mit
Snappy-Kompression geschrieben. Außerdem entstehen ``manifest_<Jahr>.json`` und
``manifest_model_ready_<Jahr>.json`` mit Laufstatistiken.

Verfügbare Optionen:

.. code-block:: text

   --input-dir PATH
   --additional-dir PATH
   --output-dir PATH
   --model-ready-dir PATH
   --no-model-ready
   --year JAHR
   --limit-files N
   --include-streets
   --overwrite

Ohne ``--overwrite`` werden bereits vorhandene Ausgaben übersprungen. Mit
``--no-model-ready`` wird die reduzierte ML-Ausgabe nicht geschrieben.

Beispiele:

.. code-block:: console

   python3 scripts/build_2025_enriched_parquets.py --limit-files 2
   python3 scripts/build_2025_enriched_parquets.py --include-streets --overwrite

Fahrpläne bereinigen
--------------------

``scripts/clean_timetables.py`` wandelt extrahierte CSV-Fahrplantabellen in ein
einheitliches Long- und Routenformat um.

.. code-block:: console

   python3 scripts/clean_timetables.py

Die Standardpfade lauten:

.. code-block:: text

   Eingabe:  downloads/wvv-pjs-2026/Fahrplaene_neu
   Ausgabe:  downloads/wvv-pjs-2026/Fahrplaene_clean

Eigene Verzeichnisse können über ``--raw-dir`` und ``--clean-dir`` angegeben
werden. Das Skript sucht rekursiv nach CSV-Dateien und ignoriert
``conversion_log.csv``.

Aus Datei- und Tabelleninhalt werden unter anderem Linie, Gültigkeitsdatum,
Tabellennummer und Betriebstag erkannt. Die unterstützten Betriebstagsklassen
sind ``weekday``, ``saturday``, ``sunday_holiday`` und als Fallback ``unknown``.
Haltestellen, die Marker ``ab`` und ``an`` sowie alle Zeitangaben werden in
separate Spalten überführt. Jede erkannte Fahrt erhält einen ``trip_key`` und
jede Uhrzeit zusätzlich einen Minutenwert seit Tagesbeginn.

Das Skript schreibt:

* ``fahrplan_long.csv`` und, sofern Parquet-Schreiben möglich ist,
  ``fahrplan_long.parquet``
* ``fahrplan_routes.csv`` und, sofern Parquet-Schreiben möglich ist,
  ``fahrplan_routes.parquet``
* ``fahrplan_clean_summary.json``

Die CSV-Dateien werden immer geschrieben. Schlägt ausschließlich die
Parquet-Ausgabe fehl, läuft das Skript mit den CSV-Ausgaben weiter. Das Dashboard
bevorzugt Parquet und fällt auf CSV zurück.

Prediction-Modelle trainieren
-----------------------------

``scripts/train_prediction_models.py`` trainiert und speichert das
Random-Forest-/Graph-Light-Ensemble zeilenweise pro Linie außerhalb der GUI.
Ohne ``--lines`` werden die Linien aus den Dateinamen im angereicherten
2025-Verzeichnis ermittelt.

Training starten:

.. code-block:: console

   python3 scripts/train_prediction_models.py run
   python3 scripts/train_prediction_models.py run --lines 10 20 27

Ohne ``--force`` werden zuerst Linien ohne gespeichertes Modell verarbeitet.
Bereits vorhandene Modelle werden danach standardmäßig inkrementell erweitert.
``--additional-trees`` legt die Zahl der zusätzlichen Random-Forest-Bäume fest;
der Standard ist ``30``.

.. list-table:: Trainingsoptionen
   :header-rows: 1

   * - Option
     - Verhalten
   * - ``--lines ...``
     - Linien als einzelne Werte oder kommaseparierte Liste
   * - ``--force``
     - gespeicherte Modelle ignorieren und vollständig neu trainieren
   * - ``--skip-existing``
     - vorhandene Modelle überspringen
   * - ``--additional-trees N``
     - Baumzahl je inkrementellem Lauf

Status und kontrollierter Stopp:

.. code-block:: console

   python3 scripts/train_prediction_models.py status
   python3 scripts/train_prediction_models.py stop

Ein Stopp wird nach der gerade laufenden Linie wirksam. Das Skript verwaltet in
``prediction/outputs``:

* ``train_prediction_models.pid``
* ``train_prediction_models_progress.json``
* ``train_prediction_models.log``
* ``train_prediction_models.stop`` während einer Stop-Anforderung

Die Modelle werden als
``prediction/models/wvv_prediction_lines_<Linie>.pkl`` gespeichert. Weitere
Metriken und Ausgaben des Service sind unter :doc:`predicition` beschrieben.

Gradient Boosting trainieren
----------------------------

``scripts/train_timetable_boosting.py`` trainiert den optionalen
Gradient-Boosting-Optimierer für stündliche Fahrtenzahlen. Dieses Modell ist vom
regelbasierten Mechanismus getrennt, der aktuell die konkrete GUI-
Fahrplantabelle erzeugt.

Ein einzelner Lauf mit Standardkosten:

.. code-block:: console

   python3 scripts/train_timetable_boosting.py run

Ausgewählte Linien und mehrere Iterationen:

.. code-block:: console

   python3 scripts/train_timetable_boosting.py run --lines 10 20 27 --iterations 3

Optionen:

.. code-block:: text

   --lines ...       Ohne Angabe alle erkannten enriched-2025-Linien
   --cost EUR        Kosten pro Busstunde; Standard 230
   --iterations N    Zahl der Trainingsiterationen; Standard 1
   --estimators N    neue Trees pro Iteration; Standard 40
   --sleep SEKUNDEN  Pause zwischen Iterationen
   --continuous      bis Ctrl+C oder Stop-Datei weitertrainieren
   --reset           vorhandenes Modell beim ersten Lauf ignorieren

Status und Stopp:

.. code-block:: console

   python3 scripts/train_timetable_boosting.py status
   python3 scripts/train_timetable_boosting.py stop

Das Skript schreibt Fortschritt, PID, Log und Stop-Datei mit dem Präfix
``train_timetable_boosting`` nach ``prediction/outputs``. Modell und Metriken
heißen:

.. code-block:: text

   prediction/models/wvv_timetable_gradient_boosting.pkl
   prediction/outputs/timetable_gradient_boosting_metrics.json

Beim regulären Ende wird der Fortschrittsstatus als ``stopped`` geschrieben,
auch wenn die konfigurierte Zahl der Iterationen vollständig erreicht wurde.

Ausführungsreihenfolge
----------------------

Für eine neue Datenbasis ist die durch den Code vorgegebene Abhängigkeit:

#. ``build_2025_enriched_parquets.py`` für Trainings- und Kontextdaten
#. ``clean_timetables.py`` für den WVV-Fahrplanvergleich
#. ``train_prediction_models.py`` für Nachfrageprognosen
#. optional ``train_timetable_boosting.py`` für den separaten lernenden
   Fahrtenzahl-Optimierer

Die ersten beiden Schritte sind voneinander unabhängig. Das Prediction-Training
benötigt angereicherte Dateien oder ein funktionsfähiges Dashboard-Repository;
das Boosting-Skript ermittelt seine Standardlinien ausschließlich aus den
angereicherten 2025-Dateinamen.
