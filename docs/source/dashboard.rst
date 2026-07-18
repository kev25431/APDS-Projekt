Dashboard und adaptive Fahrplanerstellung
=========================================

Das Paket ``wvv_dashboard`` enthält die grafische Anwendung, das Laden der
Dashboard- und Fahrplandaten sowie die Logik für Fahrplanvergleiche und KPIs.
Die GUI verbindet die Nachfrageprognose aus :doc:`predicition` mit einem
bereinigten WVV-Fahrplan.

Die konkrete adaptive Fahrplantabelle wird regelbasiert erzeugt. Maßgeblich sind
``wvv_dashboard/service_policy.py`` und ``wvv_dashboard/timetable.py``. Das
separat trainierbare Gradient Boosting aus ``prediction/boosting.py`` ist nicht
der zentrale Mechanismus dieser Tabelle.

Aufbau der Anwendung
--------------------

``wvv_dashboard/app.py`` stellt unter anderem Ansichten für Dashboard, Karte,
Prediction, Fahrplan und KPI-Vergleich bereit. Für die hier beschriebene Kette
sind vor allem folgende Schritte relevant:

#. Ein Modell für die ausgewählte Linie wird geladen oder trainiert.
#. ``DemandPredictionService.predict_short_term`` berechnet die Nachfrage und
   eine stündliche Schedule-Ausgabe.
#. ``TimetableRepository`` lädt den passenden WVV-Fahrplan.
#. Die tatsächlichen WVV-Fahrten je Stunde werden mit der Prognose verbunden.
#. ``constrained_adaptive_runs`` bestimmt unter Einhaltung der Service-Policy
   die adaptive Fahrtenzahl.
#. Aus realen oder synthetisch ergänzten Abfahrten wird eine Vergleichstabelle
   aufgebaut.
#. ``kpis.py`` bewertet WVV-Bestand und adaptiven Vorschlag.

Fahrplandaten
-------------

``TimetableRepository`` sucht standardmäßig in
``downloads/wvv-pjs-2026/Fahrplaene_clean``. Bevorzugt werden

.. code-block:: text

   fahrplan_long.parquet
   fahrplan_routes.parquet

Sind die Parquet-Dateien nicht vorhanden, werden die gleichnamigen CSV-Dateien
verwendet. Die Erstellung dieser Dateien ist unter :doc:`scripts` beschrieben.

Auswahl des Betriebstags
~~~~~~~~~~~~~~~~~~~~~~~~

Für ein Datum wird einer von drei Service-Schlüsseln gebildet:

.. list-table:: Service-Schlüssel
   :header-rows: 1

   * - Bedingung
     - Schlüssel
   * - Montag bis Freitag
     - ``weekday``
   * - Samstag
     - ``saturday``
   * - Sonntag oder Feiertag
     - ``sunday_holiday``

Feiertage werden aus
``downloads/wvv-pjs-2026/Additional Data/bavarian_public_holidays_daily.csv``
geladen. Ein Datum mit einem positiven Wert in ``public_holiday`` wird wie ein
Sonntag behandelt. Fehlt diese Datei oder lässt sie sich nicht lesen, bleibt die
Feiertagsmenge leer.

Auswahl des Fahrplans
~~~~~~~~~~~~~~~~~~~~~

Zunächst sucht das Repository einen Fahrplan mit dem ermittelten
Service-Schlüssel. Sind keine Daten vorhanden, werden in dieser Reihenfolge auch
``unknown``, ``weekday``, ``saturday`` und ``sunday_holiday`` geprüft; doppelte
Einträge in dieser Suchliste werden entfernt.

Innerhalb des gewählten Typs werden Tabellen bevorzugt, die den angeforderten
Zeitraum abdecken. Die weitere Sortierung erfolgt nach:

#. neuestem ``effective_date``
#. Zahl unterschiedlicher Haltestellen
#. Zahl unterschiedlicher Fahrten
#. Zahl der Datensätze

Tabellen desselben Plans mit identischer Haltestellenfolge werden
zusammengeführt. Dadurch kann ein über mehrere CSV-Tabellen verteilter Tagesplan
gemeinsam verarbeitet werden.

Originaler WVV-Fahrplan
-----------------------

Für jede ``trip_key`` wird die früheste Minute als Abfahrtszeit der Fahrt
bestimmt. Liegen mehrere Fahrten auf derselben gerundeten Startminute, bleibt
diejenige mit den meisten Haltestellendatensätzen erhalten. Anschließend wird
auf den gewählten Stundenhorizont und auf höchstens ``max_trips`` begrenzt;
``build_comparison`` verwendet standardmäßig ``160``.

Die typische Fahrzeit von der ersten bis zu jeder weiteren Haltestelle wird aus
den vorhandenen Fahrten abgeleitet:

.. code-block:: text

   offset = minutes_at_station - first_minute

Der Median dieser Offsets je Haltestelle wird verwendet. Die Offsets werden
entlang der Route nicht fallend gemacht. Die WVV-Matrix selbst zeigt die in den
Quelldaten enthaltenen Zeiten je Fahrt und Haltestelle.

Service-Policy
--------------

``service_policy_for_line`` ordnet Linien einer von vier Gruppen zu:

.. list-table:: Liniengruppen und Grenzwerte
   :header-rows: 1
   :widths: 18 20 8 11 11 11 11 10

   * - Gruppe
     - Linien
     - Mindestfahrten
     - Takt normal
     - Takt Peak
     - Bestandsanteil
     - Zielauslastung
     - Umlaufzeit
   * - Uni-/Hauptkorridor
     - 10, 14, 114, 214
     - 3
     - 20 min
     - 15 min
     - 0,62
     - 0,68
     - 58 min
   * - Stadt-Hauptlinie
     - 4, 5, 6, 16, 20, 26
     - 2
     - 30 min
     - 20 min
     - 0,55
     - 0,72
     - 52 min
   * - Pendler-/Grundversorgung
     - 8, 13, 21, 27, 29, 33, 54, 55
     - 1
     - 60 min
     - 30 min
     - 0,45
     - 0,76
     - 46 min
   * - Standardlinie
     - alle übrigen Linien
     - 1
     - 60 min
     - 30 min
     - 0,42
     - 0,76
     - 45 min

Weitere Policy-Werte sind:

.. list-table:: Zusätzliche Policy-Grenzen
   :header-rows: 1

   * - Gruppe
     - Produktive Untergrenze
     - Maximale Zusatzfahrten
     - Maximale neue Fahrten
   * - Uni-/Hauptkorridor
     - 0,34
     - 4
     - 3
   * - Stadt-Hauptlinie
     - 0,34
     - 3
     - 2
   * - Pendler-/Grundversorgung
     - 0,32
     - 3
     - 2
   * - Standardlinie
     - 0,30
     - 2
     - 2

Als zeitliche Peak Hours gelten ``06:00`` bis ``09:59`` und ``15:00`` bis
``18:59``. Unabhängig davon wird eine Stunde als Peak behandelt, wenn

.. code-block:: text

   demand >= baseline * comfort_load * 0.82

Berechnung der adaptiven Fahrtenzahl
------------------------------------

``constrained_adaptive_runs`` erhält Nachfrage, vorhandene Fahrten,
Fahrzeugkapazität, Kosten, Linie und Stunde. Die Kapazität wird auf mindestens
``35`` gesetzt. Der Kostenfaktor beeinflusst Zielauslastung und produktive
Untergrenze:

.. code-block:: text

   cost_pressure = clamp(cost / default_cost, 0.65, 1.65)
   target_utilization = clamp(policy.target_utilization
                              + (cost_pressure - 1) * 0.04,
                              0.60, 0.84)
   productive_floor = clamp(policy.productive_floor
                            + (cost_pressure - 1) * 0.05,
                            0.25, 0.48)

.. code-block:: text

   comfort_load    = capacity * target_utilization
   productive_load = capacity * productive_floor

Im Fahrplanvergleich ruft ``timetable.py`` diese Funktion mit den
Standardkosten von ``230 EUR`` pro Busstunde auf. Für die Nachfrage wird
``predicted_boardings`` verwendet. Fehlt dieser Wert, dienen ``55 %`` von
``predicted_demand`` als Fallback. Fehlt eine Prognosezeile vollständig, werden
die vorhandenen WVV-Fahrten unverändert übernommen.

Stunden ohne vorhandene WVV-Fahrt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Neue Fahrten sind nur erlaubt, wenn die Stunde im vorhandenen Betriebszeitraum
liegt oder in dieser Stunde bereits eine WVV-Fahrt existiert. Bei erlaubtem
neuem Angebot und positiver Nachfrage gilt:

.. code-block:: text

   required = ceil(demand / comfort_load)
   adaptive_runs = min(policy.max_new_runs, required)

Andernfalls bleibt die Fahrtenzahl ``0``.

Stunden mit vorhandenem Angebot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Die gerundete vorhandene Fahrtenzahl heißt ``baseline``. Aus maximaler
Taktweite und Mindestanteil am Bestand entstehen Untergrenzen:

.. code-block:: text

   headway_floor  = min(ceil(60 / max(5, headway)), baseline)
   baseline_floor = ceil(baseline * min_baseline_share)
   min_service = max(min_runs_when_active, headway_floor, baseline_floor)
   min_service = clamp(min_service, 1, baseline)

Bei einer Nachfrage von ``0`` wird ``min_service`` zurückgegeben. Sonst gilt:

.. code-block:: text

   required_for_comfort = ceil(demand / comfort_load)
   target = max(min_service, required_for_comfort)

Solange ``target`` über dem Mindestservice liegt und die Nachfrage je Fahrt
kleiner als ``productive_load`` ist, wird ``target`` reduziert.

Zusatzfahrten werden anhand des Überlastungsverhältnisses begrenzt:

.. list-table:: Zulässige Zusatzfahrten
   :header-rows: 1

   * - ``overload_ratio``
     - Höchstens zusätzliche Fahrten
   * - bis einschließlich ``1.2``
     - ``1``
   * - bis einschließlich ``1.8``
     - ``2``
   * - größer als ``1.8``
     - ``policy.max_extra_runs``

Erzeugung konkreter Abfahrten
-----------------------------

Die berechnete Zielzahl wird in ``_adaptive_hour_departures`` zunächst auf den
Bereich ``0`` bis ``12`` Fahrten pro Stunde begrenzt.

* Entspricht die Zielzahl dem Bestand, werden die vorhandenen Fahrten kopiert.
* Ist die Zielzahl kleiner, werden vorhandene Fahrten möglichst gleichmäßig
  ausgewählt. Bei genau einer Zielfahrt bleibt die mittlere Fahrt erhalten.
* Ist die Zielzahl größer, bleiben vorhandene Fahrten erhalten und zusätzliche
  Kandidaten werden gleichmäßig in der Stunde verteilt.

Für synthetische Fahrten gilt:

.. code-block:: text

   spacing = 60 / target
   minute  = hour_start + spacing / 2 + index * spacing
   trip_key = "adaptive:<gerundete Minute>"

Bereits belegte gerundete Minuten werden übersprungen. Synthetische Fahrten
erhalten in der GUI ein ``*``. Ihre Zeiten an den weiteren Haltestellen ergeben
sich aus der Startminute plus dem Median-Offset des WVV-Fahrplans. Reihenfolge
und typische Fahrzeitstruktur der Linie werden somit übernommen.

KPI-Vergleich
-------------

``wvv_dashboard/kpis.py`` berechnet die Kennzahlen aus stündlicher Nachfrage,
WVV-Fahrten, adaptiven Fahrten und mittlerer Fahrzeugkapazität. Für die KPI-
Nachfrage werden bevorzugt ``predicted_boardings`` und ersatzweise
``predicted_demand`` verwendet. Vor der Berechnung wird die adaptive Fahrtenzahl
erneut mit ``constrained_adaptive_runs`` bestimmt.

Die festen KPI-Grenzen sind:

.. code-block:: text

   DEFAULT_BUS_HOURLY_COST_EUR = 230
   COMFORT_UTILIZATION_LIMIT = 0.85
   PRODUCTIVE_UTILIZATION_FLOOR = 0.35

Kapazität und Abdeckung
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   capacity = runs * avg_capacity
   served   = min(demand, capacity)
   coverage = sum(served) / sum(demand)
   unserved = max(0, demand - capacity)

Über- und Unterauslastung
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   comfort_capacity = capacity * 0.85
   overload = max(0, demand - comfort_capacity)

   productive_capacity = capacity * 0.35
   underload = max(0, productive_capacity - demand)

   utilization = sum(demand) / sum(capacity)

Für die Peak-Abdeckung wird die Stunde mit der höchsten Nachfrage ermittelt.
Ausgewertet werden diese Stunde sowie die direkt vorherige und nachfolgende
Stunde.

Wartezeit, Busstunden und Kosten
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Die angenommene mittlere Wartezeit beträgt bei vorhandenen Fahrten die Hälfte
der Taktzeit, sonst ``60`` Minuten:

.. code-block:: text

   wait = 60 / runs / 2   # runs > 0
   wait = 60              # runs = 0

Die Stundenwerte werden nach Nachfrage gewichtet. Die geschätzten Busstunden
nutzen die Umlaufzeit der jeweiligen Linien-Policy:

.. code-block:: text

   bus_hours = runs * cycle_minutes / 60
   cost      = bus_hours * bus_hourly_cost

Die GUI kann Tages-KPIs für eine Linie sowie Jahreswerte für 2025 aus den
angereicherten Parquet-Dateien berechnen. Der Bericht vergleicht unter anderem
Abdeckung, Peak-Abdeckung, Auslastung, Überlastung, Unterauslastung, nicht
bediente Nachfrage, Wartezeit, Fahrten, Busstunden und Kosten.

Modulübersicht
--------------

* ``wvv_dashboard/app.py``: GUI, Datenrepository, Ansichten und Hintergrundläufe
* ``wvv_dashboard/config.py``: Datenpfade, Kartenparameter, Farben und
  Laufzeiteinstellungen
* ``wvv_dashboard/timetable.py``: Laden, Auswählen und Vergleichen der Fahrpläne
* ``wvv_dashboard/service_policy.py``: linienabhängige Mindestversorgung und
  adaptive Fahrtenzahl
* ``wvv_dashboard/kpis.py``: Tages-, Linien- und Jahreskennzahlen
* ``wvv_dashboard/widgets.py``: wiederverwendbare GUI-Elemente
* ``wvv_dashboard/assets``: Bilddateien für die Anwendung

Einschränkungen
---------------

Der adaptive Plan ist ein Entscheidungsvorschlag und kein betriebsverbindlicher
Fahrplan. Er berücksichtigt keine vollständigen Umläufe, Fahrerpausen,
Depotfahrten, Störungen oder Fahrzeugverfügbarkeit. Die Route und die typischen
Fahrzeiten stammen aus dem WVV-Bestand; optimiert werden die Fahrtenzahl und die
Verteilung der Abfahrten innerhalb der Stunde.
