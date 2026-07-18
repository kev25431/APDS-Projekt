Übersicht
=========
Der Code ist in Python geschrieben und verwendet die Bibliotheken `numpy`, `pandas`, `matplotlib` und `scikit-learn`. 
Die Hauptfunktionen des Codes sind folgendes: 

...

Gestartet wird die Datei über ``gui_app.py`` 
Diese Datei enthält die grafische Benutzeroberfläche (GUI) und ermöglicht es dem Benutzer, verschiedene Funktionen des Codes auszuführen.

Die Anwendung startet im Vollbildmodus und nutzt eine moderne dunkle Dashboard-Oberfläche mit rotem Hauptakzent, Sidebar-Navigation und mehreren Funktionsbereichen.

Wichtige Reiter:

Dashboard
----------

* Zeitraum über Kalender auswählen
* Linien auswählen
* tägliche Nachfrage anzeigen
* flexible Analyse mit auswählbarer X- und Y-Achse
* Haltestellenfolge mit Ein- und Ausstiegen
* Smart Insights zu Nachfrage, Peaks und Mustern

Kartenansicht
--------------

* Karte auf Würzburg zentriert
* Haltestellenmarker aus GeoJSON-Daten
* Ein- und Ausstiege als Balken an Haltestellen
* Nachfrageanimation über den Tag
* Play/Pause und Geschwindigkeitsslider
* Busse als Symbol auf der Karte
* Event-Geodaten werden eingebunden, sofern vorhanden

Prediction
-----------
* Nachfrageprognose für 2026 auf Basis der aufbereiteten 2025-Daten
* Auswahl von Linie, Tag, Uhrzeit und Prognosemodus
* Random-Forest-Modell plus graphbasiertes GNN-Light-Modell
* automatische Gewichtung der Modelle über Validierungsfehler
* Ausgabe von Hotspots, erwarteter Nachfrage und adaptiven Fahrten

Fahrplan
----------

* Vergleich zwischen WVV-Bestandsfahrplan und adaptivem Fahrplan
* Route wird oben als Linienverlauf dargestellt
* darunter Tabellen für Originalfahrplan und adaptiven Fahrplan
* mehrere Fahrplan-CSV-Dateien werden pro Linie zusammengeführt
* adaptive Fahrten werden aus Nachfrage, Fahrplanstruktur und Policy-Regeln abgeleitet

KPI Vergleich
--------------

* Gesamt-KPIs über alle Linien und Jahresstandard 2025
* Linien-KPIs für ausgewählte Linie und ausgewählten Tag
* Kosten pro Busstunde editierbar, Standardwert 230 EUR
* Linienranking nach Einsparpotenzial, Überlastung und Unterauslastung
* neue Wartezeit-KPI als Servicequalitätsindikator


Struktur der Codebasis
----------------------
Die Codebasis ist in mehrere Module unterteilt, die jeweils spezifische Funktionen und Aufgaben erfüllen. 

``wvv_dashboard``
   Enthält Desktop-Oberfläche, Fahrplanvergleich, KPI-Berechnung und Regeln
   für das adaptive Verkehrsangebot. :mod:`wvv_dashboard.service_policy`
   kapselt die linienabhängigen Mindesttakte und Auslastungsziele;
   :mod:`wvv_dashboard.kpis` berechnet daraus Kosten- und Qualitätskennzahlen.

``prediction``
   Lädt Trainingsdaten, erzeugt Features und kombiniert Random Forest und ein
   leichtgewichtiges Graph-Nachbarschaftsmodell. Der zentrale Einstiegspunkt
   ist :class:`prediction.service.DemandPredictionService`.

``scripts``
   Enthält Skripte zum Training der Prognosemodelle und des optionalen
   Gradient-Boosting-Optimizers für Fahrtenzahlen.

Aufgabenteilung
-----------------
Die Aufgaben innerhalb des Projekts sind wie folgt verteilt:
