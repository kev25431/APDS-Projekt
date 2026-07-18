Dashboard
==========

Dieser Ordner enthält die GUI- und Dashboard-Logik für die WVV-Anwendung.

- ``Data_project_app/wvv_dashboard/__init__.py``: Paket-Initialisierung für das Dashboard.
- ``Data_project_app/wvv_dashboard/app.py``: Kernanwendung der GUI. Sie baut die Bedienoberfläche, lädt Daten, rendert Karten, Diagramme und Tabs für Dashboard, Kartenansicht, Prediction und Fahrplan.
- ``Data_project_app/wvv_dashboard/config.py``: Konfiguration für Pfade, Kartenparameter, Farbpalette und Referenzdaten.
- ``Data_project_app/wvv_dashboard/kpis.py``: Berechnet Kennzahlen für den Vergleich von WVV-Bestand und adaptivem Fahrplan. Dazu gehören Coverage-, Überlastungs-, Unterauslastungs- und Kostenmetriken.
- ``Data_project_app/wvv_dashboard/service_policy.py``: Definiert die Service-Policy für verschiedene Linienarten. Dazu gehören Mindestfahrten, maximale Taktzeiten, Zielauslastung und Regeln für Spitzenzeiten.
- ``Data_project_app/wvv_dashboard/timetable.py``: Verarbeitet Fahrplandaten, baut Vergleiche zwischen WVV-Fahrplan und adaptivem Fahrplan und erzeugt Tabellen für die Darstellung.
- ``Data_project_app/wvv_dashboard/widgets.py``: Kleine UI-Bausteine für Datumsauswahl und animierte Linienliste.
- ``Data_project_app/wvv_dashboard/assets/``: Ressourcenordner für Bilddateien wie das Busicon.