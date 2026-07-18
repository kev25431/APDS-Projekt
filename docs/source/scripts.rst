Scripts
========

Dieser Ordner enthält Trainings- und Aufbereitungsskripte.

- ``Data_project_app/scripts/build_2025_enriched_parquets.py``: Baut aus Roh-Parquet-Daten des Jahres 2025 bereinigte und angereicherte Parquet-Dateien. Dazu gehören Datums- und Stunden-Features, Kontextdaten wie Feiertage, Vorlesungen, Events und Straßeninformationen sowie eine reduzierte Modell-Ready-Ausgabe.
- ``Data_project_app/scripts/clean_timetables.py``: Bereinigt Roh-Fahrplanauszüge aus den CSV-Dateien im Timetable-Ordner. Es extrahiert Haltestellen, Marker, Zeiten und Dienstags-Definitionen und schreibt saubere Long- und Route-Dateien.
- ``Data_project_app/scripts/train_prediction_models.py``: Trainiert die Vorhersagemodelle für ausgewählte Linien. Es nutzt die bereinigten 2025-Daten, schreibt Fortschritt und Logs und speichert die trainierten Modelle im Modellordner.
- ``Data_project_app/scripts/train_timetable_boosting.py``: Trainiert den Gradient-Boosting-Optimierer für Fahrplan-Run-Anzahlen. Es schreibt Fortschritt, Logs und Metriken für mehrere Iterationen.