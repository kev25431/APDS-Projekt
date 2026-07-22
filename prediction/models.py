from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import normalize_station_name


class GroupFallbackRegressor:
    """
    Einfache Fallback-Regressionslogik ohne scikit-learn.

    Dieses Modell wird genutzt, wenn der eigentliche Random-Forest-Regressor
    nicht verfügbar ist. Statt eines lernenden Baummodells werden historische
    Mittelwerte auf mehreren Aggregationsebenen verwendet.
    
    Rückgabewerte:
        GroupFallbackRegressor: Initialisierte Modellinstanz.

    Fehler/Sonderfälle:
        Das Modell benötigt keine externen ML-Abhängigkeiten. Bei fehlenden
        Detailprofilen wird schrittweise auf gröbere Gruppen und schließlich
        auf globale Mittelwerte zurückgegriffen.

    Projektkontext:
        Die Klasse stellt sicher, dass das Gesamtsystem auch in reduzierten
        Laufzeitumgebungen lauffähig bleibt. Sie dient damit als robuster
        technischer Rückfallmechanismus für die Nachfrageprognose.
    """

    def __init__(self) -> None:
        """
        Initialisiert die internen Mittelwerttabellen des Fallback-Modells.

        Es werden leere Profile für haltestellen- und stundenspezifische
        Aggregationen sowie ein globaler Zweiziel-Mittelwert für Ein- und
        Aussteiger angelegt.

        Parameter:
            Keine.

        Rückgabewerte:
            None: Die Methode initialisiert ausschließlich Objektzustand.

        Fehler/Sonderfälle:
            Keine besondere Fehlerbehandlung erforderlich.

        Projektkontext:
            Die vorbereiteten Strukturen werden später mit historischen
            Mittelwerten befüllt und ersetzen im Notfall das Random-Forest-
            Modell innerhalb des Ensembles.
        """
        self.by_station_hour = pd.DataFrame()
        self.by_station = pd.DataFrame()
        self.global_mean = np.array([0.0, 0.0], dtype=float)

    def fit(self, frame: pd.DataFrame) -> None:
        """
        Lernt historische Mittelwerte für mehrere Granularitäten.

        Zunächst werden Mittelwerte je Linie, Haltestelle, Stunde und Wochentag
        gebildet. Zusätzlich wird ein gröberes Haltestellenprofil ohne Zeitbezug
        sowie ein globaler Durchschnitt über beide Zielgrößen erzeugt.

        Parameter:
            frame (pd.DataFrame): Trainingsdaten mit den Zielspalten
                ``boardings`` und ``exiting`` sowie den Gruppierungsmerkmalen.

        Rückgabewerte:
            None: Die gelernten Profile werden im Objekt gespeichert.

        Fehler/Sonderfälle:
            Leere oder unvollständige Eingaben werden nicht gesondert geprüft;
            sie würden zu leeren Profilen oder NaN-freien globalen Mittelwerten
            nach ``fillna(0)`` führen.

        Projektkontext:
            Die Methode bildet die fachliche Minimalvariante der
            Nachfrageprognose: historische Durchschnittsnachfrage pro Ort und
            Zeitfenster statt eines datengetriebenen Ensemble-Modells.
        """
        self.by_station_hour = (
            frame.groupby(["line", "station_key", "hour", "weekday"], as_index=False)
            .agg(boardings=("boardings", "mean"), exiting=("exiting", "mean"))
        )

        self.by_station = (
            frame.groupby(["line", "station_key"], as_index=False)
            .agg(boardings=("boardings", "mean"), exiting=("exiting", "mean"))
        )

        self.global_mean = frame[["boardings", "exiting"]].mean().fillna(0).to_numpy(dtype=float)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """
        Erzeugt Vorhersagen aus den zuvor gelernten Gruppenprofilen.

        Die Methode verwendet zunächst das feinste Profil je Linie, Haltestelle,
        Stunde und Wochentag. Fehlen dort Werte, wird auf das gröbere
        Haltestellenprofil zurückgegriffen. Als letzte Sicherung dient der
        globale Mittelwert.

        Parameter:
            frame (pd.DataFrame): Eingabedaten mit den Merkmalen ``line``,
                ``station_key``, ``hour`` und ``weekday``.

        Rückgabewerte:
            np.ndarray: Zweidimensionales Array mit vorhergesagten Ein- und
            Aussteigerwerten.

        Fehler/Sonderfälle:
            Bei leerem Eingabe-Frame wird ein leeres Array mit zwei Spalten
            zurückgegeben. Fehlende Profile werden automatisch stufenweise
            ersetzt.

        Projektkontext:
            Diese Methode hält die Vorhersagepipeline auch dann funktionsfähig,
            wenn keine vollständige ML-Umgebung verfügbar ist. Sie unterstützt
            damit Testbarkeit, Portabilität und betriebliche Robustheit.
        """
        if frame.empty:
            return np.empty((0, 2), dtype=float)
        lookup = self.by_station_hour.rename(columns={"boardings": "pred_boardings", "exiting": "pred_exiting"})
        merged = frame[["line", "station_key", "hour", "weekday"]].merge(
            lookup,
            on=["line", "station_key", "hour", "weekday"],
            how="left",
        )

        missing = merged["pred_boardings"].isna()
        if missing.any():
            station_lookup = self.by_station.rename(columns={"boardings": "pred_boardings", "exiting": "pred_exiting"})
            fallback = frame.loc[missing, ["line", "station_key"]].merge(
                station_lookup,
                on=["line", "station_key"],
                how="left",
            )
            merged.loc[missing, "pred_boardings"] = fallback["pred_boardings"].fillna(self.global_mean[0]).to_numpy()
            merged.loc[missing, "pred_exiting"] = fallback["pred_exiting"].fillna(self.global_mean[1]).to_numpy()
        merged["pred_boardings"] = merged["pred_boardings"].fillna(self.global_mean[0])
        merged["pred_exiting"] = merged["pred_exiting"].fillna(self.global_mean[1])
        return merged[["pred_boardings", "pred_exiting"]].to_numpy(dtype=float)


class GraphMessagePassingRegressor:
    """
    Leichtgewichtiges Graph-Modell zur Nachfrageprognose pro Haltestelle.

    Anders als ein tiefes neuronales GNN arbeitet diese Implementierung mit
    historischen Haltestellenprofilen und einer einfachen Nachbarschaftslogik.
    Für jede Haltestelle werden eigene historische Muster und Werte benachbarter
    Haltestellen entlang realer Fahrten kombiniert.

    Parameter:
        Keine.

    Rückgabewerte:
        GraphMessagePassingRegressor: Initialisierte Modellinstanz.

    Fehler/Sonderfälle:
        Wenn keine Nachbarschaften oder feinen Profile vorliegen, greift das
        Modell automatisch auf gröbere historische Ebenen bis zum globalen
        Mittelwert zurück.

    Projektkontext:
        Das Modell bildet laut Projektübersicht den „Graph-Light“-Anteil des
        Ensembles. Es ergänzt den Random Forest um Strukturwissen aus der
        Reihenfolge real benachbarter Haltestellen.
    """

    def __init__(self) -> None:
        """
        Initialisiert Profil- und Nachbarschaftsspeicher des Graph-Modells.

        Angelegt werden Datenstrukturen für exakte, stundenbezogene und reine
        Haltestellenprofile sowie ein globaler Mittelwert für beide Zielgrößen.

        Parameter:
            Keine.

        Rückgabewerte:
            None: Die Methode initialisiert ausschließlich internen Zustand.

        Fehler/Sonderfälle:
            Keine besondere Fehlerbehandlung erforderlich.

        Projektkontext:
            Diese Strukturen werden beim Training mit historischen Nachfrage-
            und Graphinformationen befüllt und später für die Ensemble-Prognose
            verwendet.
        """
        self.neighbors: dict[tuple[int, str], set[str]] = {}
        self.profile_exact = pd.DataFrame()
        self.profile_hour = pd.DataFrame()
        self.profile_station = pd.DataFrame()
        self.global_mean = np.array([0.0, 0.0], dtype=float)

    def fit(self, feature_frame: pd.DataFrame, raw_frame: pd.DataFrame) -> None:
        """
        Lernt historische Profile und den Haltestellen-Nachbarschaftsgraphen.

        Aus dem aggregierten Feature-Frame werden Mittelwerte auf drei Ebenen
        gebildet: exaktes Profil je Linie/Haltestelle/Stunde/Wochentag,
        Stundenprofil je Linie/Haltestelle/Stunde und ein allgemeines
        Haltestellenprofil je Linie/Haltestelle. Aus den Rohfahrtdaten wird
        zusätzlich ein ungerichteter Nachbarschaftsgraph entlang realer
        Haltestellenfolgen konstruiert.

        Parameter:
            feature_frame (pd.DataFrame): Aggregierte Trainingsdaten mit
                Zielgrößen und Merkmalen auf Haltestellen-Stunden-Ebene.
            raw_frame (pd.DataFrame): Rohdaten einzelner Stop-Ereignisse, aus
                denen echte Nachbarschaften entlang von Fahrten abgeleitet werden.

        Rückgabewerte:
            None: Das trainierte Profilwissen wird im Objekt gespeichert.

        Fehler/Sonderfälle:
            Fehlende oder unvollständige Rohdaten führen zu einem reduzierten
            oder leeren Graphen. Die Vorhersage bleibt dennoch möglich, da
            historische Profile und globale Mittelwerte als Fallback dienen.

        Projektkontext:
            Diese Methode implementiert die fachliche Idee, dass Haltestellen
            nicht isoliert betrachtet werden, sondern entlang einer Linie durch
            reale Nachbarschaften miteinander verbunden sind.
        """
        self.profile_exact = (
            feature_frame.groupby(["line", "station_key", "hour", "weekday"], as_index=False)
            .agg(boardings=("boardings", "mean"), exiting=("exiting", "mean"))
        )

        self.profile_hour = (
            feature_frame.groupby(["line", "station_key", "hour"], as_index=False)
            .agg(boardings=("boardings", "mean"), exiting=("exiting", "mean"))
        )

        self.profile_station = (
            feature_frame.groupby(["line", "station_key"], as_index=False)
            .agg(boardings=("boardings", "mean"), exiting=("exiting", "mean"))
        )

        self.global_mean = feature_frame[["boardings", "exiting"]].mean().fillna(0).to_numpy(dtype=float)

        raw = raw_frame.sort_values(["line", "journey", "departure_plan_station"]).copy()
        raw["station_key"] = raw["station"].map(normalize_station_name)
        for (line, _journey), group in raw.groupby(["line", "journey"], sort=False):
            keys = [key for key in group["station_key"].tolist() if key]
            for left, right in zip(keys, keys[1:]):
                self.neighbors.setdefault((int(line), left), set()).add(right)
                self.neighbors.setdefault((int(line), right), set()).add(left)

    def _lookup(self, line: int, station_key: str, hour: int, weekday: int) -> np.ndarray:
        """
        Sucht das passendste historische Nachfrageprofil für eine Haltestelle.

        Die Suchreihenfolge folgt einer abgestuften Fallback-Logik: zunächst
        exaktes Profil mit Stunde und Wochentag, dann nur Stunde, dann das
        allgemeine Haltestellenprofil und zuletzt der globale Mittelwert.

        Parameter:
            line (int): Liniennummer.
            station_key (str): Normalisierter Haltestellenschlüssel.
            hour (int): Stunde des Prognosezeitpunkts.
            weekday (int): Wochentag als numerischer Index.

        Rückgabewerte:
            np.ndarray: Zweielementiges Array für Ein- und Aussteiger.

        Fehler/Sonderfälle:
            Falls kein passendes Profil vorliegt, wird immer ein numerisch
            gültiger globaler Mittelwert zurückgegeben.

        Projektkontext:
            Die Methode bildet das historische Grundsignal des Graph-Light-
            Modells und liefert sowohl den Eigenwert einer Haltestelle als auch
            die Werte benachbarter Knoten.
        """
        row = self.profile_exact[
            (self.profile_exact["line"] == line)
            & (self.profile_exact["station_key"] == station_key)
            & (self.profile_exact["hour"] == hour)
            & (self.profile_exact["weekday"] == weekday)
        ]
        if not row.empty:
            return row.iloc[0][["boardings", "exiting"]].to_numpy(dtype=float)

        row = self.profile_hour[
            (self.profile_hour["line"] == line)
            & (self.profile_hour["station_key"] == station_key)
            & (self.profile_hour["hour"] == hour)
        ]
        if not row.empty:
            return row.iloc[0][["boardings", "exiting"]].to_numpy(dtype=float)

        row = self.profile_station[
            (self.profile_station["line"] == line)
            & (self.profile_station["station_key"] == station_key)
        ]
        if not row.empty:
            return row.iloc[0][["boardings", "exiting"]].to_numpy(dtype=float)
        return self.global_mean.copy()

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """
        Erzeugt Vorhersagen durch Kombination von Eigen- und Nachbarprofilen.

        Für jede Zeile wird zunächst das eigene historische Profil der
        Haltestelle bestimmt. Anschließend werden – sofern vorhanden –
        Vorhersagewerte benachbarter Haltestellen derselben Linie einbezogen.
        Die Kombination folgt der im Projekt beschriebenen Gewichtung von
        0.72 für den Eigenwert und 0.28 für den Mittelwert der Nachbarn.

        Parameter:
            frame (pd.DataFrame): Eingabedaten auf Haltestellen-Stunden-Ebene.

        Rückgabewerte:
            np.ndarray: Array mit prognostizierten Ein- und Aussteigerwerten.

        Fehler/Sonderfälle:
            Bei leerem Eingabe-Frame wird ein leeres Array zurückgegeben.
            Existieren keine Nachbarn, wird ausschließlich das Eigenprofil
            verwendet.

        Projektkontext:
            Diese Methode ist der graphbasierte Anteil des Ensemble-Modells und
            soll räumlich-strukturelle Zusammenhänge entlang einer Linie
            berücksichtigen, die ein rein tabellarisches Modell nur begrenzt
            erfassen kann.
        """
        rows: list[np.ndarray] = []
        for _, row in frame.iterrows():
            line = int(row["line"])
            station_key = str(row["station_key"])
            hour = int(row["hour"])
            weekday = int(row["weekday"])
            own = self._lookup(line, station_key, hour, weekday)
            neighbor_values = [
                self._lookup(line, neighbor, hour, weekday)
                for neighbor in sorted(self.neighbors.get((line, station_key), set()))
            ]
            if neighbor_values:
                rows.append(0.72 * own + 0.28 * np.vstack(neighbor_values).mean(axis=0))
            else:
                rows.append(own)
        return np.vstack(rows) if rows else np.empty((0, 2), dtype=float)