from __future__ import annotations

import re
import unicodedata


def normalize_station_name(value: object) -> str:
    """
    Normalisiert Haltestellennamen zu einem robusten Vergleichsschlüssel.

    Die Funktion reduziert Schreibvarianten, Umlaute, Sonderzeichen und
    numerische Suffixe auf eine kompakte ASCII-Darstellung. Dadurch können
    Haltestellen in unterschiedlichen Datenquellen zuverlässig zusammengeführt
    werden, auch wenn die Originalschreibweise leicht abweicht.

    Parameter:
        value (object): Ursprünglicher Haltestellenname oder ein beliebiger
            Wert, der in eine Zeichenkette überführt werden kann.

    Rückgabewerte:
        str: Normalisierter Schlüssel ohne Leer- und Sonderzeichen.

    Fehler/Sonderfälle:
        None wird als leerer String behandelt. Nicht darstellbare Unicode-
        Zeichen werden entfernt. Unterschiedliche Schreibweisen wie „Straße“,
        „Strasse“ oder „Str.“ werden bewusst vereinheitlicht.

    Projektkontext:
        Die Normalisierung ist zentral für die Verknüpfung von Fahrgastdaten,
        Fahrplandaten und Graph-Nachbarschaften im Graph-Light-Modell. Ohne
        einen stabilen Schlüssel wären historische Profile und Haltestellen-
        Beziehungen nur eingeschränkt belastbar.
    """
    text = "" if value is None else str(value)
    text = re.sub(r"[_\s-]+\d+$", "", text.strip())
    text = text.replace("\u00df", "ss").replace("\u1e9e", "SS")
    text = text.replace("\u00c3\u0178", "ss").replace("\u00e1\u00ba\u017e", "SS")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = text.replace("strasse", "str").replace("str.", "str")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text