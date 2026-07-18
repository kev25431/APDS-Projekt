from __future__ import annotations

import re
import unicodedata


def normalize_station_name(value: object) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"[_\s-]+\d+$", "", text.strip())
    text = text.replace("\u00df", "ss").replace("\u1e9e", "SS")
    text = text.replace("\u00c3\u0178", "ss").replace("\u00e1\u00ba\u017e", "SS")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = text.replace("strasse", "str").replace("str.", "str")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text
