from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd


@dataclass
class PredictionResult:
    predictions: pd.DataFrame
    metrics: dict[str, float]
    weights: dict[str, float]
    message: str
    schedule: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class TrainingData:
    raw: pd.DataFrame
    source: str
    date_span: tuple[date, date] | None
