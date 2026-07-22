from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("ARROW_USER_SIMD_LEVEL", "NONE")

import numpy as np
import pandas as pd


MODEL_READY_PATTERN = "*_model_ready_2025.parquet"
DEFAULT_SPLIT_DATE = "2025-10-01"

CONTEXT_FEATURE_COLUMNS = [
    "hour",
    "weekday",
    "month",
    "day_of_year",
    "calendar_week",
    "is_weekend",
    "line_code",
    "direction_code",
    "vehicle_type_code",
    "public_holiday",
    "school_holiday",
    "lecture_period_jmu",
    "lecture_period_thws",
    "event_day",
    "event_count",
    "concert_day",
    "concert_event_count",
    "verkaufsoffener_sonntag",
]

STATIC_NODE_FEATURE_COLUMNS = [
    "mean_boarding",
    "mean_exiting",
    "mean_stop_sequence",
    "line_count",
    "direction_count",
]


@dataclass(frozen=True)
class TransitGraphDataset:
    """Graph-ready tensors and train/test tables for station-level demand forecasting."""

    station_to_node: dict[str, int]
    node_to_station: dict[int, str]
    edge_index: np.ndarray
    node_features: np.ndarray
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    context_feature_columns: list[str]
    node_feature_columns: list[str]
    encoders: dict[str, dict[str, int]]
    defaults: dict[str, float]

    @property
    def num_nodes(self) -> int:
        return len(self.station_to_node)

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1]) if self.edge_index.ndim == 2 else 0


def load_model_ready_data(
    data_dir: Path | str,
    lines: Iterable[int] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load model-ready parquet files, optionally restricted to selected lines."""

    data_path = Path(data_dir)
    line_set = set(lines) if lines is not None else None
    frames = []

    for parquet_path in sorted(data_path.glob(MODEL_READY_PATTERN)):
        frame = pd.read_parquet(parquet_path, columns=columns)
        if line_set is not None:
            frame = frame[frame["line"].isin(line_set)]
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No model-ready parquet files found in {data_path}")

    return pd.concat(frames, ignore_index=True)


def build_transit_graph_dataset(
    data_dir: Path | str,
    lines: Iterable[int] | None = None,
    split_date: str = DEFAULT_SPLIT_DATE,
) -> TransitGraphDataset:
    """Build graph topology, node features, and supervised station-hour rows."""

    df = load_model_ready_data(Path(data_dir), lines=lines)
    df = _normalize_input_frame(df)
    station_to_node = _station_mapping(df)
    encoders = _fit_encoders(df)
    graph_frame = _encode_context_features(df, encoders)
    edge_index = _build_edge_index(graph_frame, station_to_node)
    node_features = _build_node_features(graph_frame, station_to_node)
    supervised = _build_supervised_rows(graph_frame, station_to_node)

    split_timestamp = pd.to_datetime(split_date)
    train_frame = supervised[supervised["date"] < split_timestamp].copy()
    test_frame = supervised[supervised["date"] >= split_timestamp].copy()
    defaults = _feature_defaults(train_frame)

    return TransitGraphDataset(
        station_to_node=station_to_node,
        node_to_station={node_id: station for station, node_id in station_to_node.items()},
        edge_index=edge_index,
        node_features=node_features,
        train_frame=train_frame,
        test_frame=test_frame,
        context_feature_columns=list(CONTEXT_FEATURE_COLUMNS),
        node_feature_columns=list(STATIC_NODE_FEATURE_COLUMNS),
        encoders=encoders,
        defaults=defaults,
    )


def _normalize_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["station_key"] = df["station_key"].astype(str)
    df["direction"] = df["direction"].astype(str)
    df["vehicle_type"] = df["vehicle_type"].fillna("unknown").astype(str)
    df["day_type"] = df["day_type"].fillna("unknown").astype(str)

    for column in [
        "passenger_boarding",
        "passenger_exiting",
        "stop_sequence",
        *CONTEXT_FEATURE_COLUMNS,
    ]:
        if column in {"line_code", "direction_code", "vehicle_type_code"}:
            continue
        if column not in df.columns:
            df[column] = 0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    return df


def _station_mapping(df: pd.DataFrame) -> dict[str, int]:
    stations = sorted(df["station_key"].dropna().astype(str).unique())
    return {station: idx for idx, station in enumerate(stations)}


def _fit_encoders(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        "line": {str(value): idx for idx, value in enumerate(sorted(df["line"].astype(str).unique()))},
        "direction": {value: idx for idx, value in enumerate(sorted(df["direction"].astype(str).unique()))},
        "vehicle_type": {value: idx for idx, value in enumerate(sorted(df["vehicle_type"].astype(str).unique()))},
    }


def _encode_context_features(df: pd.DataFrame, encoders: dict[str, dict[str, int]]) -> pd.DataFrame:
    encoded = df.copy()
    encoded["line_code"] = encoded["line"].astype(str).map(encoders["line"]).fillna(-1).astype(float)
    encoded["direction_code"] = encoded["direction"].map(encoders["direction"]).fillna(-1).astype(float)
    encoded["vehicle_type_code"] = encoded["vehicle_type"].map(encoders["vehicle_type"]).fillna(-1).astype(float)
    return encoded


def _build_edge_index(df: pd.DataFrame, station_to_node: dict[str, int]) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    grouping_columns = ["line", "direction", "route"]

    for _, group in df.groupby(grouping_columns):
        sequence = (
            group.sort_values(["stop_sequence", "station_key"])
            .drop_duplicates("station_key")["station_key"]
            .astype(str)
            .tolist()
        )
        for source, target in zip(sequence, sequence[1:]):
            source_id = station_to_node[source]
            target_id = station_to_node[target]
            if source_id != target_id:
                edges.add((source_id, target_id))
                edges.add((target_id, source_id))

    if not edges:
        return np.empty((2, 0), dtype=np.int64)

    return np.asarray(sorted(edges), dtype=np.int64).T


def _build_node_features(df: pd.DataFrame, station_to_node: dict[str, int]) -> np.ndarray:
    grouped = (
        df.groupby("station_key")
        .agg(
            mean_boarding=("passenger_boarding", "mean"),
            mean_exiting=("passenger_exiting", "mean"),
            mean_stop_sequence=("stop_sequence", "mean"),
            line_count=("line", "nunique"),
            direction_count=("direction", "nunique"),
        )
        .reindex(station_to_node.keys())
        .fillna(0)
    )
    return grouped[STATIC_NODE_FEATURE_COLUMNS].to_numpy(dtype=np.float32)


def _build_supervised_rows(df: pd.DataFrame, station_to_node: dict[str, int]) -> pd.DataFrame:
    aggregate_spec = {
        "target": ("passenger_boarding", "sum"),
        **{column: (column, "mean") for column in CONTEXT_FEATURE_COLUMNS},
    }
    rows = (
        df.groupby(["date", "line", "direction", "station_key", "hour"], as_index=False)
        .agg(**aggregate_spec)
        .sort_values(["date", "hour", "line", "direction", "station_key"])
    )
    rows["node_id"] = rows["station_key"].map(station_to_node).astype(int)
    return rows


def _feature_defaults(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {column: 0.0 for column in CONTEXT_FEATURE_COLUMNS}
    return {
        column: float(pd.to_numeric(frame[column], errors="coerce").fillna(0).mean())
        for column in CONTEXT_FEATURE_COLUMNS
    }


def require_gnn_dependencies():
    try:
        import torch
        from torch import nn
        from torch_geometric.nn import SAGEConv
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "GNN forecasting requires torch and torch-geometric. "
            "Install the project requirements before selecting forecast_model='gnn'."
        ) from exc
    return torch, nn, SAGEConv


class TrainedGNNForecastModel:
    """Small GraphSAGE-based station demand model used by the SimPy forecast engine."""

    def __init__(
        self,
        dataset: TransitGraphDataset,
        model,
        torch_module,
        node_x,
        edge_index,
        feature_mean,
        feature_std,
        target_scale: str,
    ):
        self.dataset = dataset
        self.model = model
        self.torch = torch_module
        self.node_x = node_x
        self.edge_index = edge_index
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.target_scale = target_scale
        self.model.eval()

    def predict_station(self, route, station: str, hour: int, weekday: int) -> float:
        node_id = self.dataset.station_to_node.get(str(station))
        if node_id is None:
            return 0.0

        row = dict(self.dataset.defaults)
        row.update(
            {
                "hour": float(hour),
                "weekday": float(weekday),
                "line_code": float(self.dataset.encoders["line"].get(str(route.line), -1)),
                "direction_code": float(self.dataset.encoders["direction"].get(str(route.direction), -1)),
                "vehicle_type_code": float(self.dataset.encoders["vehicle_type"].get(str(route.vehicle_type), -1)),
            }
        )
        context = np.asarray([[row[column] for column in self.dataset.context_feature_columns]], dtype=np.float32)
        context = (context - self.feature_mean) / self.feature_std

        with self.torch.no_grad():
            node_ids = self.torch.tensor([int(node_id)], dtype=self.torch.long)
            context_x = self.torch.tensor(context, dtype=self.torch.float32)
            prediction = self.model(self.node_x, self.edge_index, node_ids, context_x).item()
        if self.target_scale == "log1p":
            prediction = np.expm1(prediction)
        return float(max(0.0, prediction))

    def predict_route_demand(self, route, hour: int) -> float:
        return sum(self.predict_station(route, station, hour, 0) for station in route.stations)


def train_gnn_forecast(
    data_dir: Path | str,
    lines: Iterable[int] | None = None,
    split_date: str = DEFAULT_SPLIT_DATE,
    epochs: int = 24,
    learning_rate: float = 0.01,
    hidden_channels: int = 48,
    max_train_rows: int = 200_000,
    random_state: int = 42,
    target_scale: str = "log1p",
) -> TrainedGNNForecastModel:
    """Train a compact GraphSAGE model for station-hour boarding regression."""

    torch, nn, SAGEConv = require_gnn_dependencies()
    dataset = build_transit_graph_dataset(data_dir, lines=lines, split_date=split_date)
    if dataset.train_frame.empty:
        raise ValueError("No training rows available for GNN forecast.")

    class StationDemandGraphSAGE(nn.Module):
        def __init__(self, node_feature_count: int, context_feature_count: int):
            super().__init__()
            self.conv1 = SAGEConv(node_feature_count, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.regressor = nn.Sequential(
                nn.Linear(hidden_channels + context_feature_count, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1),
            )

        def forward(self, node_x, edge_index, node_ids, context_x):
            node_embedding = self.conv1(node_x, edge_index).relu()
            node_embedding = self.conv2(node_embedding, edge_index).relu()
            selected = node_embedding[node_ids]
            return self.regressor(torch.cat([selected, context_x], dim=1)).squeeze(-1)

    train = dataset.train_frame
    if len(train) > max_train_rows:
        train = train.sample(max_train_rows, random_state=random_state)

    feature_values = train[dataset.context_feature_columns].to_numpy(dtype=np.float32)
    feature_mean = feature_values.mean(axis=0, keepdims=True)
    feature_std = feature_values.std(axis=0, keepdims=True)
    feature_std[feature_std == 0] = 1.0
    feature_values = (feature_values - feature_mean) / feature_std

    node_ids = torch.tensor(train["node_id"].to_numpy(dtype=np.int64), dtype=torch.long)
    context_x = torch.tensor(feature_values, dtype=torch.float32)
    target_values = train["target"].to_numpy(dtype=np.float32)
    if target_scale == "log1p":
        target_values = np.log1p(target_values)
    elif target_scale != "none":
        raise ValueError("target_scale must be 'log1p' or 'none'.")
    target = torch.tensor(target_values, dtype=torch.float32)

    node_x = torch.tensor(dataset.node_features, dtype=torch.float32)
    edge_index = torch.tensor(dataset.edge_index, dtype=torch.long)

    model = StationDemandGraphSAGE(
        node_feature_count=dataset.node_features.shape[1],
        context_feature_count=len(dataset.context_feature_columns),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.SmoothL1Loss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        prediction = model(node_x, edge_index, node_ids, context_x)
        loss = loss_fn(prediction, target)
        loss.backward()
        optimizer.step()

    return TrainedGNNForecastModel(
        dataset,
        model,
        torch,
        node_x,
        edge_index,
        feature_mean,
        feature_std,
        target_scale,
    )


def summarize_graph_dataset(data_dir: Path | str, lines: Iterable[int] | None = None) -> dict[str, int]:
    dataset = build_transit_graph_dataset(data_dir, lines=lines)
    return {
        "nodes": dataset.num_nodes,
        "edges": dataset.num_edges,
        "train_rows": len(dataset.train_frame),
        "test_rows": len(dataset.test_frame),
        "node_features": dataset.node_features.shape[1],
        "context_features": len(dataset.context_feature_columns),
    }
