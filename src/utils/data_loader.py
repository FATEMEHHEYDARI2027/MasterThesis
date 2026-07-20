"""Reusable metadata loaders for signal discovery without reading measurement data.

This module contains reusable library code only. It reads small metadata parquet
files and intentionally does not load ``signal_data_point.parquet`` or
``vibration.parquet``.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

PathLike = str | Path
MetadataFrames = dict[str, pd.DataFrame]

_METADATA_FILES: dict[str, str] = {
    "signal_data_point_rel": "signal_data_point_rel.parquet",
    "signal_data_point_rel_int": "signal_data_point_rel_int.parquet",
    "nodes": "nodes.parquet",
    "units": "units.parquet",
}

_UUID_CANDIDATE_COLUMNS: tuple[str, ...] = (
    "signal_id",
    "node_id",
    "parent_node",
    "predecessor",
)


def _as_path(base_dir: PathLike) -> Path:
    """Return ``base_dir`` as a resolved ``Path``."""

    return Path(base_dir).expanduser().resolve()


def _uuid_to_string(value: Any) -> Any:
    """Convert binary UUID values to readable UUID strings when possible."""

    if pd.isna(value):
        return pd.NA

    if isinstance(value, uuid.UUID):
        return str(value)

    if isinstance(value, memoryview):
        value = value.tobytes()

    if isinstance(value, (bytes, bytearray)):
        if len(value) == 16:
            return str(uuid.UUID(bytes=bytes(value)))
        return value.decode("utf-8")

    return value


def _convert_uuid_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert UUID-like columns in a DataFrame to readable strings."""

    converted = frame.copy()
    for column in _UUID_CANDIDATE_COLUMNS:
        if column in converted.columns:
            converted[column] = converted[column].map(_uuid_to_string)
    return converted


def _read_metadata_frame(base_path: Path, file_name: str) -> pd.DataFrame:
    """Read one required metadata parquet file."""

    file_path = base_path / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Required metadata file not found: {file_path}")

    logger.info("Loading metadata file: %s", file_path)
    return _convert_uuid_columns(pd.read_parquet(file_path))


def load_metadata(base_dir: PathLike) -> MetadataFrames:
    """Load the small metadata parquet files needed for signal lookup.

    Parameters
    ----------
    base_dir:
        Directory containing the ERA metadata parquet files.

    Returns
    -------
    dict[str, pandas.DataFrame]
        A mapping containing ``signal_data_point_rel``,
        ``signal_data_point_rel_int``, ``nodes``, and ``units``.
    """

    base_path = _as_path(base_dir)
    metadata = {
        key: _read_metadata_frame(base_path, file_name)
        for key, file_name in _METADATA_FILES.items()
    }
    logger.info(
        "Loaded metadata from %s: rel=%d, rel_int=%d, nodes=%d, units=%d",
        base_path,
        len(metadata["signal_data_point_rel"]),
        len(metadata["signal_data_point_rel_int"]),
        len(metadata["nodes"]),
        len(metadata["units"]),
    )
    return metadata


def build_node_paths(nodes_df: pd.DataFrame) -> pd.DataFrame:
    """Build full hierarchical node paths from ``nodes.parquet`` contents.

    Parameters
    ----------
    nodes_df:
        DataFrame loaded from ``nodes.parquet``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with ``node_id_uuid``, ``name``, and full ``path`` columns.
    """

    node_names = nodes_df.set_index("node_id")["name"].fillna("").to_dict()
    parent_nodes = nodes_df.set_index("node_id")["parent_node"].to_dict()
    cached_paths: dict[str, str] = {}

    def build_path(node_id: str) -> str:
        if node_id in cached_paths:
            return cached_paths[node_id]

        name = node_names.get(node_id, "")
        parent_node = parent_nodes.get(node_id)

        if pd.isna(parent_node) or parent_node in ("", None):
            path = name
        else:
            parent_path = build_path(parent_node)
            path = f"{parent_path}/{name}" if parent_path else name

        cached_paths[node_id] = path
        return path

    node_paths = nodes_df[["node_id", "name"]].copy()
    node_paths["path"] = node_paths["node_id"].map(build_path)
    return node_paths.rename(columns={"node_id": "node_id_uuid"})


def _build_unit_labels(units_df: pd.DataFrame) -> pd.DataFrame:
    """Map semantic unit codes to readable physical unit symbols."""

    unit_labels = units_df[["common_code", "symbol", "name"]].copy()
    unit_labels["unit_symbol"] = (
        unit_labels["symbol"]
        .replace("", pd.NA)
        .fillna(unit_labels["name"])
        .fillna(unit_labels["common_code"])
    )
    return unit_labels[["common_code", "unit_symbol"]].drop_duplicates()


def _finalize_signal_info(
    signals_df: pd.DataFrame,
    node_paths: pd.DataFrame,
    unit_labels: pd.DataFrame,
) -> pd.DataFrame:
    """Attach readable node paths and unit metadata to signal metadata."""

    signal_info = signals_df.merge(
        node_paths[["node_id_uuid", "path"]],
        on="node_id_uuid",
        how="left",
    )
    signal_info = signal_info.merge(
        unit_labels,
        left_on="unit_code",
        right_on="common_code",
        how="left",
    )
    signal_info["unit_symbol"] = signal_info["unit_symbol"].fillna(signal_info["unit_code"])
    signal_info = signal_info.drop(columns=["common_code"])
    signal_info["signal_id"] = signal_info["signal_id"].astype("Int64")

    return signal_info[
        ["signal_id", "signal_id_uuid", "node_id_uuid", "unit_code", "unit_symbol", "path"]
    ].sort_values(
        ["path", "unit_code", "signal_id", "signal_id_uuid"],
        na_position="last",
    ).reset_index(drop=True)


def build_uuid_signal_info_from_metadata(metadata: MetadataFrames) -> pd.DataFrame:
    """Build metadata for UUID-based signals from already loaded metadata."""

    node_paths = build_node_paths(metadata["nodes"])
    unit_labels = _build_unit_labels(metadata["units"])

    uuid_signals = metadata["signal_data_point_rel"].rename(
        columns={
            "signal_id": "signal_id_uuid",
            "node_id": "node_id_uuid",
            "unit": "unit_code",
        }
    ).copy()
    uuid_signals["signal_id"] = pd.NA
    uuid_signals = uuid_signals[
        ["signal_id", "signal_id_uuid", "node_id_uuid", "unit_code"]
    ]

    signal_info = _finalize_signal_info(uuid_signals, node_paths, unit_labels)
    logger.info("Built uuid_signal_info with %d rows", len(signal_info))
    return signal_info


def build_int_signal_info_from_metadata(metadata: MetadataFrames) -> pd.DataFrame:
    """Build metadata for INT vibration signals from already loaded metadata."""

    node_paths = build_node_paths(metadata["nodes"])
    unit_labels = _build_unit_labels(metadata["units"])

    int_signals = metadata["signal_data_point_rel_int"].rename(
        columns={"node_id": "node_id_uuid", "unit": "unit_code"}
    ).copy()
    int_signals["signal_id_uuid"] = pd.NA
    int_signals = int_signals[
        ["signal_id", "signal_id_uuid", "node_id_uuid", "unit_code"]
    ]

    signal_info = _finalize_signal_info(int_signals, node_paths, unit_labels)
    logger.info("Built int_signal_info with %d rows", len(signal_info))
    return signal_info


def build_uuid_signal_info(base_dir: PathLike) -> pd.DataFrame:
    """Build metadata for UUID-based signals without loading measurements."""

    metadata = load_metadata(base_dir)
    return build_uuid_signal_info_from_metadata(metadata)


def build_int_signal_info(base_dir: PathLike) -> pd.DataFrame:
    """Build metadata for INT vibration signals without loading measurements."""

    metadata = load_metadata(base_dir)
    return build_int_signal_info_from_metadata(metadata)


def find_signals(
    signal_info: pd.DataFrame,
    path_contains: str | None = None,
    name_contains: str | None = None,
    unit_code: str | None = None,
) -> pd.DataFrame:
    """Filter a signal metadata table by path text, leaf node name, and unit code."""

    filtered = signal_info.copy()
    mask = pd.Series(True, index=filtered.index)

    if path_contains:
        mask &= filtered["path"].fillna("").str.contains(path_contains, case=False, na=False)

    if name_contains:
        node_names = filtered["path"].fillna("").str.rsplit("/", n=1).str[-1]
        mask &= node_names.str.contains(name_contains, case=False, na=False)

    if unit_code:
        mask &= filtered["unit_code"].fillna("").str.contains(unit_code, case=False, na=False)

    return filtered.loc[mask].reset_index(drop=True)