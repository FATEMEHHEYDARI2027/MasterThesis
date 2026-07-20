"""Batch-oriented, memory-bounded cycle-level feature table generation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

logger = logging.getLogger(__name__)

CYCLE_INDEX_COLUMNS: tuple[str, ...] = (
    "experiment",
    "session_id",
    "cycle_id",
    "start_time",
    "end_time",
    "duration_seconds",
)

STAT_SUFFIXES: tuple[str, ...] = ("count", "min", "max", "mean", "std", "rms")


def _discover_signal_names(measurement_dataset_path: Path) -> list[str]:
    """Return the sorted set of signal names present in the measurement store.

    Only the ``signal_name`` column is scanned so this stays cheap even for a
    measurement store containing hundreds of millions of raw samples.
    """

    if not measurement_dataset_path.exists():
        return []

    dataset = ds.dataset(measurement_dataset_path, format="parquet")
    signal_name_table = dataset.to_table(columns=["signal_name"])
    if signal_name_table.num_rows == 0:
        return []
    signal_names = pd.unique(signal_name_table.column("signal_name").to_pandas())
    return sorted(str(name) for name in signal_names)


def _empty_feature_row(
    experiment: str,
    session_id: int,
    cycle_id: int,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    duration_seconds: float,
    signal_names: list[str],
) -> dict[str, object]:
    """Build one feature row with missing-signal indicators set to True."""

    row: dict[str, object] = {
        "experiment": experiment,
        "session_id": session_id,
        "cycle_id": cycle_id,
        "cycle_start_time": start_time,
        "cycle_end_time": end_time,
        "cycle_duration_seconds": duration_seconds,
    }
    for signal_name in signal_names:
        row[f"{signal_name}_is_missing"] = True
        for suffix in STAT_SUFFIXES:
            row[f"{signal_name}_{suffix}"] = np.nan if suffix != "count" else 0
    return row


def _compute_batch_features(
    cycle_batch: pd.DataFrame,
    measurements_df: pd.DataFrame,
    signal_names: list[str],
) -> list[dict[str, object]]:
    """Compute one feature row per cycle in the batch from its measurements."""

    grouped = (
        measurements_df.groupby(["cycle_id", "signal_name"])["value"]
        if not measurements_df.empty
        else None
    )
    stats_by_cycle: dict[int, dict[str, dict[str, float]]] = {}
    if grouped is not None:
        for (cycle_id, signal_name), values in grouped:
            values_array = values.to_numpy(dtype=float)
            stats_by_cycle.setdefault(int(cycle_id), {})[str(signal_name)] = {
                "count": int(values_array.size),
                "min": float(np.min(values_array)),
                "max": float(np.max(values_array)),
                "mean": float(np.mean(values_array)),
                "std": float(np.std(values_array)) if values_array.size > 1 else 0.0,
                "rms": float(np.sqrt(np.mean(np.square(values_array)))),
            }

    feature_rows: list[dict[str, object]] = []
    for cycle_row in cycle_batch.itertuples(index=False):
        cycle_id = int(cycle_row.cycle_id)
        session_id = int(cycle_row.session_id)
        experiment = str(cycle_row.experiment) if hasattr(cycle_row, "experiment") else ""
        start_time = pd.Timestamp(cycle_row.start_time)
        end_time = pd.Timestamp(cycle_row.end_time)
        duration_seconds = (
            float(cycle_row.duration_seconds)
            if hasattr(cycle_row, "duration_seconds")
            else (end_time - start_time).total_seconds()
        )

        row: dict[str, object] = {
            "experiment": experiment,
            "session_id": session_id,
            "cycle_id": cycle_id,
            "cycle_start_time": start_time,
            "cycle_end_time": end_time,
            "cycle_duration_seconds": duration_seconds,
        }
        cycle_stats = stats_by_cycle.get(cycle_id, {})
        for signal_name in signal_names:
            signal_stats = cycle_stats.get(signal_name)
            is_missing = signal_stats is None
            row[f"{signal_name}_is_missing"] = is_missing
            for suffix in STAT_SUFFIXES:
                if is_missing:
                    row[f"{signal_name}_{suffix}"] = np.nan if suffix != "count" else 0
                else:
                    row[f"{signal_name}_{suffix}"] = signal_stats[suffix]
        feature_rows.append(row)

    return feature_rows


def build_cycle_feature_table(
    measurement_dataset_path: Path,
    cycle_index_path: Path,
    output_path: Path,
    feature_batch_size: int = 1000,
) -> Path:
    """Build one cycle-level feature table without loading the full measurement store.

    Cycles are processed ``feature_batch_size`` at a time: only the Parquet
    row groups belonging to that batch's sessions and cycle IDs are read from
    ``measurement_dataset_path`` for each batch, features are computed, and
    the batch's raw measurements are released before moving on.
    """

    measurement_dataset_path = Path(measurement_dataset_path)
    cycle_index_path = Path(cycle_index_path)
    output_path = Path(output_path)

    cycles_df = pd.read_parquet(cycle_index_path)
    signal_names = _discover_signal_names(measurement_dataset_path)

    feature_rows: list[dict[str, object]] = []

    if cycles_df.empty:
        feature_table_df = pd.DataFrame(
            columns=list(CYCLE_INDEX_COLUMNS[:2])
            + ["cycle_id", "cycle_start_time", "cycle_end_time", "cycle_duration_seconds"]
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        feature_table_df.to_parquet(output_path, index=False)
        return output_path

    dataset_available = measurement_dataset_path.exists()
    dataset = (
        ds.dataset(measurement_dataset_path, format="parquet")
        if dataset_available
        else None
    )

    for batch_index_start in range(0, len(cycles_df), feature_batch_size):
        cycle_batch = cycles_df.iloc[batch_index_start : batch_index_start + feature_batch_size]
        cycle_ids = cycle_batch["cycle_id"].astype(int).tolist()

        if dataset is not None and cycle_ids:
            filter_expression = ds.field("cycle_id").isin(cycle_ids)
            measurements_table = dataset.to_table(filter=filter_expression)
            measurements_df = measurements_table.to_pandas()
        else:
            measurements_df = pd.DataFrame(columns=["cycle_id", "signal_name", "value"])

        feature_rows.extend(_compute_batch_features(cycle_batch, measurements_df, signal_names))

        logger.info(
            "Computed cycle features for %d cycle(s) in batch starting at row %d",
            len(cycle_ids),
            batch_index_start,
        )
        del measurements_df

    feature_table_df = pd.DataFrame(feature_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_table_df.to_parquet(output_path, index=False)
    logger.info("Wrote cycle feature table with %d row(s) to %s", len(feature_table_df), output_path)
    return output_path
