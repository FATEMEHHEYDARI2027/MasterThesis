"""Write the detected-cycle index as one scalable Parquet table."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PREFERRED_COLUMN_ORDER: tuple[str, ...] = (
    "experiment",
    "session_id",
    "cycle_id",
    "reference_signal_uuid",
    "start_time",
    "end_time",
    "duration_seconds",
    "number_of_samples",
    "minimum_position",
    "maximum_position",
    "mean_position",
)


def _order_columns(cycles_df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns so the preferred index columns come first, when present."""

    ordered_columns = [column for column in PREFERRED_COLUMN_ORDER if column in cycles_df.columns]
    remaining_columns = [column for column in cycles_df.columns if column not in ordered_columns]
    return cycles_df.loc[:, ordered_columns + remaining_columns]


def write_cycle_index(cycles_df: pd.DataFrame, output_path: Path) -> Path:
    """Save all detected cycles as one sorted, scalable Parquet file.

    The full cycle index is written as a single Parquet file (never one file
    per cycle) so it stays a cheap, queryable table even for datasets with
    hundreds of thousands of detected cycles.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ordered_df = _order_columns(cycles_df.copy())

    sort_columns = [
        column for column in ("session_id", "start_time", "cycle_id") if column in ordered_df.columns
    ]
    if sort_columns:
        ordered_df = ordered_df.sort_values(sort_columns).reset_index(drop=True)

    for time_column in ("start_time", "end_time"):
        if time_column in ordered_df.columns:
            ordered_df[time_column] = pd.to_datetime(ordered_df[time_column])

    ordered_df.to_parquet(output_path, index=False)
    logger.info("Wrote cycle index with %d cycles to %s", len(ordered_df), output_path)
    return output_path
