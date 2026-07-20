"""Exploratory candidate-cycle detection for position signals."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

CYCLE_COLUMNS = [
    "cycle_id",
    "start_index",
    "end_index",
    "start_time",
    "end_time",
    "duration_seconds",
    "number_of_samples",
    "minimum_position",
    "maximum_position",
    "mean_position",
]


def _prepare_position_frame(position_df: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted, index-reset frame ready for cycle detection."""

    required_columns = {"time", "value", "signal_id"}
    missing_columns = required_columns.difference(position_df.columns)
    if missing_columns:
        raise ValueError(
            f"position_df is missing required columns: {sorted(missing_columns)}"
        )

    cleaned_df = position_df.loc[:, ["time", "value", "signal_id"]].copy()
    cleaned_df["time"] = pd.to_datetime(cleaned_df["time"], errors="coerce")
    cleaned_df["value"] = pd.to_numeric(cleaned_df["value"], errors="coerce")

    missing_rows = cleaned_df["time"].isna() | cleaned_df["value"].isna()
    if missing_rows.any():
        logger.warning(
            "Dropping %d rows with missing or invalid time/value data before cycle detection",
            int(missing_rows.sum()),
        )
        cleaned_df = cleaned_df.loc[~missing_rows].copy()

    if cleaned_df.empty:
        return cleaned_df.reset_index(drop=True)

    cleaned_df = cleaned_df.sort_values("time").reset_index(drop=True)
    return cleaned_df


def detect_candidate_cycles(
    position_df: pd.DataFrame,
    movement_threshold: float = 1.0,
) -> pd.DataFrame:
    """Detect exploratory candidate cycles from a position signal window.

    A cycle begins at the first sample that crosses from ``<= movement_threshold``
    to ``> movement_threshold`` and ends at the first sample that crosses from
    ``> movement_threshold`` back to ``<= movement_threshold``. Incomplete cycles
    at the beginning or end of the selected window are intentionally excluded.
    """

    if movement_threshold < 0:
        raise ValueError("movement_threshold must be non-negative.")

    prepared_df = _prepare_position_frame(position_df)
    if prepared_df.empty:
        logger.info("No samples available for candidate-cycle detection")
        return pd.DataFrame(columns=CYCLE_COLUMNS)

    is_moving = prepared_df["value"] > movement_threshold
    previous_is_moving = is_moving.shift(1)

    cycle_start_indices = prepared_df.index[
        is_moving & previous_is_moving.eq(False).fillna(False)
    ].to_list()
    cycle_end_indices = prepared_df.index[
        (~is_moving) & previous_is_moving.eq(True).fillna(False)
    ].to_list()

    cycles: list[dict[str, object]] = []
    end_pointer = 0

    for cycle_id, start_index in enumerate(cycle_start_indices, start=1):
        while end_pointer < len(cycle_end_indices) and cycle_end_indices[end_pointer] <= start_index:
            end_pointer += 1

        if end_pointer >= len(cycle_end_indices):
            break

        end_index = cycle_end_indices[end_pointer]
        cycle_slice = prepared_df.iloc[start_index : end_index + 1]

        start_time = pd.Timestamp(cycle_slice["time"].iloc[0])
        end_time = pd.Timestamp(cycle_slice["time"].iloc[-1])
        cycles.append(
            {
                "cycle_id": cycle_id,
                "start_index": int(start_index),
                "end_index": int(end_index),
                "start_time": start_time,
                "end_time": end_time,
                "duration_seconds": (end_time - start_time).total_seconds(),
                "number_of_samples": int(len(cycle_slice)),
                "minimum_position": float(cycle_slice["value"].min()),
                "maximum_position": float(cycle_slice["value"].max()),
                "mean_position": float(cycle_slice["value"].mean()),
            }
        )
        end_pointer += 1

    cycles_df = pd.DataFrame(cycles, columns=CYCLE_COLUMNS)
    logger.info(
        "Detected %d candidate cycles with movement threshold %.3f",
        len(cycles_df),
        movement_threshold,
    )
    return cycles_df
