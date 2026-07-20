"""Utilities for detecting recording sessions in large UUID-based signals."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from src.utils.measurement_loader import SIGNAL_DATASET_NAME

logger = logging.getLogger(__name__)

SESSION_COLUMNS = ["time"]
SESSION_TABLE_COLUMNS = [
    "session_id",
    "start_time",
    "end_time",
    "duration_seconds",
    "number_of_samples",
    "maximum_internal_gap_seconds",
]
# Default threshold determined from the empirical timestamp-gap analysis.
# Represents one hour and can be overridden when required.
DEFAULT_SESSION_GAP_SECONDS = 3600.0


@dataclass
class _OpenSession:
    """Mutable state for one recording session being assembled incrementally."""

    session_id: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    number_of_samples: int
    maximum_internal_gap_seconds: float

    def to_row(self) -> dict[str, object]:
        """Return a finalized row representation for the session."""

        duration_seconds = (self.end_time - self.start_time).total_seconds()
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": duration_seconds,
            "number_of_samples": self.number_of_samples,
            "maximum_internal_gap_seconds": self.maximum_internal_gap_seconds,
        }


def _as_signal_dataset_path(base_dir: Path) -> Path:
    """Return the resolved path to the UUID signal dataset."""

    return Path(base_dir).expanduser().resolve() / SIGNAL_DATASET_NAME


def _append_ordered_times(
    times: np.ndarray,
    gap_threshold_seconds: float,
    sessions: list[dict[str, object]],
    current_session: _OpenSession | None,
    previous_time: np.datetime64 | None,
    next_session_id: int,
) -> tuple[_OpenSession | None, np.datetime64 | None, int]:
    """Append one sorted timestamp array into the incremental session state."""

    if len(times) == 0:
        return current_session, previous_time, next_session_id

    gaps = (
        np.diff(times) / np.timedelta64(1, "s")
        if len(times) > 1
        else np.empty(0, dtype=float)
    )
    boundary_gap_seconds: float | None = None
    if previous_time is not None:
        boundary_gap_seconds = float((times[0] - previous_time) / np.timedelta64(1, "s"))

    split_points = np.nonzero(gaps > gap_threshold_seconds)[0] + 1
    segment_starts = np.concatenate(([0], split_points))
    segment_ends = np.concatenate((split_points - 1, [len(times) - 1]))

    for segment_index, (segment_start, segment_end) in enumerate(
        zip(segment_starts, segment_ends, strict=True)
    ):
        segment_sample_count = int(segment_end - segment_start + 1)
        segment_start_time = pd.Timestamp(times[segment_start])
        segment_end_time = pd.Timestamp(times[segment_end])
        segment_max_gap = (
            float(gaps[segment_start:segment_end].max())
            if segment_end > segment_start
            else 0.0
        )

        start_new_session = current_session is None or segment_index > 0
        if (
            segment_index == 0
            and boundary_gap_seconds is not None
            and boundary_gap_seconds > gap_threshold_seconds
        ):
            start_new_session = True

        if start_new_session:
            if current_session is not None:
                sessions.append(current_session.to_row())

            current_session = _OpenSession(
                session_id=next_session_id,
                start_time=segment_start_time,
                end_time=segment_end_time,
                number_of_samples=segment_sample_count,
                maximum_internal_gap_seconds=segment_max_gap,
            )
            next_session_id += 1
        else:
            current_session.end_time = segment_end_time
            current_session.number_of_samples += segment_sample_count
            current_session.maximum_internal_gap_seconds = max(
                current_session.maximum_internal_gap_seconds,
                segment_max_gap,
            )
            if boundary_gap_seconds is not None:
                current_session.maximum_internal_gap_seconds = max(
                    current_session.maximum_internal_gap_seconds,
                    boundary_gap_seconds,
                )

        previous_time = times[segment_end]

    return current_session, previous_time, next_session_id


def detect_recording_sessions(
    base_dir: Path,
    signal_id_uuid: str,
    gap_threshold_seconds: float = DEFAULT_SESSION_GAP_SECONDS,
    batch_size: int = 100_000,
) -> pd.DataFrame:
    """Detect continuous recording sessions for one UUID-based signal.

    The dataset is scanned incrementally in record batches so the complete signal
    lifetime is never loaded into a single pandas DataFrame. The default gap
    threshold comes from the empirical timestamp-gap analysis for this dataset
    and may be overridden when adapting the preprocessing step to future
    datasets.
    """

    if gap_threshold_seconds <= 0:
        raise ValueError("gap_threshold_seconds must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    dataset_root = _as_signal_dataset_path(base_dir)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Signal dataset not found: {dataset_root}")

    dataset = ds.dataset(dataset_root, format="parquet", partitioning="hive")
    filter_expression = ds.field("signal_id") == signal_id_uuid
    fragments = sorted(
        dataset.get_fragments(filter=filter_expression),
        key=lambda fragment: fragment.path,
    )

    logger.info(
        (
            "Detecting recording sessions for UUID signal %s from %s. "
            "Gap threshold: %.0f seconds (%.1f hour)"
        ),
        signal_id_uuid,
        dataset_root,
        gap_threshold_seconds,
        gap_threshold_seconds / 3600.0,
    )

    if not fragments:
        logger.info("No fragments found for UUID signal %s", signal_id_uuid)
        return pd.DataFrame(columns=SESSION_TABLE_COLUMNS)

    sessions: list[dict[str, object]] = []
    current_session: _OpenSession | None = None
    previous_time: np.datetime64 | None = None
    next_session_id = 1
    total_samples_scanned = 0

    for fragment in fragments:
        fragment_time_chunks: list[np.ndarray] = []

        for batch in fragment.to_batches(
            columns=SESSION_COLUMNS,
            batch_size=batch_size,
        ):
            times = batch.column("time").to_numpy(zero_copy_only=False)
            if len(times) == 0:
                continue

            total_samples_scanned += len(times)
            fragment_time_chunks.append(times.copy())

        if not fragment_time_chunks:
            continue

        fragment_times = np.concatenate(fragment_time_chunks)
        fragment_times.sort()
        current_session, previous_time, next_session_id = _append_ordered_times(
            fragment_times,
            gap_threshold_seconds,
            sessions,
            current_session,
            previous_time,
            next_session_id,
        )

    if current_session is not None:
        sessions.append(current_session.to_row())

    sessions_df = pd.DataFrame(sessions, columns=SESSION_TABLE_COLUMNS)
    logger.info(
        (
            "Detected %d recording sessions across %d samples for UUID signal %s. "
            "Gap threshold: %.0f seconds (%.1f hour)"
        ),
        len(sessions_df),
        total_samples_scanned,
        signal_id_uuid,
        gap_threshold_seconds,
        gap_threshold_seconds / 3600.0,
    )
    return sessions_df
