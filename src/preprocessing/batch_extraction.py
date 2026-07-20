"""Batch-oriented, memory-bounded multi-sensor cycle extraction."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path

import pandas as pd

from src.preprocessing.multi_sensor_cycle_extraction import (
    extract_cycle_measurements,
    list_experiment_signals,
)

logger = logging.getLogger(__name__)

LONG_FORMAT_COLUMNS: tuple[str, ...] = (
    "experiment",
    "session_id",
    "cycle_id",
    "signal_name",
    "source",
    "signal_id",
    "signal_id_uuid",
    "unit_code",
    "unit_symbol",
    "time",
    "value",
)

SIGNAL_SUMMARY_COLUMNS: tuple[str, ...] = (
    "experiment",
    "session_id",
    "cycle_id",
    "signal_name",
    "number_of_samples",
    "is_missing",
    "minimum_time",
    "maximum_time",
    "minimum_value",
    "maximum_value",
    "signal_present",
    "missing_signal",
    "sample_count",
    "finite_sample_count",
    "extraction_error",
    "first_timestamp",
    "last_timestamp",
)


def iter_cycle_batches(cycles_df: pd.DataFrame, batch_size: int) -> Iterator[pd.DataFrame]:
    """Yield consecutive, bounded-size slices of the cycle index.

    Cycles are never loaded into one giant DataFrame downstream; each batch is
    processed and released independently by the caller.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    total_rows = len(cycles_df)
    for start_index in range(0, total_rows, batch_size):
        yield cycles_df.iloc[start_index : start_index + batch_size]


def _resolve_signal_descriptors(
    uuid_signal_info: pd.DataFrame,
    int_signal_info: pd.DataFrame,
    experiment: str,
    selected_signals: Sequence[str] | None,
) -> pd.DataFrame:
    """Return the experiment's signal catalogue, optionally restricted."""

    signal_descriptors = list_experiment_signals(
        uuid_signal_info=uuid_signal_info,
        int_signal_info=int_signal_info,
        experiment=experiment,
    )
    if selected_signals:
        allowed_names = set(selected_signals)
        signal_descriptors = signal_descriptors[
            signal_descriptors["signal_name"].isin(allowed_names)
        ]
    return signal_descriptors.reset_index(drop=True)


def _extract_cycle_batch_with_summary(
    base_dir: Path,
    cycle_batch: pd.DataFrame,
    uuid_signal_info: pd.DataFrame,
    int_signal_info: pd.DataFrame,
    experiment: str,
    selected_signals: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract one cycle batch, returning both measurements and a summary.

    This is the shared implementation behind :func:`extract_cycle_batch`
    (measurements only) and :func:`src.storage.batched_measurement_writer`
    (which also needs the missing-signal summary) so the underlying signal
    catalogue and raw extraction only happen once per batch.
    """

    signal_descriptors = _resolve_signal_descriptors(
        uuid_signal_info, int_signal_info, experiment, selected_signals
    )
    descriptor_by_name = {
        str(row.signal_name): row for row in signal_descriptors.itertuples(index=False)
    }

    measurement_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for cycle_row in cycle_batch.itertuples(index=False):
        cycle_id = int(cycle_row.cycle_id)
        session_id = int(cycle_row.session_id) if hasattr(cycle_row, "session_id") else None

        cycle_extraction_error = False
        try:
            extracted_signals = extract_cycle_measurements(
                base_dir=base_dir,
                cycle_start=pd.Timestamp(cycle_row.start_time),
                cycle_end=pd.Timestamp(cycle_row.end_time) + pd.Timedelta(microseconds=1),
                uuid_signal_info=uuid_signal_info,
                int_signal_info=int_signal_info,
                experiment=experiment,
            )
        except FileNotFoundError:
            logger.warning(
                "Skipping cycle_id=%d because its measurement partitions are missing",
                cycle_id,
            )
            extracted_signals = {}
            cycle_extraction_error = True
        except Exception:  # noqa: BLE001 - a technical extraction error must
            # never silently propagate and abort the whole batch; the cycle
            # is recorded as a genuine extraction failure instead.
            logger.exception(
                "Extraction failed for cycle_id=%d due to a technical error", cycle_id
            )
            extracted_signals = {}
            cycle_extraction_error = True

        if selected_signals:
            extracted_signals = {
                name: signal_df
                for name, signal_df in extracted_signals.items()
                if name in descriptor_by_name
            }

        for signal_name, descriptor in descriptor_by_name.items():
            signal_df = extracted_signals.get(signal_name)
            is_missing = signal_df is None or signal_df.empty

            if signal_df is not None and not signal_df.empty:
                measurement_frames.append(
                    pd.DataFrame(
                        {
                            "experiment": experiment,
                            "session_id": session_id,
                            "cycle_id": cycle_id,
                            "signal_name": signal_name,
                            "source": descriptor.source,
                            "signal_id": descriptor.signal_id,
                            "signal_id_uuid": descriptor.signal_id_uuid,
                            "unit_code": descriptor.unit_code,
                            "unit_symbol": descriptor.unit_symbol,
                            "time": signal_df["time"].to_numpy(),
                            "value": signal_df["value"].to_numpy(),
                        }
                    )
                )

            sample_count = 0 if signal_df is None else int(len(signal_df))
            finite_sample_count = (
                0
                if signal_df is None or signal_df.empty
                else int(pd.to_numeric(signal_df["value"], errors="coerce").notna().sum())
            )

            summary_rows.append(
                {
                    "experiment": experiment,
                    "session_id": session_id,
                    "cycle_id": cycle_id,
                    "signal_name": signal_name,
                    "number_of_samples": sample_count,
                    # ``is_missing`` reflects signal *unavailability*, never a
                    # technical extraction failure. An absent optional signal
                    # (e.g. duty-cycled vibration) must not be conflated with
                    # a real extraction error.
                    "is_missing": bool(is_missing),
                    "minimum_time": (
                        signal_df["time"].min() if signal_df is not None and not signal_df.empty else None
                    ),
                    "maximum_time": (
                        signal_df["time"].max() if signal_df is not None and not signal_df.empty else None
                    ),
                    "minimum_value": (
                        signal_df["value"].min() if signal_df is not None and not signal_df.empty else None
                    ),
                    "maximum_value": (
                        signal_df["value"].max() if signal_df is not None and not signal_df.empty else None
                    ),
                    "signal_present": not is_missing,
                    "missing_signal": bool(is_missing),
                    "sample_count": sample_count,
                    "finite_sample_count": finite_sample_count,
                    "extraction_error": cycle_extraction_error,
                    "first_timestamp": (
                        signal_df["time"].min() if signal_df is not None and not signal_df.empty else None
                    ),
                    "last_timestamp": (
                        signal_df["time"].max() if signal_df is not None and not signal_df.empty else None
                    ),
                }
            )

    measurements_df = (
        pd.concat(measurement_frames, ignore_index=True)
        if measurement_frames
        else pd.DataFrame(columns=list(LONG_FORMAT_COLUMNS))
    )
    summary_df = pd.DataFrame(summary_rows, columns=list(SIGNAL_SUMMARY_COLUMNS))
    return measurements_df, summary_df


def extract_cycle_batch(
    base_dir: Path,
    cycle_batch: pd.DataFrame,
    uuid_signal_info: pd.DataFrame,
    int_signal_info: pd.DataFrame,
    experiment: str,
    selected_signals: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Extract one batch of cycles into a single long-format DataFrame.

    Original timestamps and native per-signal sampling frequencies are kept
    as-is; nothing is resampled or interpolated. Missing signals or missing
    measurement partitions are skipped (with a warning) rather than filled
    with fabricated values.
    """

    measurements_df, _ = _extract_cycle_batch_with_summary(
        base_dir=base_dir,
        cycle_batch=cycle_batch,
        uuid_signal_info=uuid_signal_info,
        int_signal_info=int_signal_info,
        experiment=experiment,
        selected_signals=selected_signals,
    )
    return measurements_df
