"""Reusable extraction of multi-sensor measurements for one detected cycle."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.measurement_loader import load_int_signal, load_uuid_signal
from src.utils.session_signal_cache import SessionSignalCache

logger = logging.getLogger(__name__)


def _slugify_path_part(value: str) -> str:
    """Convert one path segment to a lowercase identifier."""

    return value.strip().lower().replace(" ", "_")


def _base_signal_name(path: str, unit_code: str, source: str) -> str:
    """Derive a stable logical signal name from metadata."""

    path_parts = [_slugify_path_part(part) for part in path.split("/") if part]
    leaf_name = path_parts[-1] if path_parts else unit_code.lower()
    unit_name = unit_code.lower()

    if leaf_name.startswith("sensor_") or leaf_name.startswith("analog_"):
        return leaf_name

    if leaf_name == "drive" and unit_name in {"position", "velocity", "current"}:
        return unit_name

    if unit_name == "pressure":
        return "pressure"
    if unit_name == "temperature":
        return "temperature"
    if unit_name == "counter":
        return "counter"
    if unit_name == "vibration" and source == "uuid":
        return f"vibration_{leaf_name}"

    return f"{unit_name}_{leaf_name}"


def _unique_signal_name(base_name: str, path: str, used_names: set[str]) -> str:
    """Return a unique signal name by adding path-based suffixes when needed."""

    if base_name not in used_names:
        used_names.add(base_name)
        return base_name

    path_parts = [_slugify_path_part(part) for part in path.split("/") if part]
    for part in reversed(path_parts):
        candidate = f"{base_name}_{part}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate

    suffix = 2
    while True:
        candidate = f"{base_name}_{suffix}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        suffix += 1


def list_experiment_signals(
    uuid_signal_info: pd.DataFrame,
    int_signal_info: pd.DataFrame,
    experiment: str,
) -> pd.DataFrame:
    """List all available signals belonging to one experiment with stable names."""

    uuid_signals = uuid_signal_info[
        uuid_signal_info["path"].fillna("").str.contains(experiment, case=False, na=False)
    ].copy()
    int_signals = int_signal_info[
        int_signal_info["path"].fillna("").str.contains(experiment, case=False, na=False)
    ].copy()

    uuid_signals["source"] = "uuid"
    int_signals["source"] = "int"
    combined = pd.concat([uuid_signals, int_signals], ignore_index=True, sort=False)
    if combined.empty:
        logger.warning("No signals found for experiment %s", experiment)
        return combined

    used_names: set[str] = set()
    signal_names: list[str] = []
    for row in combined.itertuples(index=False):
        base_name = _base_signal_name(row.path, row.unit_code, row.source)
        signal_names.append(_unique_signal_name(base_name, row.path, used_names))

    described = combined.copy()
    described["signal_name"] = signal_names
    return described[
        [
            "signal_name",
            "source",
            "signal_id",
            "signal_id_uuid",
            "unit_code",
            "unit_symbol",
            "path",
        ]
    ].reset_index(drop=True)


def extract_cycle_measurements(
    base_dir: Path,
    cycle_start: pd.Timestamp,
    cycle_end: pd.Timestamp,
    uuid_signal_info: pd.DataFrame,
    int_signal_info: pd.DataFrame,
    experiment: str,
    signal_cache: SessionSignalCache | None = None,
) -> dict[str, pd.DataFrame]:
    """Extract all available experiment signals for one cycle time interval.

    When ``signal_cache`` is provided (one cache per recording session, see
    :class:`src.utils.session_signal_cache.SessionSignalCache`), signals are
    sliced from the in-memory session cache instead of re-opening the
    Parquet dataset for every cycle. When omitted, behavior is unchanged:
    each signal is loaded directly from Parquet for this cycle only.
    """

    signal_descriptors = list_experiment_signals(
        uuid_signal_info=uuid_signal_info,
        int_signal_info=int_signal_info,
        experiment=experiment,
    )

    extracted_signals: dict[str, pd.DataFrame] = {}
    for row in signal_descriptors.itertuples(index=False):
        try:
            if row.source == "uuid":
                if pd.isna(row.signal_id_uuid):
                    logger.warning("Skipping UUID signal %s because signal_id_uuid is missing", row.signal_name)
                    continue
                if signal_cache is not None:
                    signal_df = signal_cache.slice_uuid_signal(
                        str(row.signal_id_uuid),
                        start_time=cycle_start,
                        end_time=cycle_end,
                    )
                else:
                    signal_df = load_uuid_signal(
                        base_dir,
                        str(row.signal_id_uuid),
                        start_time=cycle_start,
                        end_time=cycle_end,
                    )
            else:
                if pd.isna(row.signal_id):
                    logger.warning("Skipping INT signal %s because signal_id is missing", row.signal_name)
                    continue
                if signal_cache is not None:
                    signal_df = signal_cache.slice_int_signal(
                        int(row.signal_id),
                        start_time=cycle_start,
                        end_time=cycle_end,
                    )
                else:
                    signal_df = load_int_signal(
                        base_dir,
                        int(row.signal_id),
                        start_time=cycle_start,
                        end_time=cycle_end,
                    )
        except FileNotFoundError:
            logger.warning(
                "Skipping signal %s because its measurement dataset partition is missing",
                row.signal_name,
            )
            continue

        extracted_signals[str(row.signal_name)] = signal_df

    logger.info(
        "Extracted %d signal windows for experiment %s in interval [%s, %s)",
        len(extracted_signals),
        experiment,
        cycle_start,
        cycle_end,
    )
    return extracted_signals
