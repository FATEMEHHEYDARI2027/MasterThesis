"""Select a small, high-quality subset of cycles for interactive validation."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.preprocessing.multi_sensor_cycle_extraction import extract_cycle_measurements

logger = logging.getLogger(__name__)

QUALITY_TABLE_BASE_COLUMNS: tuple[str, ...] = (
    "experiment",
    "session_id",
    "cycle_id",
    "start_time",
    "end_time",
    "is_validation_ready",
    "missing_required_signals",
    "insufficient_required_signals",
    "rejection_reason",
)


@dataclass(slots=True)
class ValidationSelectionResult:
    """Outcome of scanning cycles for a small multi-sensor validation subset."""

    selected_cycles: pd.DataFrame
    extracted_signals_by_cycle: dict[int, dict[str, pd.DataFrame]] = field(default_factory=dict)
    quality_table: pd.DataFrame = field(default_factory=pd.DataFrame)


def _evaluate_cycle_quality(
    cycle_id: int,
    session_id: int | None,
    experiment: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    extracted_signals: dict[str, pd.DataFrame],
    required_signals: Sequence[str],
    minimum_samples: Mapping[str, int],
) -> dict[str, object]:
    """Build one quality-table row and report whether the cycle is usable."""

    missing_signals: list[str] = []
    insufficient_signals: list[str] = []
    sample_counts: dict[str, int] = {}

    for signal_name in required_signals:
        signal_df = extracted_signals.get(signal_name)
        sample_count = 0 if signal_df is None else int(len(signal_df))
        sample_counts[f"{signal_name}_sample_count"] = sample_count

        if signal_df is None or signal_df.empty:
            missing_signals.append(signal_name)
            continue

        required_minimum = int(minimum_samples.get(signal_name, 1))
        if sample_count < required_minimum:
            insufficient_signals.append(signal_name)

    is_ready = not missing_signals and not insufficient_signals
    if is_ready:
        rejection_reason = ""
    elif missing_signals and insufficient_signals:
        rejection_reason = "missing and insufficient required signals"
    elif missing_signals:
        rejection_reason = "missing required signals"
    else:
        rejection_reason = "insufficient samples for required signals"

    row: dict[str, object] = {
        "experiment": experiment,
        "session_id": session_id,
        "cycle_id": cycle_id,
        "start_time": start_time,
        "end_time": end_time,
        "is_validation_ready": is_ready,
        "missing_required_signals": ",".join(missing_signals),
        "insufficient_required_signals": ",".join(insufficient_signals),
        "rejection_reason": rejection_reason,
    }
    row.update(sample_counts)
    return row


def select_validation_cycles(
    base_dir: Path,
    cycles_df: pd.DataFrame,
    uuid_signal_info: pd.DataFrame,
    int_signal_info: pd.DataFrame,
    experiment: str,
    required_signals: Sequence[str],
    minimum_samples: Mapping[str, int],
    validation_cycle_count: int,
    require_consecutive: bool = True,
    max_cycles_to_scan: int | None = 10_000,
) -> ValidationSelectionResult:
    """Scan cycles chronologically and select the earliest valid block for validation.

    Cycles are inspected in chronological order (grouped by session, then by
    start time) using :func:`extract_cycle_measurements`. A cycle is
    "validation ready" when every configured required signal exists and has
    at least its configured minimum sample count; missing data is never
    interpolated or filled. When ``require_consecutive`` is ``True``, one
    failed cycle resets the candidate block (the block must be back-to-back
    within cycles_df, which never crosses a session boundary because cycles
    are grouped by session before scanning). Already-extracted signals for
    scanned cycles are kept and returned so callers do not need to
    re-extract them when generating validation HTML files.
    """

    if validation_cycle_count <= 0:
        raise ValueError("validation_cycle_count must be positive.")
    if cycles_df.empty:
        raise ValueError("No cycles are available to select a validation subset from.")

    sort_columns = [column for column in ("session_id", "start_time", "cycle_id") if column in cycles_df.columns]
    ordered_cycles_df = cycles_df.sort_values(sort_columns).reset_index(drop=True) if sort_columns else cycles_df

    quality_rows: list[dict[str, object]] = []
    extracted_signals_by_cycle: dict[int, dict[str, pd.DataFrame]] = {}

    candidate_cycle_ids: list[int] = []
    candidate_session_id: int | None = None
    selected_cycle_ids: list[int] | None = None
    scanned_count = 0

    for cycle_row in ordered_cycles_df.itertuples(index=False):
        if max_cycles_to_scan is not None and scanned_count >= max_cycles_to_scan:
            break
        scanned_count += 1

        cycle_id = int(cycle_row.cycle_id)
        session_id = int(cycle_row.session_id) if hasattr(cycle_row, "session_id") else None
        start_time = pd.Timestamp(cycle_row.start_time)
        end_time = pd.Timestamp(cycle_row.end_time)

        extracted_signals = extract_cycle_measurements(
            base_dir=base_dir,
            cycle_start=start_time,
            cycle_end=end_time + pd.Timedelta(microseconds=1),
            uuid_signal_info=uuid_signal_info,
            int_signal_info=int_signal_info,
            experiment=experiment,
        )
        extracted_signals_by_cycle[cycle_id] = extracted_signals

        quality_row = _evaluate_cycle_quality(
            cycle_id=cycle_id,
            session_id=session_id,
            experiment=experiment,
            start_time=start_time,
            end_time=end_time,
            extracted_signals=extracted_signals,
            required_signals=required_signals,
            minimum_samples=minimum_samples,
        )
        quality_rows.append(quality_row)

        if selected_cycle_ids is not None:
            continue

        is_ready = bool(quality_row["is_validation_ready"])
        if not is_ready:
            if require_consecutive:
                candidate_cycle_ids = []
                candidate_session_id = None
            continue

        if require_consecutive:
            if candidate_session_id is not None and candidate_session_id != session_id:
                candidate_cycle_ids = []
            candidate_session_id = session_id
            candidate_cycle_ids.append(cycle_id)
        else:
            candidate_cycle_ids.append(cycle_id)

        if len(candidate_cycle_ids) >= validation_cycle_count:
            selected_cycle_ids = candidate_cycle_ids[:validation_cycle_count]
            break

    quality_table = pd.DataFrame(quality_rows)

    if selected_cycle_ids is None:
        raise ValueError(
            "No valid multi-sensor validation block of "
            f"{validation_cycle_count} cycle(s) was found for experiment "
            f"{experiment!r} within the first {scanned_count} scanned cycle(s). "
            "Inspect the validation quality table for missing or insufficient signals."
        )

    selected_cycles = ordered_cycles_df[
        ordered_cycles_df["cycle_id"].isin(selected_cycle_ids)
    ].sort_values("cycle_id").reset_index(drop=True)

    selected_signals_by_cycle = {
        cycle_id: extracted_signals_by_cycle[cycle_id] for cycle_id in selected_cycle_ids
    }

    logger.info(
        "Selected %d validation cycle(s) for experiment %s after scanning %d cycle(s)",
        len(selected_cycle_ids),
        experiment,
        scanned_count,
    )

    return ValidationSelectionResult(
        selected_cycles=selected_cycles,
        extracted_signals_by_cycle=selected_signals_by_cycle,
        quality_table=quality_table,
    )
