"""Vibration-aware, scalable candidate cycle selection.

This module implements the ``cycle_selection`` pipeline stage that runs
after ``cycle_detection`` and before ``multi_sensor_extraction``. It solves a
different problem than plain first-N cycle truncation: because the ESP32
vibration signals are duty-cycled (short bursts separated by long silent
gaps), a naive "first N Position cycles" selection can produce a pilot
dataset where almost none of the extracted cycles actually contain
vibration measurements.

Two modes are supported:

``first_n``
    Preserves the previous, simple behavior: the leading detected cycles are
    returned unchanged, with no signal-completeness scanning performed. This
    keeps the stage fast and behavior-compatible for callers that do not
    need multisensor completeness guarantees.

``complete_multisensor_stratified``
    Scans candidate cycles in bounded batches, checks the *actual* raw
    measurements inside each cycle's exact time window for every required
    signal (never relying only on a signal's global first/last timestamp),
    detects vibration recording bursts from real inter-sample gaps, and
    picks a deterministic, representative subset of complete cycles spread
    across sessions, time strata, and vibration bursts.

Scalability is a hard requirement: cycles are streamed from
``cycles.parquet`` in bounded batches (never loaded fully into memory line
by line), signal measurements are read once per batch as a single
time-windowed slice per required signal (predicate pushdown via
``load_uuid_signal``/``load_int_signal``), and every per-cycle metric is
computed with vectorized NumPy operations (``searchsorted`` on sorted
timestamps) instead of one measurement scan per cycle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.preprocessing.multi_sensor_cycle_extraction import list_experiment_signals
from src.utils.measurement_loader import load_int_signal, load_uuid_signal

logger = logging.getLogger(__name__)

DEFAULT_CYCLE_SELECTION_CONFIG: dict[str, object] = {
    "mode": "first_n",
    "target_cycle_count": 100,
    "candidate_batch_size": 500,
    "max_cycles_to_scan": 100_000,
    "distribute_across_sessions": True,
    "distribute_across_time": True,
    "time_strata_per_session": 5,
    "max_cycles_per_vibration_burst": 4,
    "vibration_burst_gap_seconds": 60.0,
    "random_seed": 42,
}

MODE_FIRST_N = "first_n"
MODE_COMPLETE_MULTISENSOR_STRATIFIED = "complete_multisensor_stratified"
VALID_MODES: tuple[str, ...] = (MODE_FIRST_N, MODE_COMPLETE_MULTISENSOR_STRATIFIED)

INVALID_CYCLE_INTERVAL = "invalid_cycle_interval"
NON_FINITE_REQUIRED_SIGNAL = "non_finite_required_signal"
BURST_SELECTION_LIMIT = "burst_selection_limit"

CYCLE_INDEX_COLUMNS: tuple[str, ...] = (
    "experiment",
    "session_id",
    "cycle_id",
    "start_time",
    "end_time",
    "duration_seconds",
)

SELECTED_CYCLES_BASE_COLUMNS: tuple[str, ...] = (
    "cycle_id",
    "session_id",
    "start_time",
    "end_time",
    "duration_seconds",
    "source_candidate_index",
    "vibration_burst_id",
    "time_stratum",
    "selection_rank",
)

CANDIDATE_EVALUATION_BASE_COLUMNS: tuple[str, ...] = (
    "cycle_id",
    "session_id",
    "start_time",
    "end_time",
    "eligible",
    "rejection_reasons",
    "vibration_burst_id",
    "source_candidate_index",
)


def missing_signal_code(signal_name: str) -> str:
    """Return the rejection code used when a required signal has zero samples."""

    return f"missing_{signal_name}"


def insufficient_signal_code(signal_name: str) -> str:
    """Return the rejection code used when a required signal is below its minimum."""

    return f"insufficient_{signal_name}_samples"


@dataclass(slots=True)
class CycleSelectionConfig:
    """Validated, defaulted configuration for the cycle_selection stage."""

    mode: str = MODE_FIRST_N
    target_cycle_count: int = 100
    candidate_batch_size: int = 500
    max_cycles_to_scan: int = 100_000
    distribute_across_sessions: bool = True
    distribute_across_time: bool = True
    time_strata_per_session: int = 5
    max_cycles_per_vibration_burst: int = 4
    vibration_burst_gap_seconds: float = 60.0
    random_seed: int = 42

    @classmethod
    def from_mapping(cls, values: dict[str, object] | None) -> "CycleSelectionConfig":
        """Build a config, filling any missing key with its documented default."""

        merged = dict(DEFAULT_CYCLE_SELECTION_CONFIG)
        if values:
            merged.update(values)
        mode = str(merged["mode"])
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid cycle_selection mode {mode!r}. Valid modes: {VALID_MODES}.")
        return cls(
            mode=mode,
            target_cycle_count=int(merged["target_cycle_count"]),
            candidate_batch_size=int(merged["candidate_batch_size"]),
            max_cycles_to_scan=int(merged["max_cycles_to_scan"]),
            distribute_across_sessions=bool(merged["distribute_across_sessions"]),
            distribute_across_time=bool(merged["distribute_across_time"]),
            time_strata_per_session=int(merged["time_strata_per_session"]),
            max_cycles_per_vibration_burst=int(merged["max_cycles_per_vibration_burst"]),
            vibration_burst_gap_seconds=float(merged["vibration_burst_gap_seconds"]),
            random_seed=int(merged["random_seed"]),
        )


@dataclass(slots=True)
class CycleSelectionResult:
    """Outcome of one cycle_selection run."""

    selected_cycles: pd.DataFrame = field(default_factory=pd.DataFrame)
    candidate_evaluation: pd.DataFrame = field(default_factory=pd.DataFrame)
    rejection_reason_counts: dict[str, int] = field(default_factory=dict)
    summary: dict[str, object] = field(default_factory=dict)


class _BurstTracker:
    """Assign monotonically increasing burst ids to a chronological time stream.

    A new burst starts whenever the gap since the previous seen timestamp
    exceeds ``gap_seconds``. State is kept across successive calls so bursts
    are detected correctly across batch boundaries without re-scanning
    earlier data.
    """

    def __init__(self, gap_seconds: float) -> None:
        self._gap_seconds = gap_seconds
        self._last_time: pd.Timestamp | None = None
        self._current_id: int = -1

    def assign(self, sorted_times: np.ndarray) -> np.ndarray:
        """Return one burst id per (already time-sorted) input timestamp."""

        if sorted_times.size == 0:
            return np.array([], dtype=np.int64)

        gaps_seconds = np.empty(sorted_times.size, dtype=np.float64)
        if self._last_time is not None:
            gaps_seconds[0] = (sorted_times[0] - np.datetime64(self._last_time)) / np.timedelta64(1, "s")
        else:
            gaps_seconds[0] = np.inf
        if sorted_times.size > 1:
            gaps_seconds[1:] = np.diff(sorted_times) / np.timedelta64(1, "s")

        new_burst_starts = gaps_seconds > self._gap_seconds
        burst_ids = self._current_id + np.cumsum(new_burst_starts.astype(np.int64))

        self._last_time = pd.Timestamp(sorted_times[-1])
        self._current_id = int(burst_ids[-1])
        return burst_ids


def _resolve_signal_descriptors(
    uuid_signal_info: pd.DataFrame,
    int_signal_info: pd.DataFrame,
    experiment: str,
    required_signal_names: tuple[str, ...],
) -> dict[str, object]:
    """Return one signal descriptor row per required signal name."""

    catalogue_df = list_experiment_signals(
        uuid_signal_info=uuid_signal_info,
        int_signal_info=int_signal_info,
        experiment=experiment,
    )
    descriptors: dict[str, object] = {}
    for row in catalogue_df.itertuples(index=False):
        if str(row.signal_name) in required_signal_names:
            descriptors[str(row.signal_name)] = row
    return descriptors


def _load_signal_window(
    base_dir: Path, descriptor: object, start_time: pd.Timestamp, end_time_exclusive: pd.Timestamp
) -> pd.DataFrame:
    """Load one required signal's raw samples for one batch's time window."""

    try:
        if descriptor.source == "uuid":
            if pd.isna(descriptor.signal_id_uuid):
                return pd.DataFrame(columns=["time", "value"])
            return load_uuid_signal(
                base_dir, str(descriptor.signal_id_uuid), start_time=start_time, end_time=end_time_exclusive
            )
        if pd.isna(descriptor.signal_id):
            return pd.DataFrame(columns=["time", "value"])
        return load_int_signal(
            base_dir, int(descriptor.signal_id), start_time=start_time, end_time=end_time_exclusive
        )
    except FileNotFoundError:
        logger.warning("Signal partition missing for %s; treating window as empty", descriptor.signal_name)
        return pd.DataFrame(columns=["time", "value"])


def _vectorized_window_counts(
    signal_df: pd.DataFrame, starts: np.ndarray, ends_exclusive: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized per-cycle (sample_count, finite_count) via sorted-time searchsorted.

    This reads the batch's signal slice exactly once and derives every
    cycle's counts with two ``searchsorted`` calls instead of one scan per
    cycle.
    """

    if signal_df.empty:
        zeros = np.zeros(starts.shape, dtype=np.int64)
        return zeros, zeros.copy()

    times = signal_df["time"].to_numpy()
    values = signal_df["value"].to_numpy(dtype=float)
    finite_cumsum = np.concatenate([[0], np.cumsum(np.isfinite(values))])

    left_indices = np.searchsorted(times, starts, side="left")
    right_indices = np.searchsorted(times, ends_exclusive, side="left")
    counts = np.maximum(right_indices - left_indices, 0).astype(np.int64)
    finite_counts = np.maximum(finite_cumsum[right_indices] - finite_cumsum[left_indices], 0)
    return counts, finite_counts.astype(np.int64)


def _vectorized_first_index_burst_id(
    signal_df: pd.DataFrame, burst_ids: np.ndarray, starts: np.ndarray, ends_exclusive: np.ndarray
) -> np.ndarray:
    """Return the burst id of the first in-window sample for each cycle, or -1."""

    if signal_df.empty:
        return np.full(starts.shape, -1, dtype=np.int64)

    times = signal_df["time"].to_numpy()
    left_indices = np.searchsorted(times, starts, side="left")
    right_indices = np.searchsorted(times, ends_exclusive, side="left")
    has_samples = right_indices > left_indices
    result = np.full(starts.shape, -1, dtype=np.int64)
    result[has_samples] = burst_ids[left_indices[has_samples]]
    return result


def _evaluate_batch(
    cycle_batch: pd.DataFrame,
    base_dir: Path,
    signal_descriptors: dict[str, object],
    required_signals: tuple[str, ...],
    minimum_samples: dict[str, int],
    burst_tracker: _BurstTracker,
    vibration_signal_names: tuple[str, ...],
) -> pd.DataFrame:
    """Evaluate one batch of candidate cycles against every required signal."""

    starts = pd.to_datetime(cycle_batch["start_time"]).to_numpy()
    ends = pd.to_datetime(cycle_batch["end_time"]).to_numpy()
    ends_exclusive = ends + np.timedelta64(1, "us")

    invalid_interval = starts >= ends

    window_start = pd.Timestamp(min(cycle_batch["start_time"].min(), cycle_batch["end_time"].min()))
    window_end = pd.Timestamp(
        max(cycle_batch["start_time"].max(), cycle_batch["end_time"].max())
    ) + pd.Timedelta(microseconds=1)

    metrics = pd.DataFrame(index=cycle_batch.index)
    rejection_lists: list[list[str]] = [
        ([INVALID_CYCLE_INTERVAL] if invalid else []) for invalid in invalid_interval
    ]

    reference_vibration_name = vibration_signal_names[0] if vibration_signal_names else None
    vibration_burst_ids = np.full(len(cycle_batch), -1, dtype=np.int64)

    for signal_name in required_signals:
        descriptor = signal_descriptors.get(signal_name)
        if descriptor is None:
            counts = np.zeros(len(cycle_batch), dtype=np.int64)
            finite_counts = counts.copy()
        else:
            signal_df = _load_signal_window(base_dir, descriptor, window_start, window_end)
            counts, finite_counts = _vectorized_window_counts(signal_df, starts, ends_exclusive)

            if signal_name == reference_vibration_name and not signal_df.empty:
                sorted_signal_df = signal_df.sort_values("time")
                signal_times = sorted_signal_df["time"].to_numpy()
                burst_ids_for_samples = burst_tracker.assign(signal_times)
                vibration_burst_ids = _vectorized_first_index_burst_id(
                    sorted_signal_df, burst_ids_for_samples, starts, ends_exclusive
                )
            elif signal_name == reference_vibration_name:
                # No samples at all in this batch window; still advance no
                # tracker state (nothing to advance) and leave burst ids at -1.
                pass

        metrics[f"{signal_name}_sample_count"] = counts
        metrics[f"{signal_name}_finite_count"] = finite_counts

        minimum_required = int(minimum_samples.get(signal_name, 1))
        for row_position in range(len(cycle_batch)):
            if invalid_interval[row_position]:
                continue
            sample_count = int(counts[row_position])
            finite_count = int(finite_counts[row_position])
            if sample_count == 0:
                rejection_lists[row_position].append(missing_signal_code(signal_name))
            elif sample_count < minimum_required:
                rejection_lists[row_position].append(insufficient_signal_code(signal_name))
            elif finite_count == 0:
                rejection_lists[row_position].append(NON_FINITE_REQUIRED_SIGNAL)

    metrics["vibration_burst_id"] = vibration_burst_ids
    metrics["rejection_reasons"] = [",".join(reasons) for reasons in rejection_lists]
    metrics["eligible"] = [len(reasons) == 0 for reasons in rejection_lists]

    evaluated_df = pd.concat(
        [cycle_batch[["cycle_id", "session_id", "start_time", "end_time", "duration_seconds"]].reset_index(drop=True), metrics.reset_index(drop=True)],
        axis=1,
    )
    return evaluated_df


def _iter_cycle_batches(cycles_parquet_path: Path, batch_size: int, max_cycles_to_scan: int):
    """Yield bounded-size cycle batches directly from the Parquet file without full materialization."""

    parquet_file = pq.ParquetFile(cycles_parquet_path)
    available_columns = set(parquet_file.schema_arrow.names)
    columns = [column for column in CYCLE_INDEX_COLUMNS if column in available_columns]

    scanned = 0
    for record_batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        if scanned >= max_cycles_to_scan:
            break
        batch_df = record_batch.to_pandas()
        if scanned + len(batch_df) > max_cycles_to_scan:
            batch_df = batch_df.iloc[: max_cycles_to_scan - scanned]
        scanned += len(batch_df)
        yield batch_df
        if scanned >= max_cycles_to_scan:
            break


def _assign_time_strata(
    eligible_df: pd.DataFrame, sessions_df: pd.DataFrame, time_strata_per_session: int
) -> pd.Series:
    """Assign each eligible cycle a 0-based time stratum within its session."""

    session_bounds = sessions_df.set_index("session_id")[["start_time", "end_time"]]
    strata = np.zeros(len(eligible_df), dtype=np.int64)

    for position, row in enumerate(eligible_df.itertuples(index=False)):
        session_id = int(row.session_id)
        if session_id not in session_bounds.index:
            strata[position] = 0
            continue
        session_start = pd.Timestamp(session_bounds.loc[session_id, "start_time"])
        session_end = pd.Timestamp(session_bounds.loc[session_id, "end_time"])
        session_span = (session_end - session_start).total_seconds()
        if session_span <= 0:
            strata[position] = 0
            continue
        fraction = (pd.Timestamp(row.start_time) - session_start).total_seconds() / session_span
        stratum = int(np.floor(fraction * time_strata_per_session))
        strata[position] = min(max(stratum, 0), time_strata_per_session - 1)

    return pd.Series(strata, index=eligible_df.index)


def _select_representative_cycles(
    eligible_df: pd.DataFrame,
    config: CycleSelectionConfig,
) -> tuple[pd.DataFrame, set[int]]:
    """Deterministically select a representative subset from eligible candidates.

    Cycles are grouped into (session, time_stratum) buckets, each shuffled
    with a seeded random generator, and picked in round-robin order across
    buckets so the final selection spreads across sessions and across
    early/middle/late time strata instead of clustering at the start of the
    experiment, while never exceeding ``max_cycles_per_vibration_burst`` per
    burst.
    """

    rng = np.random.default_rng(config.random_seed)

    buckets: dict[tuple[int, int], list[int]] = {}
    for (session_id, stratum), group_df in eligible_df.groupby(["session_id", "time_stratum"], sort=True):
        order = rng.permutation(len(group_df))
        buckets[(int(session_id), int(stratum))] = list(group_df.index.to_numpy()[order])

    bucket_keys = sorted(buckets.keys())
    burst_counts: dict[int, int] = {}
    burst_limited_indices: set[int] = set()
    selected_indices: list[int] = []
    selected_cycle_ids: set[int] = set()

    made_progress = True
    while len(selected_indices) < config.target_cycle_count and made_progress:
        made_progress = False
        for key in bucket_keys:
            if len(selected_indices) >= config.target_cycle_count:
                break
            bucket = buckets[key]
            while bucket:
                candidate_index = bucket.pop(0)
                cycle_id = int(eligible_df.loc[candidate_index, "cycle_id"])
                if cycle_id in selected_cycle_ids:
                    continue
                burst_id = int(eligible_df.loc[candidate_index, "vibration_burst_id"])
                if burst_id >= 0 and burst_counts.get(burst_id, 0) >= config.max_cycles_per_vibration_burst:
                    burst_limited_indices.add(candidate_index)
                    continue
                selected_indices.append(candidate_index)
                selected_cycle_ids.add(cycle_id)
                if burst_id >= 0:
                    burst_counts[burst_id] = burst_counts.get(burst_id, 0) + 1
                made_progress = True
                break

    selection_rank = {index: rank + 1 for rank, index in enumerate(selected_indices)}
    selected_df = eligible_df.loc[selected_indices].copy()
    selected_df["selection_rank"] = [selection_rank[index] for index in selected_indices]
    return selected_df, burst_limited_indices


def select_cycles(
    base_dir: Path,
    cycles_parquet_path: Path,
    sessions_df: pd.DataFrame,
    experiment: str,
    required_signals: tuple[str, ...],
    minimum_samples: dict[str, int],
    uuid_signal_info: pd.DataFrame,
    int_signal_info: pd.DataFrame,
    config: CycleSelectionConfig,
    vibration_signal_names: tuple[str, ...] = ("vibration_x", "vibration_y", "vibration_z"),
) -> CycleSelectionResult:
    """Scan candidate cycles and select a representative, complete subset.

    In :data:`MODE_FIRST_N`, the leading ``target_cycle_count`` cycles are
    returned unchanged with no per-signal scanning, preserving the original
    pipeline behavior. In
    :data:`MODE_COMPLETE_MULTISENSOR_STRATIFIED`, every scanned candidate is
    checked against the real measurements inside its own time window.
    """

    if config.mode == MODE_FIRST_N:
        return _select_first_n(cycles_parquet_path, config)

    return _select_complete_multisensor_stratified(
        base_dir=base_dir,
        cycles_parquet_path=cycles_parquet_path,
        sessions_df=sessions_df,
        experiment=experiment,
        required_signals=required_signals,
        minimum_samples=minimum_samples,
        uuid_signal_info=uuid_signal_info,
        int_signal_info=int_signal_info,
        config=config,
        vibration_signal_names=vibration_signal_names,
    )


def _select_first_n(cycles_parquet_path: Path, config: CycleSelectionConfig) -> CycleSelectionResult:
    """Fast pass-through selection mirroring the previous first-N behavior."""

    parquet_file = pq.ParquetFile(cycles_parquet_path)
    available_columns = set(parquet_file.schema_arrow.names)
    columns = [column for column in CYCLE_INDEX_COLUMNS if column in available_columns]

    collected: list[pd.DataFrame] = []
    scanned = 0
    for record_batch in parquet_file.iter_batches(batch_size=config.candidate_batch_size, columns=columns):
        batch_df = record_batch.to_pandas()
        collected.append(batch_df)
        scanned += len(batch_df)
        if scanned >= config.target_cycle_count:
            break

    cycles_df = pd.concat(collected, ignore_index=True) if collected else pd.DataFrame(columns=columns)
    selected_df = cycles_df.head(config.target_cycle_count).copy().reset_index(drop=True)
    selected_df["source_candidate_index"] = np.arange(len(selected_df))
    selected_df["vibration_burst_id"] = -1
    selected_df["time_stratum"] = 0
    selected_df["selection_rank"] = np.arange(1, len(selected_df) + 1)

    candidate_evaluation_df = selected_df[["cycle_id", "session_id", "start_time", "end_time"]].copy()
    candidate_evaluation_df["eligible"] = True
    candidate_evaluation_df["rejection_reasons"] = ""
    candidate_evaluation_df["vibration_burst_id"] = -1
    candidate_evaluation_df["source_candidate_index"] = selected_df["source_candidate_index"]

    summary = {
        "selection_mode": MODE_FIRST_N,
        "target_cycle_count": config.target_cycle_count,
        "candidates_scanned": int(scanned),
        "eligible_cycles_found": int(len(selected_df)),
        "selected_cycles": int(len(selected_df)),
        "selection_shortfall": max(0, config.target_cycle_count - len(selected_df)),
    }
    if len(selected_df) < config.target_cycle_count:
        logger.warning(
            "cycle_selection (first_n): only %d of %d requested cycles were available",
            len(selected_df),
            config.target_cycle_count,
        )

    return CycleSelectionResult(
        selected_cycles=selected_df,
        candidate_evaluation=candidate_evaluation_df,
        rejection_reason_counts={},
        summary=summary,
    )


def _select_complete_multisensor_stratified(
    base_dir: Path,
    cycles_parquet_path: Path,
    sessions_df: pd.DataFrame,
    experiment: str,
    required_signals: tuple[str, ...],
    minimum_samples: dict[str, int],
    uuid_signal_info: pd.DataFrame,
    int_signal_info: pd.DataFrame,
    config: CycleSelectionConfig,
    vibration_signal_names: tuple[str, ...],
) -> CycleSelectionResult:
    """Full vibration-aware, scalable, batch-scanned candidate selection."""

    signal_descriptors = _resolve_signal_descriptors(
        uuid_signal_info, int_signal_info, experiment, required_signals
    )
    burst_tracker = _BurstTracker(gap_seconds=config.vibration_burst_gap_seconds)

    evaluated_batches: list[pd.DataFrame] = []
    candidates_scanned = 0
    next_candidate_index = 0

    for cycle_batch in _iter_cycle_batches(
        cycles_parquet_path, config.candidate_batch_size, config.max_cycles_to_scan
    ):
        if cycle_batch.empty:
            continue
        evaluated_df = _evaluate_batch(
            cycle_batch=cycle_batch,
            base_dir=base_dir,
            signal_descriptors=signal_descriptors,
            required_signals=required_signals,
            minimum_samples=minimum_samples,
            burst_tracker=burst_tracker,
            vibration_signal_names=vibration_signal_names,
        )
        evaluated_df["source_candidate_index"] = np.arange(
            next_candidate_index, next_candidate_index + len(evaluated_df)
        )
        next_candidate_index += len(evaluated_df)
        candidates_scanned += len(cycle_batch)
        evaluated_batches.append(evaluated_df)
        logger.info(
            "cycle_selection: evaluated %d candidate cycle(s) so far (%d eligible)",
            candidates_scanned,
            int(evaluated_df["eligible"].sum()) + sum(int(b["eligible"].sum()) for b in evaluated_batches[:-1]),
        )

    candidate_evaluation_df = (
        pd.concat(evaluated_batches, ignore_index=True)
        if evaluated_batches
        else pd.DataFrame(columns=list(CANDIDATE_EVALUATION_BASE_COLUMNS))
    )

    eligible_df = candidate_evaluation_df[candidate_evaluation_df["eligible"]].copy().reset_index(drop=True)
    eligible_cycles_found = len(eligible_df)

    if eligible_cycles_found == 0:
        logger.warning("cycle_selection: no eligible complete multisensor cycles were found")
        summary = _build_summary(
            config=config,
            candidates_scanned=candidates_scanned,
            eligible_cycles_found=0,
            selected_df=pd.DataFrame(),
            candidate_evaluation_df=candidate_evaluation_df,
            required_signals=required_signals,
            minimum_samples=minimum_samples,
        )
        return CycleSelectionResult(
            selected_cycles=pd.DataFrame(columns=list(SELECTED_CYCLES_BASE_COLUMNS)),
            candidate_evaluation=candidate_evaluation_df,
            rejection_reason_counts=_count_rejection_reasons(candidate_evaluation_df),
            summary=summary,
        )

    eligible_df["time_stratum"] = _assign_time_strata(
        eligible_df, sessions_df, config.time_strata_per_session
    ).to_numpy()

    selected_df, burst_limited_indices = _select_representative_cycles(eligible_df, config)

    if burst_limited_indices:
        limited_cycle_ids = set(eligible_df.loc[list(burst_limited_indices), "cycle_id"].astype(int))
        limit_mask = candidate_evaluation_df["cycle_id"].astype(int).isin(limited_cycle_ids)
        candidate_evaluation_df.loc[limit_mask, "rejection_reasons"] = candidate_evaluation_df.loc[
            limit_mask, "rejection_reasons"
        ].apply(
            lambda reasons: ",".join(filter(None, [reasons, BURST_SELECTION_LIMIT]))
        )

    selection_shortfall = max(0, config.target_cycle_count - len(selected_df))
    if selection_shortfall > 0:
        logger.warning(
            "cycle_selection: only %d of %d requested representative cycles were found (shortfall=%d)",
            len(selected_df),
            config.target_cycle_count,
            selection_shortfall,
        )

    keep_columns = [column for column in SELECTED_CYCLES_BASE_COLUMNS] + [
        f"{name}_sample_count" for name in required_signals
    ]
    for column in keep_columns:
        if column not in selected_df.columns:
            selected_df[column] = pd.NA
    selected_df = selected_df[keep_columns].sort_values("selection_rank").reset_index(drop=True)

    summary = _build_summary(
        config=config,
        candidates_scanned=candidates_scanned,
        eligible_cycles_found=eligible_cycles_found,
        selected_df=selected_df,
        candidate_evaluation_df=candidate_evaluation_df,
        required_signals=required_signals,
        minimum_samples=minimum_samples,
    )

    return CycleSelectionResult(
        selected_cycles=selected_df,
        candidate_evaluation=candidate_evaluation_df,
        rejection_reason_counts=_count_rejection_reasons(candidate_evaluation_df),
        summary=summary,
    )


def _count_rejection_reasons(candidate_evaluation_df: pd.DataFrame) -> dict[str, int]:
    """Count how many times each rejection code appears across all candidates."""

    counts: dict[str, int] = {}
    if candidate_evaluation_df.empty or "rejection_reasons" not in candidate_evaluation_df.columns:
        return counts
    for reasons_text in candidate_evaluation_df["rejection_reasons"]:
        if not reasons_text:
            continue
        for code in str(reasons_text).split(","):
            if code:
                counts[code] = counts.get(code, 0) + 1
    return counts


def _build_summary(
    config: CycleSelectionConfig,
    candidates_scanned: int,
    eligible_cycles_found: int,
    selected_df: pd.DataFrame,
    candidate_evaluation_df: pd.DataFrame,
    required_signals: tuple[str, ...],
    minimum_samples: dict[str, int],
) -> dict[str, object]:
    """Build the JSON-safe selection summary payload."""

    selected_cycles_per_session = (
        selected_df.groupby("session_id").size().to_dict() if not selected_df.empty else {}
    )
    selected_cycles_per_time_stratum = (
        selected_df.groupby("time_stratum").size().to_dict() if not selected_df.empty else {}
    )
    selected_cycles_per_vibration_burst = (
        selected_df.groupby("vibration_burst_id").size().to_dict() if not selected_df.empty else {}
    )

    return {
        "selection_mode": config.mode,
        "target_cycle_count": config.target_cycle_count,
        "candidates_scanned": int(candidates_scanned),
        "eligible_cycles_found": int(eligible_cycles_found),
        "selected_cycles": int(len(selected_df)),
        "selection_shortfall": max(0, config.target_cycle_count - len(selected_df)),
        "selected_cycles_per_session": {str(k): int(v) for k, v in selected_cycles_per_session.items()},
        "selected_cycles_per_time_stratum": {
            str(k): int(v) for k, v in selected_cycles_per_time_stratum.items()
        },
        "selected_cycles_per_vibration_burst": {
            str(k): int(v) for k, v in selected_cycles_per_vibration_burst.items()
        },
        "rejection_reason_counts": _count_rejection_reasons(candidate_evaluation_df),
        "minimum_sample_requirements": {name: int(minimum_samples.get(name, 1)) for name in required_signals},
        "first_selected_cycle_time": (
            str(selected_df["start_time"].min()) if not selected_df.empty else None
        ),
        "last_selected_cycle_time": (
            str(selected_df["end_time"].max()) if not selected_df.empty else None
        ),
    }
