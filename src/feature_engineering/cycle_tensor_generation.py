"""Fixed-length cycle tensor dataset generation (pipeline stage 10.1).

This module implements the ``feature_engineering`` pipeline stage. It runs
**after** ``dataset_validation`` and turns every ``valid_core_cycle`` into a
fixed-length, machine-learning-ready sample using the thesis' **padding**
methodology -- it never resamples, interpolates, or otherwise fabricates
values.

Methodology
-----------
Unlike a resampling/interpolation approach, this stage keeps every original
measured sample exactly as extracted:

1. The length of one cycle is defined as the number of raw samples of the
   configured ``reference_signal`` (``position`` by default -- the signal
   ``cycle_detection`` uses to delimit cycles in the first place).
2. A single ``target_length`` is derived once, up front, from the observed
   distribution of cycle lengths across *every* ``valid_core_cycle``
   (``max`` of all lengths, or a configurable percentile -- ``99`` by
   default). This mirrors the thesis' analysis of maximum/95th/99th
   percentile cycle length.
3. For every cycle and every required signal, the raw, native-rate value
   sequence (sorted by timestamp, otherwise untouched) is padded at the
   **end** with its own last value (edge padding) if shorter than
   ``target_length``, or truncated at the **end** if longer. No signal is
   ever resampled onto another signal's timeline.
4. A padding mask (``1`` = real sample, ``0`` = padding) is derived
   **independently for every required signal**, since each signal is padded
   or truncated to ``target_length`` on its own, native-rate timeline. The
   mask therefore has the same shape as the tensor itself
   (``target_length x number_of_signals`` for one cycle,
   ``batch_size x target_length x number_of_signals`` for one batch) and is
   stored alongside every tensor batch, so that padded time steps can be
   masked out downstream (e.g. by a loss function or a sequence model), on a
   per-signal basis.

Only the ``cycle_id`` column is read from ``valid_core_cycles.parquet`` --
this stage never re-derives or re-checks validity, it only consumes the
frozen decision made by ``dataset_validation``. Cycle timing (``start_time``
/ ``end_time``) is taken from the cycle index produced by
``cycle_detection``, and raw measurements are taken from the Parquet
measurement store produced by ``multi_sensor_extraction``. None of those
upstream stages are read from in a way that could mutate their outputs, and
none of their modules are modified by this stage.

Cycles are processed and written in bounded batches so memory usage stays
independent of the total number of cycles, matching the batching strategy
used throughout the rest of the pipeline (see
:mod:`src.preprocessing.batch_extraction`).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from src.preprocessing.batch_extraction import iter_cycle_batches

logger = logging.getLogger(__name__)

DEFAULT_REQUIRED_SIGNALS: tuple[str, ...] = (
    "position",
    "velocity",
    "current",
    "pressure",
    "temperature",
)

DEFAULT_REFERENCE_SIGNAL = "position"

DEFAULT_CYCLE_TENSOR_GENERATION_CONFIG: dict[str, object] = {
    "enabled": True,
    "required_signals": list(DEFAULT_REQUIRED_SIGNALS),
    "reference_signal": DEFAULT_REFERENCE_SIGNAL,
    "target_length_strategy": "percentile",
    "target_length_percentile": 99,
    "padding_method": "edge",
    "truncate_long_cycles": True,
    "save_padding_mask": True,
    "output_format": "npy",
    "cycles_per_file": 64,
}

SUPPORTED_TARGET_LENGTH_STRATEGIES: tuple[str, ...] = ("max", "percentile")
SUPPORTED_PADDING_METHODS: tuple[str, ...] = ("edge",)
SUPPORTED_OUTPUT_FORMATS: tuple[str, ...] = ("npy",)

METADATA_COLUMNS: tuple[str, ...] = (
    "cycle_id",
    "session_id",
    "cycle_start",
    "cycle_end",
    "cycle_duration_seconds",
    "original_cycle_length",
    "target_length",
    "padded_samples",
    "truncated_samples",
    "signal_original_lengths",
    "batch_file",
)

CYCLE_TENSOR_METADATA_FILE_NAME = "cycle_tensor_metadata.parquet"
SKIPPED_CYCLES_FILE_NAME = "skipped_cycles.parquet"
SUMMARY_FILE_NAME = "cycle_tensor_generation_summary.json"
STATISTICS_FILE_NAME = "cycle_length_statistics.json"
MASK_FILE_SUFFIX = "_mask"


class MissingRequiredSignalError(ValueError):
    """Raised when a cycle is missing one of the configured required signals."""


@dataclass(slots=True)
class CycleTensorGenerationConfig:
    """Validated configuration for the ``cycle_tensor_generation`` stage."""

    enabled: bool = True
    required_signals: tuple[str, ...] = DEFAULT_REQUIRED_SIGNALS
    reference_signal: str = DEFAULT_REFERENCE_SIGNAL
    target_length_strategy: str = "percentile"
    target_length_percentile: float = 99
    padding_method: str = "edge"
    truncate_long_cycles: bool = True
    save_padding_mask: bool = True
    output_format: str = "npy"
    cycles_per_file: int = 64

    @classmethod
    def from_mapping(
        cls, mapping: dict[str, object] | None
    ) -> "CycleTensorGenerationConfig":
        merged = dict(DEFAULT_CYCLE_TENSOR_GENERATION_CONFIG)
        if mapping:
            merged.update(mapping)

        required_signals = tuple(str(name) for name in merged["required_signals"])
        reference_signal = str(merged["reference_signal"])
        target_length_strategy = str(merged["target_length_strategy"])
        target_length_percentile = float(merged["target_length_percentile"])
        padding_method = str(merged["padding_method"])
        output_format = str(merged["output_format"])
        cycles_per_file = int(merged["cycles_per_file"])

        if not required_signals:
            raise ValueError("required_signals must not be empty.")
        if reference_signal not in required_signals:
            raise ValueError(
                f"reference_signal {reference_signal!r} must be one of required_signals "
                f"{required_signals!r}."
            )
        if target_length_strategy not in SUPPORTED_TARGET_LENGTH_STRATEGIES:
            raise ValueError(
                "Unsupported target_length_strategy "
                f"{target_length_strategy!r}. Supported: {SUPPORTED_TARGET_LENGTH_STRATEGIES}."
            )
        if not (0 < target_length_percentile <= 100):
            raise ValueError("target_length_percentile must be in (0, 100].")
        if padding_method not in SUPPORTED_PADDING_METHODS:
            raise ValueError(
                f"Unsupported padding_method {padding_method!r}. "
                f"Supported: {SUPPORTED_PADDING_METHODS}."
            )
        if output_format not in SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(
                f"Unsupported output_format {output_format!r}. "
                f"Supported: {SUPPORTED_OUTPUT_FORMATS}."
            )
        if cycles_per_file <= 0:
            raise ValueError("cycles_per_file must be positive.")

        return cls(
            enabled=bool(merged["enabled"]),
            required_signals=required_signals,
            reference_signal=reference_signal,
            target_length_strategy=target_length_strategy,
            target_length_percentile=target_length_percentile,
            padding_method=padding_method,
            truncate_long_cycles=bool(merged["truncate_long_cycles"]),
            save_padding_mask=bool(merged["save_padding_mask"]),
            output_format=output_format,
            cycles_per_file=cycles_per_file,
        )


@dataclass(slots=True)
class CycleTensorGenerationResult:
    """Outcome of one cycle tensor generation run."""

    metadata: pd.DataFrame = field(default_factory=pd.DataFrame)
    skipped_cycles: pd.DataFrame = field(default_factory=pd.DataFrame)
    written_files: list[Path] = field(default_factory=list)
    mask_files: list[Path] = field(default_factory=list)
    length_statistics: dict[str, object] = field(default_factory=dict)
    summary: dict[str, object] = field(default_factory=dict)


def _resolve_time_column(measurements_df: pd.DataFrame) -> str:
    """Return the column holding sample timestamps for one measurement frame."""

    if "time" in measurements_df.columns:
        return "time"
    if "timestamp" in measurements_df.columns:
        return "timestamp"
    raise KeyError("Measurement frame has neither a 'time' nor a 'timestamp' column.")


def _sorted_signal_values(cycle_measurements: pd.DataFrame, signal_name: str) -> np.ndarray:
    """Return one signal's raw values for one cycle, sorted by timestamp.

    Values are returned exactly as measured -- no resampling, no
    interpolation, no fabricated samples.
    """

    signal_rows = (
        cycle_measurements[cycle_measurements["signal_name"] == signal_name]
        if not cycle_measurements.empty
        else cycle_measurements
    )
    if signal_rows.empty:
        raise MissingRequiredSignalError(
            f"Required signal {signal_name!r} has no samples for this cycle."
        )

    time_column = _resolve_time_column(signal_rows)
    ordered_rows = signal_rows.sort_values(time_column, kind="stable")
    return pd.to_numeric(ordered_rows["value"], errors="coerce").to_numpy(dtype=np.float64)


def compute_cycle_lengths(measurements_df: pd.DataFrame, reference_signal: str) -> pd.Series:
    """Count the raw ``reference_signal`` samples available for each cycle.

    ``measurements_df`` is a long-format frame with at least ``cycle_id`` and
    ``signal_name`` columns. Returns a ``cycle_id``-indexed ``Series`` of
    integer sample counts; cycles absent from ``measurements_df`` simply do
    not appear in the result.
    """

    if measurements_df.empty:
        return pd.Series(dtype=int, name="cycle_length")

    reference_rows = measurements_df[measurements_df["signal_name"] == reference_signal]
    if reference_rows.empty:
        return pd.Series(dtype=int, name="cycle_length")

    lengths = reference_rows.groupby("cycle_id").size()
    lengths.name = "cycle_length"
    return lengths.astype(int)


def determine_target_length(
    cycle_lengths: pd.Series, strategy: str, percentile: float
) -> int:
    """Derive one shared ``target_length`` from the observed cycle lengths.

    ``strategy="max"`` uses the longest observed cycle so no cycle is ever
    truncated. ``strategy="percentile"`` uses the requested percentile (the
    thesis default is the 99th percentile), trading a small, bounded amount
    of truncation for a shorter, more memory-efficient target length.
    """

    if cycle_lengths.empty:
        raise ValueError("Cannot determine target_length from an empty set of cycle lengths.")

    if strategy == "max":
        target_length = int(cycle_lengths.max())
    elif strategy == "percentile":
        target_length = int(np.ceil(np.percentile(cycle_lengths.to_numpy(), percentile)))
    else:
        raise ValueError(
            f"Unsupported strategy {strategy!r}. Supported: {SUPPORTED_TARGET_LENGTH_STRATEGIES}."
        )

    return max(target_length, 1)


def build_cycle_length_statistics(
    cycle_lengths: pd.Series,
    target_length: int,
    strategy: str,
    percentile: float,
) -> dict[str, object]:
    """Summarize the observed cycle-length distribution and the chosen target."""

    if cycle_lengths.empty:
        return {
            "cycles_considered": 0,
            "minimum_cycle_length": None,
            "maximum_cycle_length": None,
            "mean_cycle_length": None,
            "median_cycle_length": None,
            "p95_cycle_length": None,
            "p99_cycle_length": None,
            "target_length_strategy": strategy,
            "target_length_percentile": percentile,
            "selected_target_length": target_length,
            "number_of_padded_cycles": 0,
            "number_of_truncated_cycles": 0,
        }

    values = cycle_lengths.to_numpy()
    return {
        "cycles_considered": int(values.size),
        "minimum_cycle_length": int(values.min()),
        "maximum_cycle_length": int(values.max()),
        "mean_cycle_length": float(values.mean()),
        "median_cycle_length": float(np.median(values)),
        "p95_cycle_length": float(np.percentile(values, 95)),
        "p99_cycle_length": float(np.percentile(values, 99)),
        "target_length_strategy": strategy,
        "target_length_percentile": percentile,
        "selected_target_length": target_length,
        "number_of_padded_cycles": int(np.sum(values < target_length)),
        "number_of_truncated_cycles": int(np.sum(values > target_length)),
    }


def pad_signal(
    values: np.ndarray, target_length: int, padding_method: str = "edge"
) -> tuple[np.ndarray, int, int]:
    """Pad or truncate one signal's raw values to ``target_length``.

    Padding is only ever applied at the **end** of the sequence, by
    repeating the last real value (edge padding) -- zero padding is never
    used. Truncation, when needed, also only removes samples from the end.
    Returns ``(resized_values, padded_samples, truncated_samples)``.
    """

    if padding_method not in SUPPORTED_PADDING_METHODS:
        raise ValueError(
            f"Unsupported padding_method {padding_method!r}. "
            f"Supported: {SUPPORTED_PADDING_METHODS}."
        )
    if values.size == 0:
        raise ValueError("Cannot pad an empty signal.")

    original_length = values.size
    if original_length == target_length:
        return values.copy(), 0, 0
    if original_length < target_length:
        padded_count = target_length - original_length
        # padding_method == "edge": repeat the last real value.
        padded_values = np.pad(values, (0, padded_count), mode="edge")
        return padded_values, padded_count, 0

    truncated_count = original_length - target_length
    return values[:target_length].copy(), 0, truncated_count


def build_padding_mask(original_length: int, target_length: int) -> np.ndarray:
    """Build one padding mask: ``1`` for real samples, ``0`` for padding.

    A truncated cycle (``original_length > target_length``) is all real
    samples within ``target_length`` -- truncation removes samples, it never
    introduces padding.
    """

    mask = np.zeros(target_length, dtype=np.int8)
    real_sample_count = min(original_length, target_length)
    mask[:real_sample_count] = 1
    return mask


def build_cycle_tensor(
    cycle_measurements: pd.DataFrame,
    required_signals: tuple[str, ...],
    reference_signal: str,
    target_length: int,
    padding_method: str = "edge",
) -> tuple[np.ndarray, np.ndarray, int, int, int, dict[str, int]]:
    """Build one ``target_length x number_of_signals`` matrix for one cycle.

    Every required signal keeps its own original, measured values -- each
    column is independently padded/truncated to ``target_length``, never
    resampled onto another signal's timeline. Columns are emitted in the
    exact order of ``required_signals``.

    The returned mask has the same ``target_length x number_of_signals``
    shape as the matrix: because every signal is padded or truncated
    independently, each mask column is derived from that signal's own
    original length, not just the reference signal's.

    Returns ``(matrix, mask, original_cycle_length, padded_samples,
    truncated_samples, signal_original_lengths)``, where
    ``original_cycle_length``/``padded_samples``/``truncated_samples``
    describe the cycle as measured by ``reference_signal`` (the signal used
    to define cycle length), and ``signal_original_lengths`` maps every
    required signal name to its own original (pre-padding/truncation) sample
    count for this cycle.
    Raises :class:`MissingRequiredSignalError` if any required signal has no
    samples for this cycle.
    """

    matrix = np.empty((target_length, len(required_signals)), dtype=np.float64)
    mask = np.empty((target_length, len(required_signals)), dtype=np.int8)
    signal_original_lengths: dict[str, int] = {}
    reference_original_length: int | None = None
    reference_padded_samples = 0
    reference_truncated_samples = 0

    for column_index, signal_name in enumerate(required_signals):
        values = _sorted_signal_values(cycle_measurements, signal_name)
        resized_values, padded_samples, truncated_samples = pad_signal(
            values, target_length, padding_method
        )
        matrix[:, column_index] = resized_values
        mask[:, column_index] = build_padding_mask(values.size, target_length)
        signal_original_lengths[signal_name] = int(values.size)

        if signal_name == reference_signal:
            reference_original_length = values.size
            reference_padded_samples = padded_samples
            reference_truncated_samples = truncated_samples

    if reference_original_length is None:
        # reference_signal is validated to be part of required_signals, so
        # this only happens if that invariant is violated by a caller.
        raise MissingRequiredSignalError(
            f"Reference signal {reference_signal!r} has no samples for this cycle."
        )

    return (
        matrix,
        mask,
        reference_original_length,
        reference_padded_samples,
        reference_truncated_samples,
        signal_original_lengths,
    )


def _batch_file_name(
    dataset_name: str, experiment: str, start_position: int, end_position: int, output_format: str
) -> str:
    """Build one batch file name, e.g. ``D63_Nr7_Versuch1_cycles_000001_000064.npy``."""

    prefix_parts = [part for part in (dataset_name, experiment) if part]
    prefix = "_".join(prefix_parts) if prefix_parts else "cycles"
    return f"{prefix}_cycles_{start_position:06d}_{end_position:06d}.{output_format}"


def write_tensor_batch(
    batch_tensor: np.ndarray,
    batch_mask: np.ndarray | None,
    output_directory: Path,
    file_name: str,
) -> tuple[Path, Path | None]:
    """Write one tensor batch (and its optional padding mask) to disk.

    Returns ``(tensor_path, mask_path)``; ``mask_path`` is ``None`` when
    ``batch_mask`` is ``None`` (i.e. mask generation was disabled).
    """

    output_directory = Path(output_directory)
    tensor_path = output_directory / file_name
    np.save(tensor_path, batch_tensor)

    mask_path: Path | None = None
    if batch_mask is not None:
        mask_file_name = tensor_path.stem + MASK_FILE_SUFFIX + tensor_path.suffix
        mask_path = output_directory / mask_file_name
        np.save(mask_path, batch_mask)

    return tensor_path, mask_path


def _load_cycle_ids(valid_core_cycles: pd.DataFrame | Path | str) -> pd.DataFrame:
    """Read only the ``cycle_id`` column from ``valid_core_cycles.parquet``."""

    if isinstance(valid_core_cycles, (Path, str)):
        path = Path(valid_core_cycles)
        if not path.exists():
            return pd.DataFrame(columns=["cycle_id"])
        cycle_ids_df = pd.read_parquet(path, columns=["cycle_id"])
    else:
        cycle_ids_df = valid_core_cycles[["cycle_id"]].copy()

    cycle_ids_df["cycle_id"] = cycle_ids_df["cycle_id"].astype(int)
    return cycle_ids_df.drop_duplicates().sort_values("cycle_id").reset_index(drop=True)


def _resolve_cycle_timing(cycle_ids_df: pd.DataFrame, cycle_index: pd.DataFrame) -> pd.DataFrame:
    """Join validated cycle ids with their timing/session information."""

    required_columns = {"cycle_id", "session_id", "start_time", "end_time"}
    missing_columns = required_columns - set(cycle_index.columns)
    if missing_columns:
        raise KeyError(f"cycle_index is missing required column(s): {sorted(missing_columns)}")

    timing_df = cycle_index[list(required_columns)].drop_duplicates(subset=["cycle_id"])
    merged = cycle_ids_df.merge(timing_df, on="cycle_id", how="left", validate="one_to_one")
    return merged.sort_values("cycle_id").reset_index(drop=True)


def _query_measurements(
    measurement_dataset: ds.Dataset | None,
    cycle_ids: list[int],
    signal_names: list[str],
) -> pd.DataFrame:
    """Read one bounded slice of the measurement store for the given cycles."""

    if measurement_dataset is None or not cycle_ids:
        return pd.DataFrame(columns=["cycle_id", "signal_name", "time", "value"])

    filter_expression = ds.field("cycle_id").isin(cycle_ids) & ds.field("signal_name").isin(
        signal_names
    )
    return measurement_dataset.to_table(filter=filter_expression).to_pandas()


def _compute_reference_signal_lengths(
    measurement_dataset: ds.Dataset | None,
    cycle_ids: list[int],
    reference_signal: str,
    query_batch_size: int,
) -> pd.Series:
    """Compute reference-signal cycle lengths for every requested cycle, in bounded chunks."""

    length_series_parts: list[pd.Series] = []
    for chunk_start in range(0, len(cycle_ids), query_batch_size):
        chunk_ids = cycle_ids[chunk_start : chunk_start + query_batch_size]
        chunk_measurements_df = _query_measurements(
            measurement_dataset, chunk_ids, [reference_signal]
        )
        length_series_parts.append(compute_cycle_lengths(chunk_measurements_df, reference_signal))

    if not length_series_parts:
        return pd.Series(dtype=int, name="cycle_length")
    return pd.concat(length_series_parts)


def generate_cycle_tensor_dataset(
    valid_core_cycles: pd.DataFrame | Path | str,
    cycle_index: pd.DataFrame,
    measurement_dataset_path: Path,
    config: CycleTensorGenerationConfig,
    output_directory: Path,
    dataset_name: str = "",
    experiment: str = "",
) -> CycleTensorGenerationResult:
    """Generate the fixed-length, padding-based cycle tensor dataset.

    One NumPy tensor file (and, if enabled, one matching padding-mask file)
    is written per batch of ``config.cycles_per_file`` cycles, each with
    shape ``(batch_size, target_length, number_of_signals)`` /
    ``(batch_size, target_length, number_of_signals)`` (the mask has the
    same shape as the tensor, since every signal is padded/truncated
    independently), plus one metadata Parquet file with one row per
    successfully written cycle and one cycle-length statistics JSON file.
    Cycles missing a required signal are skipped (and recorded in the
    returned ``skipped_cycles`` table) rather than aborting the whole run.
    """

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    measurement_dataset_path = Path(measurement_dataset_path)

    cycle_ids_df = _load_cycle_ids(valid_core_cycles)
    cycles_requested = int(len(cycle_ids_df))

    if cycles_requested == 0:
        empty_result = CycleTensorGenerationResult(
            metadata=pd.DataFrame(columns=list(METADATA_COLUMNS)),
            skipped_cycles=pd.DataFrame(columns=["cycle_id", "reason"]),
            written_files=[],
            mask_files=[],
            length_statistics=build_cycle_length_statistics(
                pd.Series(dtype=int),
                0,
                config.target_length_strategy,
                config.target_length_percentile,
            ),
            summary={
                "cycles_requested": 0,
                "cycles_written": 0,
                "cycles_skipped": 0,
                "batches_written": 0,
                "target_length": 0,
                "number_of_signals": len(config.required_signals),
                "required_signals": list(config.required_signals),
            },
        )
        _write_outputs(empty_result, output_directory)
        return empty_result

    cycles_with_timing_df = _resolve_cycle_timing(cycle_ids_df, cycle_index)

    dataset_available = measurement_dataset_path.exists()
    measurement_dataset = (
        ds.dataset(measurement_dataset_path, format="parquet") if dataset_available else None
    )

    cycle_lengths = _compute_reference_signal_lengths(
        measurement_dataset,
        cycle_ids_df["cycle_id"].tolist(),
        config.reference_signal,
        config.cycles_per_file,
    )
    target_length = determine_target_length(
        cycle_lengths, config.target_length_strategy, config.target_length_percentile
    )
    length_statistics = build_cycle_length_statistics(
        cycle_lengths, target_length, config.target_length_strategy, config.target_length_percentile
    )

    metadata_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    written_files: list[Path] = []
    mask_files: list[Path] = []

    batch_start_position = 0
    for batch in iter_cycle_batches(cycles_with_timing_df, config.cycles_per_file):
        cycle_ids_in_batch = batch["cycle_id"].astype(int).tolist()
        batch_measurements_df = _query_measurements(
            measurement_dataset, cycle_ids_in_batch, list(config.required_signals)
        )
        measurements_by_cycle = (
            batch_measurements_df.groupby("cycle_id")
            if not batch_measurements_df.empty
            else None
        )

        batch_matrices: list[np.ndarray] = []
        batch_masks: list[np.ndarray] = []
        batch_metadata_rows: list[dict[str, object]] = []

        for row in batch.itertuples(index=False):
            cycle_id = int(row.cycle_id)
            session_id = row.session_id
            cycle_start = pd.Timestamp(row.start_time)
            cycle_end = pd.Timestamp(row.end_time)

            if pd.isna(cycle_start) or pd.isna(cycle_end):
                skipped_rows.append({"cycle_id": cycle_id, "reason": "missing_cycle_timing"})
                continue

            cycle_measurements_df = (
                measurements_by_cycle.get_group(cycle_id)
                if measurements_by_cycle is not None
                and cycle_id in getattr(measurements_by_cycle, "groups", {})
                else pd.DataFrame(columns=["cycle_id", "signal_name", "time", "value"])
            )

            try:
                (
                    matrix,
                    mask,
                    original_length,
                    padded_samples,
                    truncated_samples,
                    signal_original_lengths,
                ) = build_cycle_tensor(
                    cycle_measurements_df,
                    config.required_signals,
                    config.reference_signal,
                    target_length,
                    config.padding_method,
                )
            except MissingRequiredSignalError as exc:
                logger.warning("Skipping cycle_id=%d: %s", cycle_id, exc)
                skipped_rows.append({"cycle_id": cycle_id, "reason": str(exc)})
                continue

            if truncated_samples > 0:
                if not config.truncate_long_cycles:
                    logger.warning(
                        "Skipping cycle_id=%d because it is longer than target_length=%d "
                        "(original_cycle_length=%d) and truncate_long_cycles is disabled",
                        cycle_id,
                        target_length,
                        original_length,
                    )
                    skipped_rows.append(
                        {"cycle_id": cycle_id, "reason": "cycle_longer_than_target_length"}
                    )
                    continue
                logger.info(
                    "Truncated cycle_id=%d by %d sample(s) "
                    "(original_cycle_length=%d, target_length=%d)",
                    cycle_id,
                    truncated_samples,
                    original_length,
                    target_length,
                )

            batch_matrices.append(matrix)
            batch_masks.append(mask)
            batch_metadata_rows.append(
                {
                    "cycle_id": cycle_id,
                    "session_id": session_id,
                    "cycle_start": cycle_start,
                    "cycle_end": cycle_end,
                    "cycle_duration_seconds": (cycle_end - cycle_start).total_seconds(),
                    "original_cycle_length": original_length,
                    "target_length": target_length,
                    "padded_samples": padded_samples,
                    "truncated_samples": truncated_samples,
                    "signal_original_lengths": json.dumps(signal_original_lengths, sort_keys=True),
                }
            )

        if not batch_matrices:
            batch_start_position += len(batch)
            continue

        batch_tensor = np.stack(batch_matrices, axis=0)
        batch_mask_array = np.stack(batch_masks, axis=0) if config.save_padding_mask else None
        file_name = _batch_file_name(
            dataset_name,
            experiment,
            batch_start_position + 1,
            batch_start_position + len(batch),
            config.output_format,
        )
        tensor_path, mask_path = write_tensor_batch(
            batch_tensor, batch_mask_array, output_directory, file_name
        )
        written_files.append(tensor_path)
        if mask_path is not None:
            mask_files.append(mask_path)

        for metadata_row in batch_metadata_rows:
            metadata_row["batch_file"] = file_name
            metadata_rows.append(metadata_row)

        logger.info("Wrote cycle tensor batch %s with shape %s", file_name, batch_tensor.shape)
        batch_start_position += len(batch)

    metadata_df = pd.DataFrame(metadata_rows, columns=list(METADATA_COLUMNS))
    skipped_df = pd.DataFrame(skipped_rows, columns=["cycle_id", "reason"])

    summary: dict[str, object] = {
        "dataset": dataset_name,
        "experiment": experiment,
        "cycles_requested": cycles_requested,
        "cycles_written": int(len(metadata_df)),
        "cycles_skipped": int(len(skipped_df)),
        "batches_written": int(len(written_files)),
        "target_length": target_length,
        "number_of_signals": len(config.required_signals),
        "required_signals": list(config.required_signals),
        "reference_signal": config.reference_signal,
        "cycles_per_file": config.cycles_per_file,
        "target_length_strategy": config.target_length_strategy,
        "target_length_percentile": config.target_length_percentile,
        "padding_method": config.padding_method,
        "truncate_long_cycles": config.truncate_long_cycles,
        "save_padding_mask": config.save_padding_mask,
        "output_format": config.output_format,
    }

    result = CycleTensorGenerationResult(
        metadata=metadata_df,
        skipped_cycles=skipped_df,
        written_files=written_files,
        mask_files=mask_files,
        length_statistics=length_statistics,
        summary=summary,
    )
    _write_outputs(result, output_directory)
    logger.info(
        "Cycle tensor generation: %d requested, %d written, %d skipped, %d batch file(s), "
        "target_length=%d",
        summary["cycles_requested"],
        summary["cycles_written"],
        summary["cycles_skipped"],
        summary["batches_written"],
        target_length,
    )
    return result


def _write_outputs(result: CycleTensorGenerationResult, output_directory: Path) -> None:
    """Persist the metadata table, skipped-cycle table, and JSON summaries."""

    result.metadata.to_parquet(output_directory / CYCLE_TENSOR_METADATA_FILE_NAME, index=False)
    result.skipped_cycles.to_parquet(output_directory / SKIPPED_CYCLES_FILE_NAME, index=False)
    (output_directory / SUMMARY_FILE_NAME).write_text(
        json.dumps(result.summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    (output_directory / STATISTICS_FILE_NAME).write_text(
        json.dumps(result.length_statistics, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
