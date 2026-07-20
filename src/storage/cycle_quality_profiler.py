"""Exploratory, batch-oriented cycle quality profiling.

This module computes descriptive quality metrics for extracted cycles so
that robust validation thresholds can later be derived from the observed
distributions. It intentionally does **not** reject cycles or apply any
hard pass/fail thresholds; every metric is purely descriptive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

logger = logging.getLogger(__name__)

DISTRIBUTION_PERCENTILES: tuple[float, ...] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)

SIGNAL_QUALITY_METRIC_COLUMNS: tuple[str, ...] = (
    "number_of_samples",
    "sampling_rate_hz",
    "duration_covered_seconds",
    "coverage_ratio",
    "value_range",
    "mean_value",
    "std_value",
    "max_time_gap_seconds",
    "mean_time_gap_seconds",
    "non_finite_value_count",
)


@dataclass(slots=True)
class CycleQualityProfileResult:
    """Outcome of one cycle quality profiling run."""

    signal_quality_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    cycle_quality_profile: pd.DataFrame = field(default_factory=pd.DataFrame)
    distribution_summary: pd.DataFrame = field(default_factory=pd.DataFrame)


def _discover_signal_names(measurement_dataset_path: Path) -> list[str]:
    """Return the sorted set of signal names present in the measurement store."""

    if not measurement_dataset_path.exists():
        return []

    dataset = ds.dataset(measurement_dataset_path, format="parquet")
    signal_name_table = dataset.to_table(columns=["signal_name"])
    if signal_name_table.num_rows == 0:
        return []
    signal_names = pd.unique(signal_name_table.column("signal_name").to_pandas())
    return sorted(str(name) for name in signal_names)


def _time_gap_stats(sorted_times: np.ndarray) -> tuple[float, float]:
    """Return (max, mean) inter-sample gap in seconds for one sorted time array."""

    if sorted_times.size < 2:
        return (0.0, 0.0)
    gaps_seconds = np.diff(sorted_times).astype("timedelta64[ns]").astype(np.float64) / 1e9
    return (float(np.max(gaps_seconds)), float(np.mean(gaps_seconds)))


def _median_sampling_interval_seconds(sorted_times: np.ndarray) -> float:
    """Return the median inter-sample interval in seconds, or NaN when undefined."""

    if sorted_times.size < 2:
        return np.nan
    gaps_seconds = np.diff(sorted_times).astype("timedelta64[ns]").astype(np.float64) / 1e9
    return float(np.median(gaps_seconds))


def _duplicate_timestamp_count(sorted_times: np.ndarray) -> int:
    """Count consecutive samples that share the exact same timestamp."""

    if sorted_times.size < 2:
        return 0
    gaps_seconds = np.diff(sorted_times).astype("timedelta64[ns]").astype(np.float64) / 1e9
    return int(np.sum(gaps_seconds == 0.0))


def _signal_metrics_for_group(values: pd.DataFrame) -> dict[str, object]:
    """Compute descriptive quality metrics for one (cycle, signal) group."""

    sorted_group = values.sort_values("time")
    times = sorted_group["time"].to_numpy()
    raw_values = sorted_group["value"].to_numpy(dtype=float)

    number_of_samples = int(raw_values.size)
    finite_mask = np.isfinite(raw_values)
    non_finite_value_count = int(number_of_samples - int(finite_mask.sum()))
    finite_values = raw_values[finite_mask]

    duration_covered_seconds = (
        float((times[-1] - times[0]) / np.timedelta64(1, "s")) if number_of_samples >= 2 else 0.0
    )
    max_gap_seconds, mean_gap_seconds = _time_gap_stats(times)

    mean_value = float(np.mean(finite_values)) if finite_values.size else np.nan
    std_value = float(np.std(finite_values)) if finite_values.size > 1 else 0.0
    min_value = float(np.min(finite_values)) if finite_values.size else np.nan
    max_value = float(np.max(finite_values)) if finite_values.size else np.nan
    median_value = float(np.median(finite_values)) if finite_values.size else np.nan
    value_range = (max_value - min_value) if finite_values.size else np.nan

    sampling_rate_hz = (
        float((number_of_samples - 1) / duration_covered_seconds)
        if duration_covered_seconds > 0
        else np.nan
    )
    median_sampling_interval_seconds = _median_sampling_interval_seconds(times)
    duplicate_timestamp_count = _duplicate_timestamp_count(times)
    first_timestamp = sorted_group["time"].iloc[0]
    last_timestamp = sorted_group["time"].iloc[-1]
    is_constant_signal = bool(finite_values.size > 1 and std_value == 0.0)

    return {
        "number_of_samples": number_of_samples,
        "non_finite_value_count": non_finite_value_count,
        "duration_covered_seconds": duration_covered_seconds,
        "sampling_rate_hz": sampling_rate_hz,
        "max_time_gap_seconds": max_gap_seconds,
        "mean_time_gap_seconds": mean_gap_seconds,
        "mean_value": mean_value,
        "std_value": std_value,
        "min_value": min_value,
        "max_value": max_value,
        "median_value": median_value,
        "value_range": value_range,
        "is_constant_signal": is_constant_signal,
        # Aliases and additions matching the cycle_quality_profiling metric
        # vocabulary used by downstream validation-rule generation.
        "sample_count": number_of_samples,
        "finite_sample_count": int(finite_values.size),
        "non_finite_count": non_finite_value_count,
        "missing_signal": False,
        "constant_signal": is_constant_signal,
        "signal_min": min_value,
        "signal_max": max_value,
        "signal_range": value_range,
        "mean": mean_value,
        "median": median_value,
        "standard_deviation": std_value,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "observed_duration_seconds": duration_covered_seconds,
        "median_sampling_interval": median_sampling_interval_seconds,
        "estimated_sampling_rate": sampling_rate_hz,
        "maximum_timestamp_gap": max_gap_seconds,
        "duplicate_timestamp_count": duplicate_timestamp_count,
    }


def _compute_batch_signal_quality(
    cycle_batch: pd.DataFrame,
    measurements_df: pd.DataFrame,
    signal_names: list[str],
) -> list[dict[str, object]]:
    """Compute one signal-quality row per (cycle, signal) pair in the batch."""

    grouped = (
        measurements_df.groupby(["cycle_id", "signal_name"])
        if not measurements_df.empty
        else None
    )
    metrics_by_cycle: dict[int, dict[str, dict[str, object]]] = {}
    if grouped is not None:
        for (cycle_id, signal_name), group_df in grouped:
            metrics_by_cycle.setdefault(int(cycle_id), {})[str(signal_name)] = (
                _signal_metrics_for_group(group_df)
            )

    rows: list[dict[str, object]] = []
    for cycle_row in cycle_batch.itertuples(index=False):
        cycle_id = int(cycle_row.cycle_id)
        session_id = int(cycle_row.session_id) if hasattr(cycle_row, "session_id") else None
        experiment = str(cycle_row.experiment) if hasattr(cycle_row, "experiment") else ""
        cycle_duration_seconds = (
            float(cycle_row.duration_seconds) if hasattr(cycle_row, "duration_seconds") else np.nan
        )

        cycle_metrics = metrics_by_cycle.get(cycle_id, {})
        for signal_name in signal_names:
            signal_metrics = cycle_metrics.get(signal_name)
            is_missing = signal_metrics is None

            row: dict[str, object] = {
                "experiment": experiment,
                "session_id": session_id,
                "cycle_id": cycle_id,
                "signal_name": signal_name,
                "is_missing": is_missing,
            }
            if is_missing:
                row.update(
                    {
                        "number_of_samples": 0,
                        "non_finite_value_count": 0,
                        "duration_covered_seconds": np.nan,
                        "coverage_ratio": np.nan,
                        "sampling_rate_hz": np.nan,
                        "max_time_gap_seconds": np.nan,
                        "mean_time_gap_seconds": np.nan,
                        "mean_value": np.nan,
                        "std_value": np.nan,
                        "min_value": np.nan,
                        "max_value": np.nan,
                        "median_value": np.nan,
                        "value_range": np.nan,
                        "is_constant_signal": False,
                        # Aliases matching the extended metric vocabulary. A
                        # missing signal is explicitly represented rather than
                        # treated as an extraction failure.
                        "sample_count": 0,
                        "finite_sample_count": 0,
                        "non_finite_count": 0,
                        "missing_signal": True,
                        "constant_signal": False,
                        "signal_min": np.nan,
                        "signal_max": np.nan,
                        "signal_range": np.nan,
                        "mean": np.nan,
                        "median": np.nan,
                        "standard_deviation": np.nan,
                        "first_timestamp": pd.NaT,
                        "last_timestamp": pd.NaT,
                        "observed_duration_seconds": np.nan,
                        "median_sampling_interval": np.nan,
                        "estimated_sampling_rate": np.nan,
                        "maximum_timestamp_gap": np.nan,
                        "duplicate_timestamp_count": 0,
                        "extraction_error": False,
                    }
                )
            else:
                coverage_ratio = (
                    float(signal_metrics["duration_covered_seconds"] / cycle_duration_seconds)
                    if cycle_duration_seconds and cycle_duration_seconds > 0
                    else np.nan
                )
                row.update(signal_metrics)
                row["coverage_ratio"] = coverage_ratio
                row["extraction_error"] = False
            rows.append(row)

    return rows


def _aggregate_cycle_profile(
    cycle_batch: pd.DataFrame, signal_quality_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Aggregate per-signal quality rows into one descriptive row per cycle."""

    quality_df = pd.DataFrame(signal_quality_rows)
    cycle_rows: list[dict[str, object]] = []

    for cycle_row in cycle_batch.itertuples(index=False):
        cycle_id = int(cycle_row.cycle_id)
        session_id = int(cycle_row.session_id) if hasattr(cycle_row, "session_id") else None
        experiment = str(cycle_row.experiment) if hasattr(cycle_row, "experiment") else ""

        cycle_signals_df = quality_df[quality_df["cycle_id"] == cycle_id]
        present_df = cycle_signals_df[~cycle_signals_df["is_missing"]]
        missing_names = sorted(
            cycle_signals_df.loc[cycle_signals_df["is_missing"], "signal_name"].tolist()
        )

        row: dict[str, object] = {
            "experiment": experiment,
            "session_id": session_id,
            "cycle_id": cycle_id,
            "cycle_duration_seconds": (
                float(cycle_row.duration_seconds) if hasattr(cycle_row, "duration_seconds") else np.nan
            ),
            "total_signals_expected": int(len(cycle_signals_df)),
            "signals_present": int(len(present_df)),
            "signals_missing": int(len(missing_names)),
            "missing_signal_names": ",".join(missing_names),
            "constant_signal_count": int(present_df["is_constant_signal"].sum()) if not present_df.empty else 0,
        }
        for metric in (
            "sampling_rate_hz",
            "number_of_samples",
            "coverage_ratio",
            "max_time_gap_seconds",
        ):
            values = present_df[metric].dropna() if not present_df.empty else pd.Series(dtype=float)
            row[f"min_{metric}"] = float(values.min()) if not values.empty else np.nan
            row[f"mean_{metric}"] = float(values.mean()) if not values.empty else np.nan
            row[f"max_{metric}"] = float(values.max()) if not values.empty else np.nan

        position_rows = cycle_signals_df[cycle_signals_df["signal_name"] == "position"]
        row["position_stroke_range"] = (
            float(position_rows["value_range"].iloc[0])
            if not position_rows.empty and pd.notna(position_rows["value_range"].iloc[0])
            else np.nan
        )

        cycle_rows.append(row)

    return cycle_rows


def _build_distribution_summary(signal_quality_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize each metric's observed distribution per signal for threshold derivation."""

    if signal_quality_df.empty:
        return pd.DataFrame()

    present_df = signal_quality_df[~signal_quality_df["is_missing"]]
    summary_rows: list[dict[str, object]] = []

    for signal_name, group_df in present_df.groupby("signal_name"):
        for metric in SIGNAL_QUALITY_METRIC_COLUMNS:
            values = group_df[metric].dropna()
            row: dict[str, object] = {
                "signal_name": signal_name,
                "metric": metric,
                "count": int(values.size),
            }
            if values.empty:
                row.update({"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan})
                for percentile in DISTRIBUTION_PERCENTILES:
                    row[f"p{int(percentile * 100)}"] = np.nan
            else:
                row.update(
                    {
                        "mean": float(values.mean()),
                        "std": float(values.std()) if values.size > 1 else 0.0,
                        "min": float(values.min()),
                        "max": float(values.max()),
                    }
                )
                for percentile in DISTRIBUTION_PERCENTILES:
                    row[f"p{int(percentile * 100)}"] = float(values.quantile(percentile))
            summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def build_cycle_quality_profile(
    measurement_dataset_path: Path,
    cycles_df: pd.DataFrame,
    output_directory: Path,
    quality_batch_size: int = 1000,
) -> CycleQualityProfileResult:
    """Compute exploratory, non-rejecting quality metrics for extracted cycles.

    Cycles are processed ``quality_batch_size`` at a time so memory usage
    stays bounded regardless of the total cycle count, matching the batching
    strategy used elsewhere in the pipeline. No cycle is rejected and no hard
    validation threshold is applied here; the returned tables are purely
    descriptive and intended to help derive robust thresholds later from the
    observed distributions.
    """

    measurement_dataset_path = Path(measurement_dataset_path)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    if cycles_df.empty:
        empty_result = CycleQualityProfileResult()
        _save_outputs(empty_result, output_directory)
        return empty_result

    signal_names = _discover_signal_names(measurement_dataset_path)
    dataset_available = measurement_dataset_path.exists()
    dataset = ds.dataset(measurement_dataset_path, format="parquet") if dataset_available else None

    signal_quality_rows: list[dict[str, object]] = []
    cycle_profile_rows: list[dict[str, object]] = []

    for batch_start in range(0, len(cycles_df), quality_batch_size):
        cycle_batch = cycles_df.iloc[batch_start : batch_start + quality_batch_size]
        cycle_ids = cycle_batch["cycle_id"].astype(int).tolist()

        if dataset is not None and cycle_ids:
            filter_expression = ds.field("cycle_id").isin(cycle_ids)
            measurements_df = dataset.to_table(filter=filter_expression).to_pandas()
        else:
            measurements_df = pd.DataFrame(columns=["cycle_id", "signal_name", "time", "value"])

        batch_signal_rows = _compute_batch_signal_quality(cycle_batch, measurements_df, signal_names)
        signal_quality_rows.extend(batch_signal_rows)
        cycle_profile_rows.extend(_aggregate_cycle_profile(cycle_batch, batch_signal_rows))

        logger.info(
            "Profiled quality for %d cycle(s) in batch starting at row %d",
            len(cycle_ids),
            batch_start,
        )
        del measurements_df

    signal_quality_df = pd.DataFrame(signal_quality_rows)
    cycle_quality_df = pd.DataFrame(cycle_profile_rows)
    distribution_summary_df = _build_distribution_summary(signal_quality_df)

    result = CycleQualityProfileResult(
        signal_quality_metrics=signal_quality_df,
        cycle_quality_profile=cycle_quality_df,
        distribution_summary=distribution_summary_df,
    )
    _save_outputs(result, output_directory)
    logger.info(
        "Wrote cycle quality profile: %d signal-quality row(s), %d cycle-profile row(s)",
        len(signal_quality_df),
        len(cycle_quality_df),
    )
    return result


def _save_outputs(result: CycleQualityProfileResult, output_directory: Path) -> None:
    """Persist all three quality-profiling tables to the output directory."""

    result.signal_quality_metrics.to_parquet(
        output_directory / "signal_quality_metrics.parquet", index=False
    )
    result.cycle_quality_profile.to_parquet(
        output_directory / "cycle_quality_profile.parquet", index=False
    )
    result.distribution_summary.to_csv(
        output_directory / "quality_metric_distribution_summary.csv", index=False
    )
