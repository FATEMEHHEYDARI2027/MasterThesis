"""Reusable orchestration for the thesis preprocessing pipeline."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.preprocessing.cycle_detection import detect_candidate_cycles
from src.preprocessing.dataset_validation import (
    DatasetValidationConfig,
    validate_dataset,
)
from src.preprocessing.multi_sensor_cycle_extraction import list_experiment_signals
from src.preprocessing.session_detection import (
    DEFAULT_SESSION_GAP_SECONDS,
    detect_recording_sessions,
)
from src.preprocessing.time_gap_analysis import (
    analyze_time_gaps,
    plot_time_gap_histogram,
    save_time_gap_statistics,
)
from src.preprocessing.validation_cycle_selection import select_validation_cycles
from src.preprocessing.validation_rule_generation import (
    RuleGenerationConfig,
    generate_validation_rules,
    normalize_signal_roles,
)
from src.storage.batched_measurement_writer import (
    MEASUREMENTS_DIRECTORY_NAME,
    SIGNAL_SUMMARY_FILE_NAME,
    write_measurement_batches,
)
from src.storage.cycle_index_writer import write_cycle_index
from src.storage.cycle_quality_profiler import build_cycle_quality_profile
from src.storage.feature_writer import build_cycle_feature_table
from src.utils.data_loader import (
    build_int_signal_info_from_metadata,
    build_uuid_signal_info_from_metadata,
    find_signals,
    load_metadata,
)
from src.utils.measurement_loader import load_uuid_signal
from src.visualization.multi_sensor_cycle_plot import plot_multi_sensor_cycle

logger = logging.getLogger(__name__)


class PipelineStage(str, Enum):
    """Ordered pipeline stages."""

    METADATA = "metadata"
    SIGNAL_DISCOVERY = "signal_discovery"
    TIMESTAMP_ANALYSIS = "timestamp_analysis"
    SESSION_DETECTION = "session_detection"
    CYCLE_DETECTION = "cycle_detection"
    MULTI_SENSOR_EXTRACTION = "multi_sensor_extraction"
    CYCLE_QUALITY_PROFILING = "cycle_quality_profiling"
    VALIDATION_RULE_GENERATION = "validation_rule_generation"
    DATASET_VALIDATION = "dataset_validation"
    FEATURE_ENGINEERING = "feature_engineering"
    DATASET_GENERATION = "dataset_generation"


STAGE_ORDER: tuple[PipelineStage, ...] = (
    PipelineStage.METADATA,
    PipelineStage.SIGNAL_DISCOVERY,
    PipelineStage.TIMESTAMP_ANALYSIS,
    PipelineStage.SESSION_DETECTION,
    PipelineStage.CYCLE_DETECTION,
    PipelineStage.MULTI_SENSOR_EXTRACTION,
    PipelineStage.CYCLE_QUALITY_PROFILING,
    PipelineStage.VALIDATION_RULE_GENERATION,
    PipelineStage.DATASET_VALIDATION,
    PipelineStage.FEATURE_ENGINEERING,
    PipelineStage.DATASET_GENERATION,
)
IMPLEMENTED_STAGES: frozenset[PipelineStage] = frozenset(
    {
        PipelineStage.METADATA,
        PipelineStage.SIGNAL_DISCOVERY,
        PipelineStage.TIMESTAMP_ANALYSIS,
        PipelineStage.SESSION_DETECTION,
        PipelineStage.CYCLE_DETECTION,
        PipelineStage.MULTI_SENSOR_EXTRACTION,
        PipelineStage.CYCLE_QUALITY_PROFILING,
        PipelineStage.VALIDATION_RULE_GENERATION,
        PipelineStage.DATASET_VALIDATION,
    }
)
STAGE_DIRECTORIES: dict[PipelineStage, str] = {
    PipelineStage.METADATA: "metadata",
    PipelineStage.SIGNAL_DISCOVERY: "signal_discovery",
    PipelineStage.TIMESTAMP_ANALYSIS: "timestamp_analysis",
    PipelineStage.SESSION_DETECTION: "sessions",
    PipelineStage.CYCLE_DETECTION: "cycles",
    PipelineStage.MULTI_SENSOR_EXTRACTION: "multi_sensor",
    PipelineStage.CYCLE_QUALITY_PROFILING: "quality_profiling",
    PipelineStage.VALIDATION_RULE_GENERATION: "validation_rule_generation",
    PipelineStage.DATASET_VALIDATION: "dataset_validation",
    PipelineStage.FEATURE_ENGINEERING: "features",
    PipelineStage.DATASET_GENERATION: "dataset",
}


@dataclass(slots=True)
class PipelineConfig:
    """Validated inputs for one pipeline run."""

    dataset_path: Path
    experiment: str
    stop_after: str
    reference_signal: str = "position"
    session_gap_seconds: float | None = None
    movement_threshold: float = 1.0
    output_root: Path = Path("outputs")
    max_cycles_to_extract: int = 3
    extract_all_cycles: bool = False
    cycle_batch_size: int = 500
    resume_extraction: bool = True
    overwrite_existing: bool = False
    selected_extraction_signals: tuple[str, ...] = ()
    validation_cycle_count: int = 3
    required_validation_signals: tuple[str, ...] = ()
    minimum_samples_per_validation_cycle: dict[str, int] | None = None
    require_consecutive_validation_cycles: bool = True
    max_cycles_to_scan_for_validation: int | None = 10_000
    generate_validation_html: bool = True
    generate_cycle_features: bool = False
    parquet_compression: str = "zstd"
    quality_profiling_batch_size: int = 1000
    signal_roles: dict[str, object] | None = None
    validation_rule_generation: dict[str, object] | None = None
    dataset_validation: dict[str, object] | None = None


@dataclass(slots=True)
class _RunPaths:
    """Paths created for one pipeline run."""

    run_directory: Path
    manifest_path: Path
    stage_directories: dict[PipelineStage, Path]


def _as_path(value: Path | str) -> Path:
    """Normalize a path-like value without requiring that it already exists."""

    return Path(value).expanduser()


def _normalize_stage(stage_name: str) -> PipelineStage:
    """Parse one stage name into the stage enum."""

    try:
        return PipelineStage(stage_name)
    except ValueError as exc:
        valid_stages = ", ".join(stage.value for stage in STAGE_ORDER)
        raise ValueError(
            f"Invalid stop_after stage {stage_name!r}. Valid stages: {valid_stages}."
        ) from exc


def _ensure_stage_is_implemented(stage: PipelineStage) -> None:
    """Raise a clear error for planned but not yet implemented stages."""

    if stage not in IMPLEMENTED_STAGES:
        raise NotImplementedError(
            f"Pipeline stage {stage.value!r} is defined but not implemented yet."
        )


def _build_run_paths(config: PipelineConfig) -> _RunPaths:
    """Create the output directory structure for one run."""

    dataset_name = _as_path(config.dataset_path).name or str(config.dataset_path)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = (
        _as_path(config.output_root)
        / dataset_name
        / config.experiment
        / run_id
    )
    run_directory.mkdir(parents=True, exist_ok=True)

    stage_directories = {
        stage: run_directory / folder_name
        for stage, folder_name in STAGE_DIRECTORIES.items()
    }
    for stage_directory in stage_directories.values():
        stage_directory.mkdir(parents=True, exist_ok=True)

    return _RunPaths(
        run_directory=run_directory,
        manifest_path=run_directory / "run_manifest.json",
        stage_directories=stage_directories,
    )


def _json_ready(value: Any) -> Any:
    """Convert common runtime values into JSON-serializable structures."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_ready(inner_value) for key, inner_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    """Persist the current manifest state."""

    manifest_path.write_text(
        json.dumps(_json_ready(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _save_frame(frame: pd.DataFrame, output_path: Path) -> Path:
    """Save one DataFrame as CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def _save_parquet(frame: pd.DataFrame, output_path: Path) -> Path:
    """Save one DataFrame as Parquet, creating parent directories as needed."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return output_path


def _load_position_window(
    dataset_path: Path,
    signal_id_uuid: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    """Load one inclusive session or cycle window for the reference signal."""

    end_time_exclusive = pd.Timestamp(end_time) + pd.Timedelta(microseconds=1)
    return load_uuid_signal(
        dataset_path,
        signal_id_uuid,
        start_time=pd.Timestamp(start_time),
        end_time=end_time_exclusive,
    )


MAX_VALIDATION_POINTS = 20_000
MAX_VALIDATION_CYCLES = 2_000
MAX_VALIDATION_PLOT_CYCLES = 100


def _evenly_spaced_indices(length: int, max_count: int) -> np.ndarray:
    """Return evenly spaced row indices, always keeping the first and last row."""

    if length <= max_count:
        return np.arange(length)
    return np.unique(np.linspace(0, length - 1, num=max_count, dtype=np.int64))


def _nearest_values(
    position_df: pd.DataFrame, timestamps: pd.Series
) -> np.ndarray:
    """Look up the Position value nearest to each timestamp via searchsorted."""

    sorted_times = position_df["time"].to_numpy()
    values = position_df["value"].to_numpy()
    query_times = pd.to_datetime(timestamps).to_numpy()

    right_indices = np.searchsorted(sorted_times, query_times, side="left")
    right_indices = np.clip(right_indices, 0, len(sorted_times) - 1)
    left_indices = np.clip(right_indices - 1, 0, len(sorted_times) - 1)

    left_diff = np.abs(sorted_times[left_indices] - query_times)
    right_diff = np.abs(sorted_times[right_indices] - query_times)
    nearest_indices = np.where(left_diff <= right_diff, left_indices, right_indices)
    return values[nearest_indices]


def _plot_cycle_validation(
    position_df: pd.DataFrame,
    cycles_df: pd.DataFrame,
    output_path: Path,
    experiment: str,
    movement_threshold: float,
) -> Path | None:
    """Save one interactive Plotly validation plot for detected cycles."""

    if position_df.empty:
        return None

    plot_indices = _evenly_spaced_indices(len(position_df), MAX_VALIDATION_POINTS)
    position_plot_df = position_df.iloc[plot_indices]
    logger.info(
        "cycle validation plot: plotting %d of %d position samples",
        len(position_plot_df),
        len(position_df),
    )

    cycle_indices = _evenly_spaced_indices(len(cycles_df), MAX_VALIDATION_CYCLES)
    cycles_plot_df = cycles_df.iloc[cycle_indices]
    logger.info(
        "cycle validation plot: plotting %d of %d detected cycles",
        len(cycles_plot_df),
        len(cycles_df),
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=position_plot_df["time"],
            y=position_plot_df["value"],
            mode="markers",
            marker=dict(size=3),
            name="Position samples",
            customdata=position_plot_df[["time", "value"]],
            hovertemplate=(
                "timestamp: %{customdata[0]}<br>value: %{customdata[1]}<extra></extra>"
            ),
        )
    )

    if not position_plot_df.empty:
        fig.add_hline(
            y=movement_threshold,
            line=dict(color="black", dash="dash", width=1.0),
        )

    if not cycles_plot_df.empty:
        has_session_id = "session_id" in cycles_plot_df.columns

        for boundary_type, time_column, symbol in (
            ("start", "start_time", "triangle-up"),
            ("end", "end_time", "triangle-down"),
        ):
            boundary_times = cycles_plot_df[time_column]
            boundary_values = _nearest_values(position_df, boundary_times)
            session_ids = (
                cycles_plot_df["session_id"]
                if has_session_id
                else pd.Series(["n/a"] * len(cycles_plot_df))
            )
            custom_data = np.column_stack(
                [
                    cycles_plot_df["cycle_id"].to_numpy(),
                    session_ids.to_numpy(),
                    boundary_times.astype(str).to_numpy(),
                    boundary_values,
                    np.full(len(cycles_plot_df), boundary_type),
                ]
            )
            fig.add_trace(
                go.Scattergl(
                    x=boundary_times,
                    y=boundary_values,
                    mode="markers",
                    marker=dict(
                        size=8,
                        symbol=symbol,
                        color="green" if boundary_type == "start" else "red",
                    ),
                    name=f"Cycle {boundary_type}s",
                    customdata=custom_data,
                    hovertemplate=(
                        "cycle_id: %{customdata[0]}<br>"
                        "session_id: %{customdata[1]}<br>"
                        "timestamp: %{customdata[2]}<br>"
                        "value: %{customdata[3]}<br>"
                        "boundary: %{customdata[4]}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=f"{experiment} reference-signal cycle validation",
        xaxis_title="Time",
        yaxis_title="Position value",
        dragmode="zoom",
        hovermode="closest",
        xaxis=dict(rangeslider=dict(visible=False)),
    )
    fig.update_layout(
        modebar_add=[
            "select2d",
            "lasso2d",
            "zoomIn2d",
            "zoomOut2d",
            "autoScale2d",
            "resetScale2d",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
    )
    return output_path


def _row_count(frame: pd.DataFrame) -> int:
    """Return the number of rows in a DataFrame."""

    return int(len(frame))


def _run_metadata_stage(
    stage_directory: Path,
    uuid_signal_info: pd.DataFrame,
    int_signal_info: pd.DataFrame,
) -> dict[str, object]:
    """Save signal catalogues derived from the shared metadata snapshot."""

    uuid_catalogue_path = _save_frame(uuid_signal_info, stage_directory / "uuid_signal_catalogue.csv")
    int_catalogue_path = _save_frame(int_signal_info, stage_directory / "int_signal_catalogue.csv")
    return {
        "uuid_signal_info": uuid_signal_info,
        "int_signal_info": int_signal_info,
        "row_counts": {
            "uuid_signals": _row_count(uuid_signal_info),
            "int_signals": _row_count(int_signal_info),
        },
        "output_paths": {
            "uuid_signal_catalogue": str(uuid_catalogue_path),
            "int_signal_catalogue": str(int_catalogue_path),
        },
    }


def _run_signal_discovery_stage(
    stage_directory: Path,
    uuid_signal_info: pd.DataFrame,
    int_signal_info: pd.DataFrame,
    experiment: str,
    reference_signal: str,
) -> dict[str, object]:
    """Discover experiment-specific signals from metadata only."""

    selected_signals = list_experiment_signals(
        uuid_signal_info=uuid_signal_info,
        int_signal_info=int_signal_info,
        experiment=experiment,
    )
    reference_signals = find_signals(
        uuid_signal_info,
        path_contains=experiment,
        unit_code=reference_signal,
    )
    if len(reference_signals) != 1:
        raise ValueError(
            "Expected exactly one reference signal for "
            f"experiment {experiment!r} and unit {reference_signal!r}, "
            f"found {len(reference_signals)}."
        )

    selected_with_marker = selected_signals.copy()
    selected_with_marker["is_reference_signal"] = (
        selected_with_marker["signal_id_uuid"].fillna("").astype(str)
        == str(reference_signals.iloc[0]["signal_id_uuid"])
    )
    selected_signals_path = _save_frame(selected_with_marker, stage_directory / "selected_signals.csv")
    return {
        "selected_signals": selected_with_marker,
        "reference_signals": reference_signals,
        "reference_signal_uuid": str(reference_signals.iloc[0]["signal_id_uuid"]),
        "row_counts": {
            "selected_signals": _row_count(selected_with_marker),
            "reference_signals": _row_count(reference_signals),
        },
        "output_paths": {
            "selected_signals_csv": str(selected_signals_path),
        },
    }


def _run_timestamp_analysis_stage(
    dataset_path: Path,
    stage_directory: Path,
    reference_signal_uuid: str,
) -> dict[str, object]:
    """Run the existing timestamp-analysis module and persist its outputs."""

    statistics_df, gap_counts_us = analyze_time_gaps(dataset_path, reference_signal_uuid)
    statistics_path = stage_directory / "statistics.csv"
    full_histogram_path = stage_directory / "time_gap_histogram.png"
    zoom_histogram_path = stage_directory / "time_gap_histogram_under_1_second.png"

    save_time_gap_statistics(statistics_df, statistics_path)
    plot_time_gap_histogram(gap_counts_us, full_histogram_path)
    plot_time_gap_histogram(gap_counts_us, zoom_histogram_path, max_gap_seconds=1.0)
    return {
        "statistics": statistics_df,
        "gap_counts_us": gap_counts_us,
        "row_counts": {
            "statistics_rows": _row_count(statistics_df),
            "unique_gap_buckets": len(gap_counts_us),
        },
        "output_paths": {
            "statistics_csv": str(statistics_path),
            "histogram": str(full_histogram_path),
            "histogram_under_1_second": str(zoom_histogram_path),
        },
    }


def _run_session_detection_stage(
    dataset_path: Path,
    stage_directory: Path,
    reference_signal_uuid: str,
    session_gap_seconds: float | None,
) -> dict[str, object]:
    """Run recording-session detection for the selected reference signal."""

    gap_threshold_seconds = (
        DEFAULT_SESSION_GAP_SECONDS
        if session_gap_seconds is None
        else float(session_gap_seconds)
    )
    sessions_df = detect_recording_sessions(
        dataset_path,
        reference_signal_uuid,
        gap_threshold_seconds=gap_threshold_seconds,
    )
    sessions_path = _save_frame(sessions_df, stage_directory / "sessions.csv")
    return {
        "sessions": sessions_df,
        "gap_threshold_seconds": gap_threshold_seconds,
        "row_counts": {"sessions": _row_count(sessions_df)},
        "output_paths": {"sessions_csv": str(sessions_path)},
    }


def _build_validation_subset(
    position_df: pd.DataFrame, session_cycles_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict one session's data to a small, fast-to-render validation subset.

    Only the first ``MAX_VALIDATION_PLOT_CYCLES`` cycles of the session are
    kept, and the Position window is limited to the selected cycles' time
    range plus a one-second margin on each side.
    """

    validation_cycles_df = session_cycles_df.iloc[:MAX_VALIDATION_PLOT_CYCLES]
    if validation_cycles_df.empty:
        return position_df.iloc[0:0], validation_cycles_df

    window_start = pd.Timestamp(validation_cycles_df["start_time"].iloc[0]) - pd.Timedelta(
        seconds=1
    )
    window_end = pd.Timestamp(validation_cycles_df["end_time"].iloc[-1]) + pd.Timedelta(
        seconds=1
    )
    in_window = (position_df["time"] >= window_start) & (position_df["time"] <= window_end)
    validation_position_df = position_df.loc[in_window]
    return validation_position_df, validation_cycles_df


def _run_cycle_detection_stage(
    dataset_path: Path,
    stage_directory: Path,
    experiment: str,
    reference_signal_uuid: str,
    sessions_df: pd.DataFrame,
    movement_threshold: float,
) -> dict[str, object]:
    """Detect cycles session by session to preserve bounded memory usage."""

    cycle_frames: list[pd.DataFrame] = []
    validation_position_df = pd.DataFrame()
    validation_cycles_df = pd.DataFrame()
    validation_session_total_cycles = 0
    validation_session_total_position_rows = 0
    captured_validation_subset = False

    for session_row in sessions_df.itertuples(index=False):
        position_df = _load_position_window(
            dataset_path=dataset_path,
            signal_id_uuid=reference_signal_uuid,
            start_time=pd.Timestamp(session_row.start_time),
            end_time=pd.Timestamp(session_row.end_time),
        )
        session_cycles_df = detect_candidate_cycles(
            position_df,
            movement_threshold=movement_threshold,
        )
        if session_cycles_df.empty:
            continue

        session_cycles_df = session_cycles_df.copy()
        session_cycles_df.insert(0, "experiment", experiment)
        session_cycles_df.insert(1, "session_id", int(session_row.session_id))
        session_cycles_df.insert(2, "reference_signal_uuid", reference_signal_uuid)
        cycle_frames.append(session_cycles_df)

        if not captured_validation_subset:
            validation_position_df, validation_cycles_df = _build_validation_subset(
                position_df, session_cycles_df
            )
            validation_session_total_cycles = len(session_cycles_df)
            validation_session_total_position_rows = len(position_df)
            captured_validation_subset = True

    if cycle_frames:
        cycles_df = pd.concat(cycle_frames, ignore_index=True)
        cycles_df["cycle_id"] = range(1, len(cycles_df) + 1)
    else:
        cycles_df = pd.DataFrame(
            columns=["experiment", "session_id", "reference_signal_uuid", "cycle_id"]
        )

    logger.info(
        "cycle validation subset: %d of %d cycles detected in session, "
        "%d of %d position rows in session",
        len(validation_cycles_df),
        validation_session_total_cycles,
        len(validation_position_df),
        validation_session_total_position_rows,
    )

    cycles_path = _save_frame(cycles_df, stage_directory / "cycles.csv")
    cycles_parquet_path = write_cycle_index(cycles_df, stage_directory / "cycles.parquet")
    validation_plot_path = _plot_cycle_validation(
        validation_position_df,
        validation_cycles_df,
        stage_directory / "cycle_validation.html",
        experiment=experiment,
        movement_threshold=movement_threshold,
    )
    return {
        "cycles": cycles_df,
        "row_counts": {"cycles": _row_count(cycles_df)},
        "output_paths": {
            "cycles_csv": str(cycles_path),
            "cycles_parquet": str(cycles_parquet_path),
            **(
                {"validation_plot": str(validation_plot_path)}
                if validation_plot_path is not None
                else {}
            ),
        },
    }


def _run_multi_sensor_extraction_stage(
    dataset_path: Path,
    stage_directory: Path,
    uuid_signal_info: pd.DataFrame,
    int_signal_info: pd.DataFrame,
    experiment: str,
    cycles_df: pd.DataFrame,
    cycle_index_path: Path,
    max_cycles_to_extract: int,
    extract_all_cycles: bool,
    cycle_batch_size: int,
    resume_extraction: bool,
    overwrite_existing: bool,
    selected_extraction_signals: tuple[str, ...],
    validation_cycle_count: int,
    required_validation_signals: tuple[str, ...],
    minimum_samples_per_validation_cycle: dict[str, int] | None,
    require_consecutive_validation_cycles: bool,
    max_cycles_to_scan_for_validation: int | None,
    generate_validation_html: bool,
    generate_cycle_features: bool,
    parquet_compression: str,
) -> dict[str, object]:
    """Run the scalable, batch-oriented multi-sensor extraction stage.

    The stage is cycle-indexed and Parquet-based: it (1) selects a small
    chronological block of cycles for human-inspectable HTML validation, then
    (2) extracts every requested cycle in bounded batches into a Parquet
    measurement store partitioned by experiment and session (never by
    cycle_id), and (3) optionally derives a cycle-level feature table. No
    per-cycle directories, per-signal CSV files, or per-cycle HTML files are
    created for the full extraction.
    """

    validation_directory = stage_directory / "validation"
    validation_directory.mkdir(parents=True, exist_ok=True)

    selected_validation_cycles_path: Path | None = None
    validation_cycle_quality_path: Path | None = None
    validation_html_paths: list[str] = []
    selected_validation_cycle_count = 0

    if not cycles_df.empty:
        minimum_samples = minimum_samples_per_validation_cycle or {}
        try:
            validation_result = select_validation_cycles(
                base_dir=dataset_path,
                cycles_df=cycles_df,
                uuid_signal_info=uuid_signal_info,
                int_signal_info=int_signal_info,
                experiment=experiment,
                required_signals=required_validation_signals,
                minimum_samples=minimum_samples,
                validation_cycle_count=validation_cycle_count,
                require_consecutive=require_consecutive_validation_cycles,
                max_cycles_to_scan=max_cycles_to_scan_for_validation,
            )
        except ValueError:
            logger.exception(
                "No multi-sensor validation block could be selected for experiment %s",
                experiment,
            )
            raise

        selected_validation_cycles_path = _save_parquet(
            validation_result.selected_cycles,
            validation_directory / "selected_validation_cycles.parquet",
        )
        validation_cycle_quality_path = _save_parquet(
            validation_result.quality_table,
            validation_directory / "validation_cycle_quality.parquet",
        )
        selected_validation_cycle_count = len(validation_result.selected_cycles)

        if generate_validation_html:
            signal_descriptors = list_experiment_signals(
                uuid_signal_info=uuid_signal_info,
                int_signal_info=int_signal_info,
                experiment=experiment,
            )
            for cycle_row in validation_result.selected_cycles.itertuples(index=False):
                cycle_id = int(cycle_row.cycle_id)
                session_id = (
                    int(cycle_row.session_id) if hasattr(cycle_row, "session_id") else None
                )
                extracted_signals = validation_result.extracted_signals_by_cycle.get(cycle_id, {})
                html_path = plot_multi_sensor_cycle(
                    extracted_signals,
                    signal_descriptors,
                    validation_directory / f"cycle_{cycle_id:04d}_multi_sensor.html",
                    experiment=experiment,
                    cycle_id=cycle_id,
                    session_id=session_id,
                )
                if html_path is not None:
                    validation_html_paths.append(str(html_path))

    if extract_all_cycles:
        cycles_to_extract_df = cycles_df
    else:
        if max_cycles_to_extract <= 0:
            raise ValueError("max_cycles_to_extract must be positive.")
        cycles_to_extract_df = cycles_df.head(max_cycles_to_extract)

    effective_resume = resume_extraction and not overwrite_existing
    batch_result = write_measurement_batches(
        base_dir=dataset_path,
        cycles_df=cycles_to_extract_df,
        uuid_signal_info=uuid_signal_info,
        int_signal_info=int_signal_info,
        experiment=experiment,
        output_root=stage_directory,
        cycle_batch_size=cycle_batch_size,
        selected_signals=selected_extraction_signals or None,
        resume=effective_resume,
        parquet_compression=parquet_compression,
    )

    written_measurement_rows = (
        int(batch_result.batch_summary["written_rows"].sum())
        if not batch_result.batch_summary.empty
        else 0
    )

    cycle_features_path: Path | None = None
    if generate_cycle_features:
        cycle_features_path = build_cycle_feature_table(
            measurement_dataset_path=stage_directory / "measurements",
            cycle_index_path=cycle_index_path,
            output_path=stage_directory / "features" / "cycle_features.parquet",
        )

    output_paths: dict[str, object] = {
        "cycle_index_parquet": str(cycle_index_path),
        "measurement_batch_files": [str(path) for path in batch_result.written_files],
        "signal_window_summary_parquet": str(stage_directory / SIGNAL_SUMMARY_FILE_NAME),
        "extraction_checkpoint": (
            str(batch_result.checkpoint_path) if batch_result.checkpoint_path is not None else ""
        ),
    }
    if selected_validation_cycles_path is not None:
        output_paths["selected_validation_cycles_parquet"] = str(selected_validation_cycles_path)
    if validation_cycle_quality_path is not None:
        output_paths["validation_cycle_quality_parquet"] = str(validation_cycle_quality_path)
    if validation_html_paths:
        output_paths["validation_html_files"] = validation_html_paths
    if cycle_features_path is not None:
        output_paths["cycle_features_parquet"] = str(cycle_features_path)

    return {
        "batch_summary": batch_result.batch_summary,
        "signal_window_summary": batch_result.signal_window_summary,
        "cycles_extracted": cycles_to_extract_df,
        "measurements_root": stage_directory / MEASUREMENTS_DIRECTORY_NAME,
        "row_counts": {
            "detected_cycles": _row_count(cycles_df),
            "selected_validation_cycles": selected_validation_cycle_count,
            "processed_cycles": batch_result.processed_cycle_count,
            "failed_cycles": batch_result.failed_cycle_count,
            "written_measurement_rows": written_measurement_rows,
            "written_batch_files": len(batch_result.written_files),
        },
        "output_paths": output_paths,
    }


def _run_cycle_quality_profiling_stage(
    stage_directory: Path,
    measurement_dataset_path: Path,
    extracted_cycles_df: pd.DataFrame,
    quality_profiling_batch_size: int,
) -> dict[str, object]:
    """Compute exploratory, non-rejecting quality metrics for extracted cycles.

    This stage does not reject cycles and does not apply any hard validation
    threshold. It only computes descriptive, cycle-level and signal-level
    quality metrics over the pilot subset of cycles that were actually
    extracted, so that robust validation thresholds can later be derived
    from the observed distributions.
    """

    profile_result = build_cycle_quality_profile(
        measurement_dataset_path=measurement_dataset_path,
        cycles_df=extracted_cycles_df,
        output_directory=stage_directory,
        quality_batch_size=quality_profiling_batch_size,
    )

    return {
        "signal_quality_metrics": profile_result.signal_quality_metrics,
        "cycle_quality_profile": profile_result.cycle_quality_profile,
        "distribution_summary": profile_result.distribution_summary,
        "row_counts": {
            "profiled_cycles": _row_count(profile_result.cycle_quality_profile),
            "signal_quality_rows": _row_count(profile_result.signal_quality_metrics),
        },
        "output_paths": {
            "signal_quality_metrics_parquet": str(
                stage_directory / "signal_quality_metrics.parquet"
            ),
            "cycle_quality_profile_parquet": str(
                stage_directory / "cycle_quality_profile.parquet"
            ),
            "quality_metric_distribution_summary_csv": str(
                stage_directory / "quality_metric_distribution_summary.csv"
            ),
        },
    }


def _run_validation_rule_generation_stage(
    stage_directory: Path,
    signal_quality_metrics: pd.DataFrame,
    cycle_quality_profile: pd.DataFrame,
    signal_roles: dict[str, object] | None,
    rule_generation_config: dict[str, object] | None,
    dataset_name: str,
    experiment: str,
) -> dict[str, object]:
    """Derive robust, data-driven validation thresholds from the profiled cycles.

    This stage never rejects or re-filters cycles; it only reads the
    descriptive profiling output produced by ``cycle_quality_profiling`` and
    derives statistical thresholds from that population, keeping hard
    logical rules separate from learned statistical ones.
    """

    roles = normalize_signal_roles(signal_roles)
    config = RuleGenerationConfig.from_mapping(rule_generation_config)

    result = generate_validation_rules(
        signal_quality_metrics=signal_quality_metrics,
        cycle_quality_profile=cycle_quality_profile,
        signal_roles=roles,
        config=config,
        output_directory=stage_directory,
        dataset_name=dataset_name,
        experiment=experiment,
    )

    return {
        "validation_thresholds": result.validation_thresholds,
        "rule_generation_summary": result.rule_generation_summary,
        "row_counts": {
            "rules_generated": int(result.rule_generation_summary.get("rules_generated", 0)),
            "metrics_skipped": int(result.rule_generation_summary.get("metrics_skipped", 0)),
            "provisional_rules": int(result.rule_generation_summary.get("provisional_rules", 0)),
            "reference_cycles": int(result.rule_generation_summary.get("reference_cycles", 0)),
        },
        "output_paths": {
            "validation_thresholds_json": str(stage_directory / "validation_thresholds.json"),
            "threshold_derivation_summary_csv": str(
                stage_directory / "threshold_derivation_summary.csv"
            ),
            "rule_generation_summary_json": str(stage_directory / "rule_generation_summary.json"),
            "skipped_metrics_csv": str(stage_directory / "skipped_metrics.csv"),
        },
    }


def _run_dataset_validation_stage(
    stage_directory: Path,
    signal_quality_metrics: pd.DataFrame,
    cycle_quality_profile: pd.DataFrame,
    validation_thresholds: pd.DataFrame,
    signal_roles: dict[str, object] | None,
    dataset_validation_config: dict[str, object] | None,
    dataset_name: str,
    experiment: str,
) -> dict[str, object]:
    """Apply the frozen validation rules to classify every profiled cycle."""

    roles = normalize_signal_roles(signal_roles)
    config = DatasetValidationConfig.from_mapping(dataset_validation_config)

    result = validate_dataset(
        signal_quality_metrics=signal_quality_metrics,
        cycle_quality_profile=cycle_quality_profile,
        validation_thresholds=validation_thresholds,
        signal_roles=roles,
        config=config,
        output_directory=stage_directory,
        dataset_name=dataset_name,
        experiment=experiment,
    )

    summary = result.validation_summary
    return {
        "cycle_validation_results": result.cycle_validation_results,
        "validation_summary": summary,
        "row_counts": {
            "cycles_evaluated": int(summary.get("cycles_evaluated", 0)),
            "valid_core_cycles": int(summary.get("valid_core_cycles", 0)),
            "valid_complete_multisensor_cycles": int(
                summary.get("valid_complete_multisensor_cycles", 0)
            ),
            "invalid_cycles": int(summary.get("invalid_cycles", 0)),
            "vibration_unavailable_cycles": int(summary.get("vibration_unavailable_cycles", 0)),
            "vibration_partial_cycles": int(summary.get("vibration_partial_cycles", 0)),
            "vibration_complete_cycles": int(summary.get("vibration_complete_cycles", 0)),
        },
        "output_paths": {
            "cycle_validation_results_parquet": str(
                stage_directory / "cycle_validation_results.parquet"
            ),
            "signal_validation_results_parquet": str(
                stage_directory / "signal_validation_results.parquet"
            ),
            "validation_reason_summary_csv": str(
                stage_directory / "validation_reason_summary.csv"
            ),
            "validation_summary_json": str(stage_directory / "validation_summary.json"),
            "valid_core_cycles_parquet": str(stage_directory / "valid_core_cycles.parquet"),
            "valid_complete_multisensor_cycles_parquet": str(
                stage_directory / "valid_complete_multisensor_cycles.parquet"
            ),
            "invalid_cycles_parquet": str(stage_directory / "invalid_cycles.parquet"),
        },
    }


def run_pipeline(config: PipelineConfig) -> dict[str, object]:
    """Execute the preprocessing pipeline up to the selected stage."""

    normalized_config = PipelineConfig(
        dataset_path=_as_path(config.dataset_path),
        experiment=config.experiment,
        stop_after=config.stop_after,
        reference_signal=config.reference_signal,
        session_gap_seconds=config.session_gap_seconds,
        movement_threshold=config.movement_threshold,
        output_root=_as_path(config.output_root),
        max_cycles_to_extract=config.max_cycles_to_extract,
        extract_all_cycles=config.extract_all_cycles,
        cycle_batch_size=config.cycle_batch_size,
        resume_extraction=config.resume_extraction,
        overwrite_existing=config.overwrite_existing,
        selected_extraction_signals=tuple(config.selected_extraction_signals),
        validation_cycle_count=config.validation_cycle_count,
        required_validation_signals=tuple(config.required_validation_signals),
        minimum_samples_per_validation_cycle=config.minimum_samples_per_validation_cycle,
        require_consecutive_validation_cycles=config.require_consecutive_validation_cycles,
        max_cycles_to_scan_for_validation=config.max_cycles_to_scan_for_validation,
        generate_validation_html=config.generate_validation_html,
        generate_cycle_features=config.generate_cycle_features,
        parquet_compression=config.parquet_compression,
        quality_profiling_batch_size=config.quality_profiling_batch_size,
        signal_roles=config.signal_roles,
        validation_rule_generation=config.validation_rule_generation,
        dataset_validation=config.dataset_validation,
    )
    stop_stage = _normalize_stage(normalized_config.stop_after)
    _ensure_stage_is_implemented(stop_stage)

    run_paths = _build_run_paths(normalized_config)
    dataset_name = normalized_config.dataset_path.name or str(normalized_config.dataset_path)
    start_time = datetime.now()
    manifest: dict[str, Any] = {
        "dataset_path": str(normalized_config.dataset_path),
        "dataset_name": dataset_name,
        "experiment": normalized_config.experiment,
        "reference_signal": normalized_config.reference_signal,
        "stop_point": stop_stage.value,
        "parameters": {
            "session_gap_seconds": normalized_config.session_gap_seconds,
            "movement_threshold": normalized_config.movement_threshold,
            "output_root": str(normalized_config.output_root),
            "max_cycles_to_extract": normalized_config.max_cycles_to_extract,
            "extract_all_cycles": normalized_config.extract_all_cycles,
            "cycle_batch_size": normalized_config.cycle_batch_size,
            "resume_extraction": normalized_config.resume_extraction,
            "overwrite_existing": normalized_config.overwrite_existing,
            "selected_extraction_signals": list(normalized_config.selected_extraction_signals),
            "validation_cycle_count": normalized_config.validation_cycle_count,
            "required_validation_signals": list(normalized_config.required_validation_signals),
            "minimum_samples_per_validation_cycle": normalized_config.minimum_samples_per_validation_cycle,
            "require_consecutive_validation_cycles": normalized_config.require_consecutive_validation_cycles,
            "max_cycles_to_scan_for_validation": normalized_config.max_cycles_to_scan_for_validation,
            "generate_validation_html": normalized_config.generate_validation_html,
            "generate_cycle_features": normalized_config.generate_cycle_features,
            "parquet_compression": normalized_config.parquet_compression,
            "quality_profiling_batch_size": normalized_config.quality_profiling_batch_size,
            "signal_roles": normalized_config.signal_roles,
            "validation_rule_generation": normalized_config.validation_rule_generation,
            "dataset_validation": normalized_config.dataset_validation,
        },
        "start_time": start_time.isoformat(),
        "end_time": None,
        "status": "running",
        "completed_stages": [],
        "generated_output_paths": {},
        "row_counts": {},
        "error_message": None,
    }
    _write_manifest(run_paths.manifest_path, manifest)

    results: dict[str, object] = {}
    try:
        logger.info(
            "Starting pipeline for dataset=%s experiment=%s stop_after=%s output=%s",
            normalized_config.dataset_path,
            normalized_config.experiment,
            stop_stage.value,
            run_paths.run_directory,
        )
        if not normalized_config.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset path does not exist: {normalized_config.dataset_path}"
            )

        metadata_frames = load_metadata(normalized_config.dataset_path)
        uuid_signal_info = build_uuid_signal_info_from_metadata(metadata_frames)
        int_signal_info = build_int_signal_info_from_metadata(metadata_frames)
        available_signals = list_experiment_signals(
            uuid_signal_info=uuid_signal_info,
            int_signal_info=int_signal_info,
            experiment=normalized_config.experiment,
        )
        if available_signals.empty:
            raise ValueError(
                "Experiment not found in metadata: "
                f"{normalized_config.experiment!r}."
            )

        stage_runners = {
            PipelineStage.METADATA: lambda: _run_metadata_stage(
                run_paths.stage_directories[PipelineStage.METADATA],
                uuid_signal_info,
                int_signal_info,
            ),
            PipelineStage.SIGNAL_DISCOVERY: lambda: _run_signal_discovery_stage(
                run_paths.stage_directories[PipelineStage.SIGNAL_DISCOVERY],
                uuid_signal_info,
                int_signal_info,
                normalized_config.experiment,
                normalized_config.reference_signal,
            ),
            PipelineStage.TIMESTAMP_ANALYSIS: lambda: _run_timestamp_analysis_stage(
                normalized_config.dataset_path,
                run_paths.stage_directories[PipelineStage.TIMESTAMP_ANALYSIS],
                results["signal_discovery"]["reference_signal_uuid"],
            ),
            PipelineStage.SESSION_DETECTION: lambda: _run_session_detection_stage(
                normalized_config.dataset_path,
                run_paths.stage_directories[PipelineStage.SESSION_DETECTION],
                results["signal_discovery"]["reference_signal_uuid"],
                normalized_config.session_gap_seconds,
            ),
            PipelineStage.CYCLE_DETECTION: lambda: _run_cycle_detection_stage(
                normalized_config.dataset_path,
                run_paths.stage_directories[PipelineStage.CYCLE_DETECTION],
                normalized_config.experiment,
                results["signal_discovery"]["reference_signal_uuid"],
                results["session_detection"]["sessions"],
                normalized_config.movement_threshold,
            ),
            PipelineStage.MULTI_SENSOR_EXTRACTION: lambda: _run_multi_sensor_extraction_stage(
                normalized_config.dataset_path,
                run_paths.stage_directories[PipelineStage.MULTI_SENSOR_EXTRACTION],
                uuid_signal_info,
                int_signal_info,
                normalized_config.experiment,
                results["cycle_detection"]["cycles"],
                _as_path(results["cycle_detection"]["output_paths"]["cycles_parquet"]),
                normalized_config.max_cycles_to_extract,
                normalized_config.extract_all_cycles,
                normalized_config.cycle_batch_size,
                normalized_config.resume_extraction,
                normalized_config.overwrite_existing,
                tuple(normalized_config.selected_extraction_signals),
                normalized_config.validation_cycle_count,
                tuple(normalized_config.required_validation_signals),
                normalized_config.minimum_samples_per_validation_cycle,
                normalized_config.require_consecutive_validation_cycles,
                normalized_config.max_cycles_to_scan_for_validation,
                normalized_config.generate_validation_html,
                normalized_config.generate_cycle_features,
                normalized_config.parquet_compression,
            ),
            PipelineStage.CYCLE_QUALITY_PROFILING: lambda: _run_cycle_quality_profiling_stage(
                run_paths.stage_directories[PipelineStage.CYCLE_QUALITY_PROFILING],
                results["multi_sensor_extraction"]["measurements_root"],
                results["multi_sensor_extraction"]["cycles_extracted"],
                normalized_config.quality_profiling_batch_size,
            ),
            PipelineStage.VALIDATION_RULE_GENERATION: lambda: _run_validation_rule_generation_stage(
                run_paths.stage_directories[PipelineStage.VALIDATION_RULE_GENERATION],
                results["cycle_quality_profiling"]["signal_quality_metrics"],
                results["cycle_quality_profiling"]["cycle_quality_profile"],
                normalized_config.signal_roles,
                normalized_config.validation_rule_generation,
                dataset_name,
                normalized_config.experiment,
            ),
            PipelineStage.DATASET_VALIDATION: lambda: _run_dataset_validation_stage(
                run_paths.stage_directories[PipelineStage.DATASET_VALIDATION],
                results["cycle_quality_profiling"]["signal_quality_metrics"],
                results["cycle_quality_profiling"]["cycle_quality_profile"],
                results["validation_rule_generation"]["validation_thresholds"],
                normalized_config.signal_roles,
                normalized_config.dataset_validation,
                dataset_name,
                normalized_config.experiment,
            ),
        }

        for stage in STAGE_ORDER:
            _ensure_stage_is_implemented(stage)
            stage_started = time.perf_counter()
            logger.info(
                "Running stage=%s dataset=%s experiment=%s stop_after=%s output=%s",
                stage.value,
                dataset_name,
                normalized_config.experiment,
                stop_stage.value,
                run_paths.stage_directories[stage],
            )
            stage_result = stage_runners[stage]()
            stage_duration = time.perf_counter() - stage_started
            stage_result["execution_time_seconds"] = stage_duration
            results[stage.value] = stage_result
            manifest["completed_stages"].append(stage.value)
            manifest["generated_output_paths"][stage.value] = stage_result["output_paths"]
            manifest["row_counts"][stage.value] = stage_result.get("row_counts", {})
            _write_manifest(run_paths.manifest_path, manifest)
            logger.info(
                "Completed stage=%s status=success duration_seconds=%.3f",
                stage.value,
                stage_duration,
            )
            if stage == stop_stage:
                break

        manifest["status"] = "success"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error_message"] = str(exc)
        logger.exception("Pipeline failed for dataset=%s experiment=%s", dataset_name, normalized_config.experiment)
        raise
    finally:
        end_time = datetime.now()
        manifest["end_time"] = end_time.isoformat()
        _write_manifest(run_paths.manifest_path, manifest)

    results["run"] = {
        "dataset_name": dataset_name,
        "experiment": normalized_config.experiment,
        "stop_after": stop_stage.value,
        "run_directory": str(run_paths.run_directory),
        "manifest_path": str(run_paths.manifest_path),
        "completed_stages": list(manifest["completed_stages"]),
        "runtime_seconds": (end_time - start_time).total_seconds(),
        "status": manifest["status"],
    }
    return results
