"""Diagnose why later Position cycles lose vibration_x/y/z coverage.

This is a read-only, exploratory diagnostic. It does not change the
pipeline, does not classify or reject cycles, and does not lower any sample
threshold. It reuses the existing repository metadata loaders
(``src.utils.data_loader``) and measurement loaders
(``src.utils.measurement_loader``) plus the outputs of an already-completed
pipeline run (cycle index, sessions, and the multi-sensor signal window
summary) to answer:

1. Raw metadata (selected id, source table, first/last raw timestamp, raw
   sample count, overlapping session ids) for every selected signal.
2. Whether each of the first 100 detected Position cycles overlaps
   vibration_x / vibration_y / vibration_z coverage.
3. Whether additional vibration UUIDs or integer signal ids exist that were
   not selected for extraction.
4. Whether Position and vibration signals share timezone, timestamp unit,
   clock origin, and recording sessions.

Usage
-----
    python scripts/diagnose_vibration_coverage.py \\
        --dataset /data/ERA/D63_Nr7_8 \\
        --experiment Versuch1 \\
        --run-directory outputs/D63_Nr7_8/Versuch1/20260717_074708 \\
        --cycle-count 100
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.multi_sensor_cycle_extraction import list_experiment_signals  # noqa: E402
from src.utils.data_loader import (  # noqa: E402
    build_int_signal_info_from_metadata,
    build_uuid_signal_info_from_metadata,
    load_metadata,
)
from src.utils.measurement_loader import (  # noqa: E402
    SIGNAL_DATASET_NAME,
    VIBRATION_DATASET_NAME,
    load_int_signal,
    load_uuid_signal,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CORE_SIGNALS: tuple[str, ...] = (
    "position",
    "velocity",
    "current",
    "pressure",
    "temperature",
    "vibration_x",
    "vibration_y",
    "vibration_z",
)
VIBRATION_SIGNALS: tuple[str, ...] = ("vibration_x", "vibration_y", "vibration_z")


@dataclass(slots=True)
class RawTimeStats:
    """Row-group-derived, metadata-only timing statistics for one signal partition."""

    total_rows: int
    minimum_time: pd.Timestamp | None
    maximum_time: pd.Timestamp | None
    file_count: int
    row_group_bounds: list[tuple[pd.Timestamp, pd.Timestamp, int]]


def _partition_directory(
    dataset_path: Path, source: str, signal_id_uuid: object, signal_id: object
) -> Path | None:
    """Return the on-disk partition directory for one signal, if resolvable."""

    if source == "uuid":
        if pd.isna(signal_id_uuid):
            return None
        return dataset_path / SIGNAL_DATASET_NAME / f"signal_id={signal_id_uuid}"
    if pd.isna(signal_id):
        return None
    return dataset_path / VIBRATION_DATASET_NAME / f"signal_id={int(signal_id)}"


def _raw_time_stats(partition_directory: Path) -> RawTimeStats:
    """Compute row count, min/max time, and per-row-group time bounds from Parquet metadata only.

    Only Parquet footer statistics are read here (no column data pages), so
    this stays cheap even for multi-gigabyte signal partitions.
    """

    if not partition_directory.exists():
        return RawTimeStats(0, None, None, 0, [])

    files = sorted(glob.glob(str(partition_directory / "**" / "*.parquet"), recursive=True))
    total_rows = 0
    minimum_time: pd.Timestamp | None = None
    maximum_time: pd.Timestamp | None = None
    row_group_bounds: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []

    for file_path in files:
        parquet_file = pq.ParquetFile(file_path)
        total_rows += parquet_file.metadata.num_rows
        time_column_index = parquet_file.schema_arrow.get_field_index("time")
        if time_column_index < 0:
            continue
        for row_group_index in range(parquet_file.metadata.num_row_groups):
            column_chunk = parquet_file.metadata.row_group(row_group_index).column(time_column_index)
            statistics = column_chunk.statistics
            if statistics is None or not statistics.has_min_max:
                continue
            row_group_min = pd.Timestamp(statistics.min)
            row_group_max = pd.Timestamp(statistics.max)
            row_group_count = parquet_file.metadata.row_group(row_group_index).num_rows
            row_group_bounds.append((row_group_min, row_group_max, row_group_count))
            minimum_time = row_group_min if minimum_time is None else min(minimum_time, row_group_min)
            maximum_time = row_group_max if maximum_time is None else max(maximum_time, row_group_max)

    return RawTimeStats(total_rows, minimum_time, maximum_time, len(files), row_group_bounds)


def _overlapping_session_ids(
    row_group_bounds: list[tuple[pd.Timestamp, pd.Timestamp, int]], sessions_df: pd.DataFrame
) -> list[int]:
    """Return the session ids whose window overlaps at least one row group's time bounds."""

    overlapping: list[int] = []
    for session_row in sessions_df.itertuples(index=False):
        session_start = pd.Timestamp(session_row.start_time)
        session_end = pd.Timestamp(session_row.end_time)
        has_overlap = any(
            rg_min <= session_end and rg_max >= session_start
            for rg_min, rg_max, _ in row_group_bounds
        )
        if has_overlap:
            overlapping.append(int(session_row.session_id))
    return overlapping


def _merge_burst_segments(
    row_group_bounds: list[tuple[pd.Timestamp, pd.Timestamp, int]], gap_seconds: float
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Merge row-group time bounds into contiguous coverage segments.

    Two row groups are merged into the same segment when the gap between
    them is smaller than ``gap_seconds``; this approximates (without reading
    any raw sample) the duty-cycled burst structure of a signal.
    """

    if not row_group_bounds:
        return []

    ordered_bounds = sorted(row_group_bounds, key=lambda bound: bound[0])
    segments: list[tuple[pd.Timestamp, pd.Timestamp]] = [
        (ordered_bounds[0][0], ordered_bounds[0][1])
    ]
    for row_group_min, row_group_max, _ in ordered_bounds[1:]:
        last_start, last_end = segments[-1]
        if (row_group_min - last_end).total_seconds() <= gap_seconds:
            segments[-1] = (last_start, max(last_end, row_group_max))
        else:
            segments.append((row_group_min, row_group_max))
    return segments


def _resolve_run_directory(run_directory: Path | None, output_root: Path, dataset_name: str, experiment: str) -> Path:
    """Resolve the most recent completed pipeline run directory if none was given."""

    if run_directory is not None:
        return run_directory

    experiment_root = output_root / dataset_name / experiment
    candidate_runs = sorted(
        (path for path in experiment_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    for candidate in candidate_runs:
        if (candidate / "multi_sensor" / "signal_window_summary.parquet").exists():
            return candidate
    if candidate_runs:
        return candidate_runs[0]
    raise FileNotFoundError(f"No pipeline run directories found under {experiment_root}")


def _build_signal_time_coverage(
    signals_df: pd.DataFrame,
    dataset_path: Path,
    sessions_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, RawTimeStats]]:
    """Build the per-signal raw time-coverage diagnostic table."""

    rows: list[dict[str, object]] = []
    stats_by_signal: dict[str, RawTimeStats] = {}

    for signal_row in signals_df[signals_df["signal_name"].isin(CORE_SIGNALS)].itertuples(index=False):
        source_table = SIGNAL_DATASET_NAME if signal_row.source == "uuid" else VIBRATION_DATASET_NAME
        partition_directory = _partition_directory(
            dataset_path, signal_row.source, signal_row.signal_id_uuid, signal_row.signal_id
        )
        stats = (
            _raw_time_stats(partition_directory)
            if partition_directory is not None
            else RawTimeStats(0, None, None, 0, [])
        )
        stats_by_signal[str(signal_row.signal_name)] = stats
        session_ids = _overlapping_session_ids(stats.row_group_bounds, sessions_df)
        session_windows = sessions_df[sessions_df["session_id"].isin(session_ids)]

        rows.append(
            {
                "signal_name": signal_row.signal_name,
                "source": signal_row.source,
                "source_table": source_table,
                "selected_uuid": (
                    signal_row.signal_id_uuid if signal_row.source == "uuid" else pd.NA
                ),
                "selected_int_id": (
                    int(signal_row.signal_id) if signal_row.source == "int" else pd.NA
                ),
                "first_raw_timestamp": stats.minimum_time,
                "last_raw_timestamp": stats.maximum_time,
                "raw_sample_count": stats.total_rows,
                "detected_session_ids": ",".join(str(sid) for sid in session_ids),
                "session_start_times": ",".join(
                    str(ts) for ts in session_windows["start_time"].tolist()
                ),
                "session_end_times": ",".join(
                    str(ts) for ts in session_windows["end_time"].tolist()
                ),
            }
        )

    return pd.DataFrame(rows), stats_by_signal


def _build_cycle_vibration_overlap(
    cycles_df: pd.DataFrame,
    signal_window_summary_path: Path,
) -> pd.DataFrame:
    """Report whether each of the first N cycles overlaps vibration coverage.

    Overlap is derived from the already-extracted ``signal_window_summary``
    produced by the multi-sensor extraction stage: a cycle "overlaps" one
    vibration axis when that axis has at least one raw sample
    (``is_missing`` is ``False``) inside the cycle's exact extraction window.
    """

    summary_df = pd.read_parquet(signal_window_summary_path)
    vibration_summary_df = summary_df[summary_df["signal_name"].isin(VIBRATION_SIGNALS)]

    pivot_df = vibration_summary_df.pivot_table(
        index="cycle_id",
        columns="signal_name",
        values="is_missing",
        aggfunc="first",
    )
    pivot_df = pivot_df.rename(columns={name: f"{name}_overlap" for name in VIBRATION_SIGNALS})
    for name in VIBRATION_SIGNALS:
        column = f"{name}_overlap"
        if column in pivot_df.columns:
            pivot_df[column] = ~pivot_df[column].astype(bool)
        else:
            pivot_df[column] = False

    merged_df = cycles_df[["cycle_id", "session_id", "start_time", "end_time"]].merge(
        pivot_df.reset_index(), on="cycle_id", how="left"
    )
    for name in VIBRATION_SIGNALS:
        column = f"{name}_overlap"
        merged_df[column] = merged_df[column].fillna(False).astype(bool)
    merged_df["any_vibration_axis_overlap"] = merged_df[
        [f"{name}_overlap" for name in VIBRATION_SIGNALS]
    ].any(axis=1)
    return merged_df


def _build_vibration_signal_candidates(
    uuid_signal_info: pd.DataFrame,
    int_signal_info: pd.DataFrame,
    dataset_path: Path,
    experiment: str,
    selected_extraction_signal_names: set[str],
) -> pd.DataFrame:
    """List every vibration-unit signal in metadata, selected or not.

    This covers both other experiments (Versuch2/Versuch3) and any
    unselected in-experiment integer vibration channels
    (``D63/<experiment>/Sensor_1..4``), so a reviewer can see every
    candidate vibration source before concluding one was wrongly chosen.
    ``selected_extraction_signal_names`` must be the exact
    ``selected_extraction_signals`` list actually used by the pipeline run
    (from its manifest), not just every signal discovered in metadata.
    """

    uuid_vibration_df = uuid_signal_info[uuid_signal_info["unit_code"] == "vibration"].copy()
    uuid_vibration_df["source"] = "uuid"
    int_vibration_df = int_signal_info[int_signal_info["unit_code"] == "vibration"].copy()
    int_vibration_df["source"] = "int"

    candidates_df = pd.concat([uuid_vibration_df, int_vibration_df], ignore_index=True, sort=False)

    experiment_signals_df = list_experiment_signals(
        uuid_signal_info=uuid_signal_info,
        int_signal_info=int_signal_info,
        experiment=experiment,
    )
    selected_paths = set(
        experiment_signals_df.loc[
            experiment_signals_df["signal_name"].isin(selected_extraction_signal_names), "path"
        ]
    )

    rows: list[dict[str, object]] = []
    for candidate_row in candidates_df.itertuples(index=False):
        partition_directory = _partition_directory(
            dataset_path, candidate_row.source, candidate_row.signal_id_uuid, candidate_row.signal_id
        )
        stats = (
            _raw_time_stats(partition_directory)
            if partition_directory is not None
            else RawTimeStats(0, None, None, 0, [])
        )
        belongs_to_current_experiment = experiment.lower() in str(candidate_row.path).lower()
        is_selected = candidate_row.path in selected_paths
        rows.append(
            {
                "path": candidate_row.path,
                "source": candidate_row.source,
                "signal_id_uuid": candidate_row.signal_id_uuid,
                "signal_id": candidate_row.signal_id,
                "belongs_to_current_experiment": belongs_to_current_experiment,
                "currently_selected_for_extraction": is_selected,
                "first_raw_timestamp": stats.minimum_time,
                "last_raw_timestamp": stats.maximum_time,
                "raw_sample_count": stats.total_rows,
            }
        )
    return pd.DataFrame(rows).sort_values(["belongs_to_current_experiment", "path"], ascending=[False, True])


def _plot_signal_time_coverage(
    stats_by_signal: dict[str, RawTimeStats],
    cycles_head_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save an interactive HTML plot of signal coverage and the first N cycle windows."""

    fig = go.Figure()
    signal_order = list(CORE_SIGNALS)

    for row_index, signal_name in enumerate(signal_order):
        stats = stats_by_signal.get(signal_name)
        if stats is None or stats.minimum_time is None:
            continue
        # Full-lifetime context bar (single span).
        fig.add_trace(
            go.Scatter(
                x=[stats.minimum_time, stats.maximum_time],
                y=[signal_name, signal_name],
                mode="lines",
                line=dict(color="lightsteelblue", width=14),
                name=f"{signal_name} full lifetime span",
                legendgroup=signal_name,
                showlegend=False,
                hovertemplate=(
                    f"{signal_name} full span<br>start: %{{x[0]}}<br>end: %{{x[1]}}<extra></extra>"
                ),
            )
        )
        # Fine-grained burst segments, merged with a short gap threshold so
        # duty-cycled recording gaps stay visible when zooming in.
        segments = _merge_burst_segments(stats.row_group_bounds, gap_seconds=1.0)
        for segment_start, segment_end in segments:
            fig.add_trace(
                go.Scatter(
                    x=[segment_start, segment_end],
                    y=[signal_name, signal_name],
                    mode="lines",
                    line=dict(color="steelblue", width=10),
                    showlegend=False,
                    hovertemplate=(
                        f"{signal_name} burst<br>start: %{{x[0]}}<br>end: %{{x[1]}}<extra></extra>"
                    ),
                )
            )

    for cycle_row in cycles_head_df.itertuples(index=False):
        fig.add_vrect(
            x0=cycle_row.start_time,
            x1=cycle_row.end_time,
            fillcolor="orange",
            opacity=0.25,
            line_width=0,
        )

    fig.update_layout(
        title=(
            "Signal time coverage vs. first "
            f"{len(cycles_head_df)} Position cycles "
            "(orange bands = cycle windows; steel-blue = actual burst coverage)"
        ),
        xaxis_title="Time",
        yaxis_title="Signal",
        hovermode="closest",
        xaxis=dict(rangeslider=dict(visible=True)),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)


def run_diagnostics(
    dataset_path: Path,
    experiment: str,
    run_directory: Path,
    cycle_count: int,
    diagnostics_directory: Path,
) -> None:
    """Run the full read-only vibration-coverage diagnostic and write all outputs."""

    metadata_frames = load_metadata(dataset_path)
    uuid_signal_info = build_uuid_signal_info_from_metadata(metadata_frames)
    int_signal_info = build_int_signal_info_from_metadata(metadata_frames)

    signals_df = list_experiment_signals(
        uuid_signal_info=uuid_signal_info,
        int_signal_info=int_signal_info,
        experiment=experiment,
    )

    sessions_df = pd.read_csv(run_directory / "sessions" / "sessions.csv", parse_dates=["start_time", "end_time"])
    cycles_df = pd.read_parquet(run_directory / "cycles" / "cycles.parquet")
    cycles_head_df = cycles_df.head(cycle_count).copy()

    signal_time_coverage_df, stats_by_signal = _build_signal_time_coverage(
        signals_df, dataset_path, sessions_df
    )
    cycle_vibration_overlap_df = _build_cycle_vibration_overlap(
        cycles_head_df,
        run_directory / "multi_sensor" / "signal_window_summary.parquet",
    )
    manifest = json.loads((run_directory / "run_manifest.json").read_text(encoding="utf-8"))
    selected_extraction_signal_names = set(
        manifest.get("parameters", {}).get("selected_extraction_signals") or signals_df["signal_name"]
    )
    vibration_signal_candidates_df = _build_vibration_signal_candidates(
        uuid_signal_info,
        int_signal_info,
        dataset_path,
        experiment,
        selected_extraction_signal_names=selected_extraction_signal_names,
    )

    diagnostics_directory.mkdir(parents=True, exist_ok=True)
    signal_time_coverage_path = diagnostics_directory / "signal_time_coverage.csv"
    cycle_vibration_overlap_path = diagnostics_directory / "cycle_vibration_overlap.csv"
    vibration_signal_candidates_path = diagnostics_directory / "vibration_signal_candidates.csv"
    signal_time_coverage_html_path = diagnostics_directory / "signal_time_coverage.html"

    signal_time_coverage_df.to_csv(signal_time_coverage_path, index=False)
    cycle_vibration_overlap_df.to_csv(cycle_vibration_overlap_path, index=False)
    vibration_signal_candidates_df.to_csv(vibration_signal_candidates_path, index=False)
    _plot_signal_time_coverage(stats_by_signal, cycles_head_df, signal_time_coverage_html_path)

    overlap_count = int(cycle_vibration_overlap_df["any_vibration_axis_overlap"].sum())
    logger.info(
        "Wrote diagnostics to %s (%d of %d cycles overlap at least one vibration axis)",
        diagnostics_directory,
        overlap_count,
        len(cycle_vibration_overlap_df),
    )
    print(f"signal_time_coverage.csv:        {signal_time_coverage_path}")
    print(f"cycle_vibration_overlap.csv:     {cycle_vibration_overlap_path}")
    print(f"vibration_signal_candidates.csv: {vibration_signal_candidates_path}")
    print(f"signal_time_coverage.html:       {signal_time_coverage_html_path}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Diagnose why later Position cycles lose vibration coverage."
    )
    parser.add_argument("--dataset", required=True, help="Dataset directory path.")
    parser.add_argument("--experiment", required=True, help="Experiment name to inspect.")
    parser.add_argument(
        "--run-directory",
        help=(
            "Existing pipeline run directory (must already contain cycles/, sessions/, and "
            "multi_sensor/signal_window_summary.parquet). Defaults to the most recent run "
            "under --output-root/<dataset-name>/<experiment>."
        ),
    )
    parser.add_argument("--output-root", default="outputs", help="Root output directory (used to auto-detect --run-directory).")
    parser.add_argument("--cycle-count", type=int, default=100, help="Number of leading cycles to inspect.")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""

    args = _parse_args()
    dataset_path = Path(args.dataset).expanduser()
    output_root = Path(args.output_root).expanduser()
    dataset_name = dataset_path.name

    run_directory = _resolve_run_directory(
        Path(args.run_directory).expanduser() if args.run_directory else None,
        output_root,
        dataset_name,
        args.experiment,
    )
    diagnostics_directory = run_directory / "diagnostics"

    run_diagnostics(
        dataset_path=dataset_path,
        experiment=args.experiment,
        run_directory=run_directory,
        cycle_count=args.cycle_count,
        diagnostics_directory=diagnostics_directory,
    )


if __name__ == "__main__":
    main()
