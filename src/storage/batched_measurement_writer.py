"""Batched, resumable, partitioned Parquet writer for multi-sensor measurements."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.preprocessing.batch_extraction import (
    SIGNAL_SUMMARY_COLUMNS,
    _extract_cycle_batch_with_summary,
    iter_cycle_batches,
)

logger = logging.getLogger(__name__)

CHECKPOINT_FILE_NAME = "extraction_checkpoint.json"
MEASUREMENTS_DIRECTORY_NAME = "measurements"
SIGNAL_SUMMARY_FILE_NAME = "signal_window_summary.parquet"


@dataclass(slots=True)
class BatchExtractionResult:
    """Outcome of one batched, resumable measurement-extraction run."""

    written_files: list[Path] = field(default_factory=list)
    batch_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    processed_cycle_count: int = 0
    failed_cycle_count: int = 0
    signal_window_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    checkpoint_path: Path | None = None


def _config_fingerprint(
    experiment: str,
    cycle_batch_size: int,
    selected_signals: Sequence[str] | None,
    cycles_df: pd.DataFrame,
) -> str:
    """Build a stable hash describing the relevant run configuration.

    Any change to the experiment, batch size, selected signals, or the cycle
    index itself must invalidate a previous checkpoint, so all of those are
    folded into the fingerprint.
    """

    cycle_ids = cycles_df["cycle_id"].to_numpy() if "cycle_id" in cycles_df.columns else []
    fingerprint_payload = {
        "experiment": experiment,
        "cycle_batch_size": int(cycle_batch_size),
        "selected_signals": sorted(selected_signals) if selected_signals else [],
        "cycle_count": int(len(cycles_df)),
        "first_cycle_id": int(cycle_ids[0]) if len(cycle_ids) else None,
        "last_cycle_id": int(cycle_ids[-1]) if len(cycle_ids) else None,
        "cycle_id_checksum": int(pd.Series(cycle_ids).sum()) if len(cycle_ids) else 0,
    }
    payload_bytes = json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload_bytes).hexdigest()


def _load_checkpoint(checkpoint_path: Path) -> dict[str, object] | None:
    """Load an existing checkpoint file, if any."""

    if not checkpoint_path.exists():
        return None
    return json.loads(checkpoint_path.read_text(encoding="utf-8"))


def _write_checkpoint(checkpoint_path: Path, checkpoint: dict[str, object]) -> None:
    """Persist the current checkpoint state."""

    checkpoint["latest_update_time"] = datetime.now().isoformat()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True), encoding="utf-8")


def write_measurement_batches(
    base_dir: Path,
    cycles_df: pd.DataFrame,
    uuid_signal_info: pd.DataFrame,
    int_signal_info: pd.DataFrame,
    experiment: str,
    output_root: Path,
    cycle_batch_size: int = 500,
    selected_signals: Sequence[str] | None = None,
    resume: bool = True,
    parquet_compression: str = "zstd",
) -> BatchExtractionResult:
    """Extract every cycle in bounded batches and write partitioned Parquet output.

    Output is partitioned by ``experiment`` and ``session_id`` only (never by
    ``cycle_id``), with one Parquet file per processed cycle batch (split by
    session when a batch spans more than one session). A JSON checkpoint next
    to the measurements directory records completed batches so a later call
    with ``resume=True`` can skip already-written work; an incompatible
    configuration change raises a clear error instead of silently resuming.
    """

    output_root = Path(output_root)
    measurements_root = output_root / MEASUREMENTS_DIRECTORY_NAME
    checkpoint_path = output_root / CHECKPOINT_FILE_NAME

    fingerprint = _config_fingerprint(experiment, cycle_batch_size, selected_signals, cycles_df)
    existing_checkpoint = _load_checkpoint(checkpoint_path) if resume else None

    if existing_checkpoint is not None:
        if existing_checkpoint.get("dataset") != str(base_dir):
            raise ValueError(
                "Cannot resume multi-sensor extraction: dataset path changed. "
                "Re-run with resume=False (or delete the checkpoint) to start fresh."
            )
        if existing_checkpoint.get("configuration_fingerprint") != fingerprint:
            raise ValueError(
                "Cannot resume multi-sensor extraction: the experiment, selected "
                "signals, cycle batch size, or cycle index changed since the last "
                "run. Re-run with resume=False (or delete the checkpoint) to start "
                "fresh with the new configuration."
            )
        completed_batch_indices = set(existing_checkpoint.get("completed_batch_ids", []))
        written_files = [Path(path) for path in existing_checkpoint.get("output_file_paths", [])]
        failed_cycle_ids: list[int] = list(existing_checkpoint.get("failed_cycle_ids", []))
        signal_summary_frames: list[pd.DataFrame] = []
        existing_summary_path = output_root / SIGNAL_SUMMARY_FILE_NAME
        if existing_summary_path.exists():
            signal_summary_frames.append(pd.read_parquet(existing_summary_path))
        batch_summary_rows: list[dict[str, object]] = list(existing_checkpoint.get("batch_summary_rows", []))
        processed_cycle_count = int(existing_checkpoint.get("processed_cycle_count", 0))
        checkpoint = existing_checkpoint
        logger.info(
            "Resuming multi-sensor extraction: %d batch(es) already completed",
            len(completed_batch_indices),
        )
    else:
        completed_batch_indices = set()
        written_files = []
        failed_cycle_ids = []
        signal_summary_frames = []
        batch_summary_rows = []
        processed_cycle_count = 0
        checkpoint = {
            "dataset": str(base_dir),
            "experiment": experiment,
            "configuration_fingerprint": fingerprint,
            "completed_batch_ids": [],
            "processed_cycle_ranges": [],
            "output_file_paths": [],
            "failed_cycle_ids": [],
            "batch_summary_rows": [],
            "processed_cycle_count": 0,
            "start_time": datetime.now().isoformat(),
            "latest_update_time": None,
            "status": "running",
        }
        _write_checkpoint(checkpoint_path, checkpoint)

    for batch_index, cycle_batch in enumerate(iter_cycle_batches(cycles_df, cycle_batch_size), start=1):
        if batch_index in completed_batch_indices:
            continue

        batch_started = datetime.now()
        measurements_df, summary_df = _extract_cycle_batch_with_summary(
            base_dir=base_dir,
            cycle_batch=cycle_batch,
            uuid_signal_info=uuid_signal_info,
            int_signal_info=int_signal_info,
            experiment=experiment,
            selected_signals=selected_signals,
        )

        batch_cycle_ids = cycle_batch["cycle_id"].astype(int).tolist()
        batch_written_paths: list[Path] = []

        if not measurements_df.empty:
            for session_id, session_measurements_df in measurements_df.groupby("session_id"):
                session_directory = (
                    measurements_root / f"experiment={experiment}" / f"session_id={session_id}"
                )
                session_directory.mkdir(parents=True, exist_ok=True)
                batch_path = session_directory / f"batch_{batch_index:06d}.parquet"
                session_measurements_df.to_parquet(
                    batch_path, index=False, compression=parquet_compression
                )
                batch_written_paths.append(batch_path)
                written_files.append(batch_path)
        else:
            logger.warning(
                "Batch %d produced no measurements for %d cycle(s); no Parquet file written",
                batch_index,
                len(batch_cycle_ids),
            )

        # A cycle only counts as an extraction *failure* when it hit a real
        # technical error (e.g. unreadable measurement partitions). An
        # otherwise-valid cycle missing an optional signal (such as
        # duty-cycled vibration) is represented explicitly per-signal and
        # must never be treated as a failed extraction.
        failed_in_batch = (
            summary_df.loc[summary_df["extraction_error"], "cycle_id"].unique().tolist()
            if not summary_df.empty and "extraction_error" in summary_df.columns
            else []
        )
        for cycle_id in failed_in_batch:
            if int(cycle_id) not in failed_cycle_ids:
                failed_cycle_ids.append(int(cycle_id))

        signal_summary_frames.append(summary_df)
        processed_cycle_count += len(batch_cycle_ids)
        completed_batch_indices.add(batch_index)

        batch_duration_seconds = (datetime.now() - batch_started).total_seconds()
        batch_summary_rows.append(
            {
                "batch_id": batch_index,
                "cycle_count": len(batch_cycle_ids),
                "first_cycle_id": batch_cycle_ids[0] if batch_cycle_ids else None,
                "last_cycle_id": batch_cycle_ids[-1] if batch_cycle_ids else None,
                "written_rows": int(len(measurements_df)),
                "output_files": [str(path) for path in batch_written_paths],
                "duration_seconds": batch_duration_seconds,
            }
        )
        logger.info(
            "Processed batch %d: %d cycle(s), %d measurement row(s), %.3fs",
            batch_index,
            len(batch_cycle_ids),
            len(measurements_df),
            batch_duration_seconds,
        )

        checkpoint["completed_batch_ids"] = sorted(completed_batch_indices)
        checkpoint["output_file_paths"] = [str(path) for path in written_files]
        checkpoint["failed_cycle_ids"] = sorted(failed_cycle_ids)
        checkpoint["batch_summary_rows"] = batch_summary_rows
        checkpoint["processed_cycle_count"] = processed_cycle_count
        checkpoint["status"] = "running"
        _write_checkpoint(checkpoint_path, checkpoint)

        # Release this batch's DataFrames before moving on to the next one so
        # memory usage stays bounded regardless of the total cycle count.
        del measurements_df, summary_df

    checkpoint["status"] = "completed"
    _write_checkpoint(checkpoint_path, checkpoint)

    signal_window_summary_df = (
        pd.concat(signal_summary_frames, ignore_index=True)
        if signal_summary_frames
        else pd.DataFrame(columns=list(SIGNAL_SUMMARY_COLUMNS))
    )
    signal_summary_path = output_root / SIGNAL_SUMMARY_FILE_NAME
    signal_summary_path.parent.mkdir(parents=True, exist_ok=True)
    signal_window_summary_df.to_parquet(signal_summary_path, index=False)

    batch_summary_df = pd.DataFrame(batch_summary_rows)

    return BatchExtractionResult(
        written_files=written_files,
        batch_summary=batch_summary_df,
        processed_cycle_count=processed_cycle_count,
        failed_cycle_count=len(failed_cycle_ids),
        signal_window_summary=signal_window_summary_df,
        checkpoint_path=checkpoint_path,
    )
