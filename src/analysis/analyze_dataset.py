"""Orchestrate dataset-level analysis using reusable preprocessing modules."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.multi_sensor_cycle_extraction import list_experiment_signals
from src.preprocessing.session_detection import detect_recording_sessions
from src.preprocessing.time_gap_analysis import analyze_time_gaps, save_time_gap_statistics
from src.utils.data_loader import build_int_signal_info, build_uuid_signal_info, find_signals

logger = logging.getLogger(__name__)
BASE_DIR = Path("/data/ERA/D63_Nr7_8")
STATISTICS_DIR = PROJECT_ROOT / "outputs" / "statistics"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"


def _discover_experiments(
    uuid_signal_info: pd.DataFrame,
    int_signal_info: pd.DataFrame,
) -> list[str]:
    """Return discovered experiment names from metadata paths."""

    combined_paths = pd.concat(
        [uuid_signal_info["path"], int_signal_info["path"]],
        ignore_index=True,
    ).dropna()

    experiments = {
        parts[1]
        for path in combined_paths
        if len(parts := str(path).split("/")) >= 2 and parts[1]
    }
    return sorted(experiments)


def _build_signal_availability_table(
    experiment: str,
    uuid_signal_info: pd.DataFrame,
    int_signal_info: pd.DataFrame,
) -> pd.DataFrame:
    """Return the metadata-driven signal availability table for one experiment."""

    signals_df = list_experiment_signals(
        uuid_signal_info=uuid_signal_info,
        int_signal_info=int_signal_info,
        experiment=experiment,
    ).copy()
    if signals_df.empty:
        return signals_df

    signals_df.insert(0, "experiment", experiment)
    return signals_df


def _build_sampling_summary(
    experiment: str,
    statistics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Reshape time-gap statistics into one summary row."""

    values = statistics_df.set_index("metric")["value"]
    summary_row = {
        "experiment": experiment,
        "signal_id_uuid": values.get("signal_id_uuid"),
        "number_of_samples": values.get("number_of_samples"),
        "minimum_gap_seconds": values.get("minimum_gap_seconds"),
        "median_gap_seconds": values.get("median_gap_seconds"),
        "mean_gap_seconds": values.get("mean_gap_seconds"),
        "percentile_95_gap_seconds": values.get("percentile_95_gap_seconds"),
        "percentile_99_gap_seconds": values.get("percentile_99_gap_seconds"),
        "maximum_gap_seconds": values.get("maximum_gap_seconds"),
    }
    return pd.DataFrame([summary_row])


def _build_session_summary(
    experiment: str,
    signal_id_uuid: str,
    sessions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize recording sessions for one position signal."""

    if sessions_df.empty:
        return pd.DataFrame(
            [
                {
                    "experiment": experiment,
                    "signal_id_uuid": signal_id_uuid,
                    "number_of_sessions": 0,
                    "minimum_duration_seconds": pd.NA,
                    "median_duration_seconds": pd.NA,
                    "maximum_duration_seconds": pd.NA,
                    "minimum_samples": pd.NA,
                    "median_samples": pd.NA,
                    "maximum_samples": pd.NA,
                    "maximum_internal_gap_seconds": pd.NA,
                }
            ]
        )

    return pd.DataFrame(
        [
            {
                "experiment": experiment,
                "signal_id_uuid": signal_id_uuid,
                "number_of_sessions": len(sessions_df),
                "minimum_duration_seconds": sessions_df["duration_seconds"].min(),
                "median_duration_seconds": sessions_df["duration_seconds"].median(),
                "maximum_duration_seconds": sessions_df["duration_seconds"].max(),
                "minimum_samples": sessions_df["number_of_samples"].min(),
                "median_samples": sessions_df["number_of_samples"].median(),
                "maximum_samples": sessions_df["number_of_samples"].max(),
                "maximum_internal_gap_seconds": sessions_df["maximum_internal_gap_seconds"].max(),
            }
        ]
    )


def _print_experiment_report(
    experiment: str,
    signal_table: pd.DataFrame,
    sampling_summary: pd.DataFrame | None,
    session_summary: pd.DataFrame | None,
    note: str | None = None,
) -> None:
    """Print a readable console report for one experiment."""

    print(f"\n=== {experiment} ===")
    print(f"signals discovered: {len(signal_table)}")
    if not signal_table.empty:
        print(
            signal_table[["signal_name", "source", "unit_code", "unit_symbol", "path"]].to_string(
                index=False
            )
        )

    if note is not None:
        print(note)
        return

    if sampling_summary is not None and not sampling_summary.empty:
        sampling_row = sampling_summary.iloc[0]
        print(
            "sampling: "
            f"samples={sampling_row['number_of_samples']} "
            f"gap_min={sampling_row['minimum_gap_seconds']} "
            f"gap_median={sampling_row['median_gap_seconds']} "
            f"gap_mean={sampling_row['mean_gap_seconds']} "
            f"gap_p95={sampling_row['percentile_95_gap_seconds']} "
            f"gap_p99={sampling_row['percentile_99_gap_seconds']} "
            f"gap_max={sampling_row['maximum_gap_seconds']}"
        )

    if session_summary is not None and not session_summary.empty:
        session_row = session_summary.iloc[0]
        print(
            "sessions: "
            f"count={session_row['number_of_sessions']} "
            f"duration_min={session_row['minimum_duration_seconds']} "
            f"duration_median={session_row['median_duration_seconds']} "
            f"duration_max={session_row['maximum_duration_seconds']} "
            f"samples_min={session_row['minimum_samples']} "
            f"samples_median={session_row['median_samples']} "
            f"samples_max={session_row['maximum_samples']} "
            f"max_internal_gap={session_row['maximum_internal_gap_seconds']}"
        )


def main() -> None:
    """Run dataset-wide orchestration using existing reusable analysis modules."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    STATISTICS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    uuid_signal_info = build_uuid_signal_info(BASE_DIR)
    int_signal_info = build_int_signal_info(BASE_DIR)
    experiments = _discover_experiments(uuid_signal_info, int_signal_info)

    all_signal_tables: list[pd.DataFrame] = []
    sampling_summaries: list[pd.DataFrame] = []
    session_summaries: list[pd.DataFrame] = []

    print("Dataset analysis report")
    print(f"base_dir: {BASE_DIR}")
    print(f"experiments discovered: {', '.join(experiments)}")

    for experiment in experiments:
        signal_table = _build_signal_availability_table(
            experiment,
            uuid_signal_info,
            int_signal_info,
        )
        all_signal_tables.append(signal_table)
        signal_table.to_csv(REPORTS_DIR / f"{experiment.lower()}_signals.csv", index=False)

        position_signals = find_signals(
            uuid_signal_info,
            path_contains=experiment,
            unit_code="position",
        )
        if len(position_signals) != 1:
            note = (
                f"position analysis skipped: expected exactly one position signal, found {len(position_signals)}."
            )
            _print_experiment_report(
                experiment,
                signal_table,
                sampling_summary=None,
                session_summary=None,
                note=note,
            )
            continue

        signal_id_uuid = str(position_signals.iloc[0]["signal_id_uuid"])
        statistics_df, _ = analyze_time_gaps(BASE_DIR, signal_id_uuid)
        save_time_gap_statistics(
            statistics_df,
            STATISTICS_DIR / f"{experiment.lower()}_time_gap_statistics.csv",
        )
        sampling_summary = _build_sampling_summary(experiment, statistics_df)
        sampling_summaries.append(sampling_summary)

        sessions_df = detect_recording_sessions(BASE_DIR, signal_id_uuid)
        sessions_df.to_csv(REPORTS_DIR / f"{experiment.lower()}_position_sessions.csv", index=False)
        session_summary = _build_session_summary(experiment, signal_id_uuid, sessions_df)
        session_summaries.append(session_summary)

        _print_experiment_report(
            experiment,
            signal_table,
            sampling_summary=sampling_summary,
            session_summary=session_summary,
        )

    signal_availability_df = (
        pd.concat(all_signal_tables, ignore_index=True) if all_signal_tables else pd.DataFrame()
    )
    signal_availability_df.to_csv(STATISTICS_DIR / "signal_availability.csv", index=False)

    if sampling_summaries:
        sampling_summary_df = pd.concat(sampling_summaries, ignore_index=True)
        sampling_summary_df.to_csv(STATISTICS_DIR / "sampling_statistics_summary.csv", index=False)

    if session_summaries:
        session_summary_df = pd.concat(session_summaries, ignore_index=True)
        session_summary_df.to_csv(STATISTICS_DIR / "session_statistics_summary.csv", index=False)

    experiments_df = pd.DataFrame({"experiment": experiments})
    experiments_df.to_csv(STATISTICS_DIR / "experiments.csv", index=False)


if __name__ == "__main__":
    main()
