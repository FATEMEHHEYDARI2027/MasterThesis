"""Analyze exploratory candidate cycles for the Versuch1 position signal."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.cycle_detection import detect_candidate_cycles  # noqa: E402
from src.utils.data_loader import build_uuid_signal_info, find_signals  # noqa: E402
from src.utils.measurement_loader import load_uuid_signal  # noqa: E402

BASE_DIR = Path("/data/ERA/D63_Nr7_8")
START_TIME = "2025-12-23 00:10:00"
END_TIME = "2025-12-23 00:20:00"
# Temporary exploratory threshold that must be validated from the observed
# cycle statistics and annotated plots before it becomes a final rule.
MOVEMENT_THRESHOLD = 1.0
OUTPUT_CSV_PATH = PROJECT_ROOT / "outputs" / "datasets" / "versuch1_candidate_cycles.csv"
OUTPUT_FIGURE_PATH = PROJECT_ROOT / "outputs" / "figures" / "versuch1_candidate_cycles.png"


def _print_series_summary(label: str, values: pd.Series) -> None:
    """Print minimum, median, and maximum values for a numeric series."""

    print(f"{label} min: {values.min()}")
    print(f"{label} median: {values.median()}")
    print(f"{label} max: {values.max()}")


def main() -> None:
    """Run exploratory candidate-cycle detection for a bounded session window."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    uuid_info = build_uuid_signal_info(BASE_DIR)
    position_signals = find_signals(
        uuid_info,
        path_contains="Versuch1",
        unit_code="position",
    )
    if len(position_signals) != 1:
        raise ValueError(
            f"Expected exactly one Versuch1 position signal, found {len(position_signals)}."
        )

    signal_id_uuid = str(position_signals.iloc[0]["signal_id_uuid"])
    position_df = load_uuid_signal(
        BASE_DIR,
        signal_id_uuid,
        start_time=START_TIME,
        end_time=END_TIME,
    )
    if position_df.empty:
        raise ValueError("No position samples were loaded for the requested cycle-analysis window.")

    cycles_df = detect_candidate_cycles(
        position_df,
        movement_threshold=MOVEMENT_THRESHOLD,
    )

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    cycles_df.to_csv(OUTPUT_CSV_PATH, index=False)

    OUTPUT_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(position_df["time"], position_df["value"], linewidth=1.0)
    for start_time in cycles_df["start_time"]:
        ax.axvline(pd.Timestamp(start_time), color="green", alpha=0.4, linewidth=1.0)
    for end_time in cycles_df["end_time"]:
        ax.axvline(pd.Timestamp(end_time), color="red", alpha=0.4, linewidth=1.0)
    ax.set_title("Versuch1 Position Candidate Cycles")
    ax.set_xlabel("Time")
    ax.set_ylabel("Position Value")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE_PATH, dpi=150)
    plt.close(fig)

    print(f"signal_id_uuid: {signal_id_uuid}")
    print(f"start_time: {START_TIME}")
    print(f"end_time: {END_TIME}")
    print(f"movement_threshold: {MOVEMENT_THRESHOLD}")
    print(f"number of candidate cycles: {len(cycles_df)}")
    if not cycles_df.empty:
        _print_series_summary("cycle duration (seconds)", cycles_df["duration_seconds"])
        _print_series_summary("sample count", cycles_df["number_of_samples"])
        _print_series_summary("peak position", cycles_df["maximum_position"])
    print(f"saved cycle table: {OUTPUT_CSV_PATH}")
    print(f"saved cycle figure: {OUTPUT_FIGURE_PATH}")


if __name__ == "__main__":
    main()
