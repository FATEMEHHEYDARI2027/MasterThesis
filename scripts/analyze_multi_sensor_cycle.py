"""Analyze one detected Versuch1 cycle across all available experiment signals."""

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
from src.preprocessing.multi_sensor_cycle_extraction import (  # noqa: E402
    extract_cycle_measurements,
    list_experiment_signals,
)
from src.utils.data_loader import (  # noqa: E402
    build_int_signal_info,
    build_uuid_signal_info,
    find_signals,
)
from src.utils.measurement_loader import load_uuid_signal  # noqa: E402

BASE_DIR = Path("/data/ERA/D63_Nr7_8")
EXPERIMENT = "Versuch1"
START_TIME = "2025-12-23 00:10:00"
END_TIME = "2025-12-23 00:20:00"
MOVEMENT_THRESHOLD = 1.0
FIGURE_PATH = PROJECT_ROOT / "outputs" / "figures" / "first_cycle_all_signals.png"
CYCLE_DATA_DIR = PROJECT_ROOT / "outputs" / "cycle_data"


def _print_signal_summary(signal_name: str, signal_df: pd.DataFrame) -> None:
    """Print a compact summary for one extracted signal window."""

    if signal_df.empty:
        print(f"{signal_name}: no samples")
        return

    print(
        f"{signal_name}: "
        f"start_time={signal_df['time'].min()} "
        f"end_time={signal_df['time'].max()} "
        f"number_of_samples={len(signal_df)} "
        f"minimum_value={signal_df['value'].min()} "
        f"maximum_value={signal_df['value'].max()}"
    )


def main() -> None:
    """Detect one candidate cycle and extract every available signal for it."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    uuid_signal_info = build_uuid_signal_info(BASE_DIR)
    int_signal_info = build_int_signal_info(BASE_DIR)
    position_signals = find_signals(
        uuid_signal_info,
        path_contains=EXPERIMENT,
        unit_code="position",
    )
    if len(position_signals) != 1:
        raise ValueError(
            f"Expected exactly one {EXPERIMENT} position signal, found {len(position_signals)}."
        )

    position_signal_id_uuid = str(position_signals.iloc[0]["signal_id_uuid"])
    position_df = load_uuid_signal(
        BASE_DIR,
        position_signal_id_uuid,
        start_time=START_TIME,
        end_time=END_TIME,
    )
    if position_df.empty:
        raise ValueError("No Position samples were loaded for the requested analysis window.")

    cycles_df = detect_candidate_cycles(position_df, movement_threshold=MOVEMENT_THRESHOLD)
    if cycles_df.empty:
        raise ValueError("No complete candidate cycles were detected in the selected window.")

    first_cycle = cycles_df.iloc[0]
    cycle_start = pd.Timestamp(first_cycle["start_time"])
    cycle_end = pd.Timestamp(first_cycle["end_time"])

    extracted_signals = extract_cycle_measurements(
        base_dir=BASE_DIR,
        cycle_start=cycle_start,
        cycle_end=cycle_end,
        uuid_signal_info=uuid_signal_info,
        int_signal_info=int_signal_info,
        experiment=EXPERIMENT,
    )
    signal_descriptors = list_experiment_signals(
        uuid_signal_info=uuid_signal_info,
        int_signal_info=int_signal_info,
        experiment=EXPERIMENT,
    ).set_index("signal_name")

    CYCLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for signal_name, signal_df in extracted_signals.items():
        signal_df.to_csv(CYCLE_DATA_DIR / f"{signal_name}.csv", index=False)
        _print_signal_summary(signal_name, signal_df)

    ordered_signal_names = list(extracted_signals)
    if not ordered_signal_names:
        raise ValueError("No signals were extracted for the selected cycle.")

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        nrows=len(ordered_signal_names),
        ncols=1,
        figsize=(12, max(3, 2.5 * len(ordered_signal_names))),
        sharex=True,
    )
    if len(ordered_signal_names) == 1:
        axes = [axes]

    for ax, signal_name in zip(axes, ordered_signal_names, strict=True):
        signal_df = extracted_signals[signal_name]
        unit_symbol = signal_descriptors.loc[signal_name, "unit_symbol"]
        ax.set_title(signal_name)
        ax.set_ylabel(f"Value ({unit_symbol})")
        if signal_df.empty:
            ax.text(0.5, 0.5, "No samples", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.plot(signal_df["time"], signal_df["value"], linewidth=1.0)
        ax.grid(True)

    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)

    print(f"cycle_start: {cycle_start}")
    print(f"cycle_end: {cycle_end}")
    print(f"saved figure: {FIGURE_PATH}")
    print(f"saved cycle data directory: {CYCLE_DATA_DIR}")


if __name__ == "__main__":
    main()
