"""Exploratory plotting script for the Versuch1 position signal."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import build_uuid_signal_info, find_signals  # noqa: E402
from src.utils.measurement_loader import load_uuid_signal  # noqa: E402

BASE_DIR = Path("/data/ERA/D63_Nr7_8")
START_TIME = "2025-10-23 00:00:00"
END_TIME = "2026-03-27 00:00:00"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "figures" / "position_versuch1_first_10_minutes.png"

logger = logging.getLogger(__name__)


def _compute_sampling_intervals_ms(signal_df: pd.DataFrame) -> pd.Series:
    """Return consecutive sampling intervals in milliseconds."""

    return signal_df["time"].diff().dropna().dt.total_seconds() * 1000.0


def main() -> None:
    """Load, summarize, and plot the Versuch1 position signal."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    uuid_info = build_uuid_signal_info(BASE_DIR)
    position_signals = find_signals(
        uuid_info,
        path_contains="Versuch1",
        unit_code="position",
    )
    if position_signals.empty:
        raise ValueError("No UUID-based position signal found for Versuch1.")

    signal_id_uuid = str(position_signals.iloc[0]["signal_id_uuid"])
    signal_df = load_uuid_signal(
        BASE_DIR,
        signal_id_uuid,
        start_time=START_TIME,
        end_time=END_TIME,
    )
    if signal_df.empty:
        raise ValueError("No position samples were loaded for the requested interval.")

    sampling_intervals_ms = _compute_sampling_intervals_ms(signal_df)
    if sampling_intervals_ms.empty:
        raise ValueError("At least two samples are required to compute sampling intervals.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(signal_df["time"], signal_df["value"], linewidth=1.0)
    ax.set_title("Position Signal – Versuch1")
    ax.set_xlabel("Time")
    ax.set_ylabel("Position Value")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)

    logger.info("Saved position plot to %s", OUTPUT_PATH)

    print(f"signal_id_uuid: {signal_id_uuid}")
    print(f"start_time: {START_TIME}")
    print(f"end_time: {END_TIME}")
    print(f"output_path: {OUTPUT_PATH}")
    print(f"number of samples: {len(signal_df)}")
    print(f"median sampling interval (ms): {sampling_intervals_ms.median()}")
    print(f"minimum sampling interval (ms): {sampling_intervals_ms.min()}")
    print(f"maximum sampling interval (ms): {sampling_intervals_ms.max()}")
    print(f"minimum position: {signal_df['value'].min()}")
    print(f"maximum position: {signal_df['value'].max()}")


if __name__ == "__main__":
    main()
