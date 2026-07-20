"""Testing and debugging script for targeted UUID measurement loading.

This file is only for manual testing and debugging. Reusable measurement loading
logic lives in ``src/utils/measurement_loader.py``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import build_uuid_signal_info, find_signals  # noqa: E402
from src.utils.measurement_loader import load_uuid_signal  # noqa: E402

BASE_DIR = Path("/data/ERA/D63_Nr7_8")
START_TIME = "2025-10-23 00:00:00"
END_TIME = "2025-10-23 01:00:00"


def main() -> None:
    """Load and preview one UUID-based position signal for Versuch1."""

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

    print(f"signal_id_uuid: {signal_id_uuid}")
    print(f"start_time: {START_TIME}")
    print(f"end_time: {END_TIME}")
    print(f"shape: {signal_df.shape}")
    print(signal_df.head().to_string(index=False))
    print(f"min time: {signal_df['time'].min()}")
    print(f"max time: {signal_df['time'].max()}")
    print(f"min value: {signal_df['value'].min()}")
    print(f"max value: {signal_df['value'].max()}")


if __name__ == "__main__":
    main()
