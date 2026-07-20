"""Analyze recording sessions for the Versuch1 position signal."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.session_detection import detect_recording_sessions  # noqa: E402
from src.utils.data_loader import build_uuid_signal_info, find_signals  # noqa: E402

BASE_DIR = Path("/data/ERA/D63_Nr7_8")
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "datasets" / "versuch1_position_sessions.csv"
# This 1 hours gap threshold is exploratory and must be validated using the
# resulting session statistics before it is treated as a final rule.



def main() -> None:
    """Find the Versuch1 position signal and summarize its recording sessions."""

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
    sessions_df = detect_recording_sessions(
        BASE_DIR,
        signal_id_uuid,
        
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sessions_df.to_csv(OUTPUT_PATH, index=False)

    display_df = sessions_df.rename(
        columns={
            "duration_seconds": "duration",
            "number_of_samples": "number of samples",
            "maximum_internal_gap_seconds": "maximum internal gap",
        }
    ).copy()
    if not display_df.empty:
        display_df["duration"] = pd.to_timedelta(display_df["duration"], unit="s")

    print(f"signal_id_uuid: {signal_id_uuid}")
    print(f"number of detected sessions: {len(sessions_df)}")
    print(display_df.to_string(index=False))
    print(f"saved session table: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
