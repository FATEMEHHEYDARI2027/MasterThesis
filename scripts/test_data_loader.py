"""Testing and debugging script for the reusable data loader module.

This file is only for manual testing and debugging. Reusable metadata loading
logic lives in ``src/utils/data_loader.py``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_loader import (  # noqa: E402
    build_int_signal_info,
    build_uuid_signal_info,
    find_signals,
)

BASE_DIR = Path("/data/ERA/D63_Nr7_8")


def main() -> None:
    """Run a small manual preview of generated signal metadata."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    uuid_info = build_uuid_signal_info(BASE_DIR)
    int_info = build_int_signal_info(BASE_DIR)
    position_signals = find_signals(uuid_info, unit_code="position")

    print("\nUUID signal info (head)")
    print(uuid_info.head().to_string(index=False))

    print("\nINT signal info (head)")
    print(int_info.head().to_string(index=False))

    print("\nPosition signals")
    print(position_signals.to_string(index=False))


if __name__ == "__main__":
    main()
