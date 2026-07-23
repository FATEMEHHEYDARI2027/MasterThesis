"""Per-session in-memory signal cache to avoid re-scanning Parquet per cycle.

Cycle detection already produces ``cycle_start``/``cycle_end`` timestamps per
cycle. Without this cache, ``extract_cycle_measurements`` re-opens the
Parquet dataset for every signal of every cycle (hundreds of millions of
rows scanned repeatedly). ``SessionSignalCache`` loads every signal for one
recording session exactly once, keeps the resulting DataFrames in memory,
and lets each cycle within that session slice the cached DataFrames instead
of touching the Parquet dataset again.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.measurement_loader import load_int_signal, load_uuid_signal

logger = logging.getLogger(__name__)


def _slice_cached_frame(
    signal_df: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    """Slice one cached, time-sorted signal frame to a half-open interval.

    Mirrors the ``[start_time, end_time)`` filtering semantics of
    :func:`src.utils.measurement_loader._load_signal_frame` so cached slices
    are identical to a fresh Parquet read for the same interval.
    """

    if signal_df.empty:
        return signal_df

    mask = (signal_df["time"] >= start_time) & (signal_df["time"] < end_time)
    return signal_df.loc[mask].reset_index(drop=True)


class SessionSignalCache:
    """Holds every signal of one recording session in memory for reuse.

    All signals referenced by ``signal_descriptors`` are loaded once, for
    the full ``[session_start, session_end)`` window, when the cache is
    constructed. Each cycle in the session then reuses these in-memory
    DataFrames via :meth:`slice_uuid_signal` / :meth:`slice_int_signal`
    instead of re-opening the Parquet dataset.
    """

    def __init__(
        self,
        base_dir: Path,
        session_start: pd.Timestamp,
        session_end: pd.Timestamp,
        signal_descriptors: pd.DataFrame,
    ) -> None:
        self._uuid_frames: dict[str, pd.DataFrame] = {}
        self._int_frames: dict[int, pd.DataFrame] = {}

        for row in signal_descriptors.itertuples(index=False):
            if row.source == "uuid":
                if pd.isna(row.signal_id_uuid):
                    continue
                key = str(row.signal_id_uuid)
                if key in self._uuid_frames:
                    continue
                try:
                    self._uuid_frames[key] = load_uuid_signal(
                        base_dir,
                        key,
                        start_time=session_start,
                        end_time=session_end,
                    )
                except FileNotFoundError:
                    logger.warning(
                        "Skipping UUID signal %s for session cache because its "
                        "measurement dataset partition is missing",
                        key,
                    )
            else:
                if pd.isna(row.signal_id):
                    continue
                signal_id = int(row.signal_id)
                if signal_id in self._int_frames:
                    continue
                try:
                    self._int_frames[signal_id] = load_int_signal(
                        base_dir,
                        signal_id,
                        start_time=session_start,
                        end_time=session_end,
                    )
                except FileNotFoundError:
                    logger.warning(
                        "Skipping INT signal %d for session cache because its "
                        "measurement dataset partition is missing",
                        signal_id,
                    )

        logger.info(
            "Built session signal cache for [%s, %s): %d UUID signal(s), %d INT signal(s)",
            session_start,
            session_end,
            len(self._uuid_frames),
            len(self._int_frames),
        )

    def slice_uuid_signal(
        self,
        signal_id_uuid: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ) -> pd.DataFrame:
        """Return the cached UUID signal rows within ``[start_time, end_time)``."""

        signal_df = self._uuid_frames.get(str(signal_id_uuid))
        if signal_df is None:
            raise FileNotFoundError(f"UUID signal {signal_id_uuid} is not present in the session cache")
        return _slice_cached_frame(signal_df, start_time, end_time)

    def slice_int_signal(
        self,
        signal_id: int,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ) -> pd.DataFrame:
        """Return the cached INT signal rows within ``[start_time, end_time)``."""

        signal_df = self._int_frames.get(int(signal_id))
        if signal_df is None:
            raise FileNotFoundError(f"INT signal {signal_id} is not present in the session cache")
        return _slice_cached_frame(signal_df, start_time, end_time)

    def release(self) -> None:
        """Drop all cached DataFrames so their memory can be reclaimed."""

        self._uuid_frames.clear()
        self._int_frames.clear()


class SessionSignalCacheManager:
    """Keeps at most one :class:`SessionSignalCache` alive at a time.

    Sessions are cached lazily on first use and released as soon as every
    cycle belonging to that session has been processed, so memory usage
    stays bounded to one recording session at a time regardless of how many
    sessions or cycles the overall run covers.
    """

    def __init__(
        self,
        base_dir: Path,
        signal_descriptors: pd.DataFrame,
        session_bounds: dict[int, tuple[pd.Timestamp, pd.Timestamp]],
        session_cycle_counts: dict[int, int],
    ) -> None:
        self._base_dir = base_dir
        self._signal_descriptors = signal_descriptors
        self._session_bounds = session_bounds
        self._remaining_cycles: dict[int, int] = dict(session_cycle_counts)
        self._active_caches: dict[int, SessionSignalCache] = {}

    def get(self, session_id: int) -> SessionSignalCache | None:
        """Return the (lazily built) cache for ``session_id``, if bounds are known."""

        if session_id in self._active_caches:
            return self._active_caches[session_id]

        bounds = self._session_bounds.get(session_id)
        if bounds is None:
            return None

        session_start, session_end = bounds
        cache = SessionSignalCache(
            base_dir=self._base_dir,
            session_start=session_start,
            session_end=session_end,
            signal_descriptors=self._signal_descriptors,
        )
        self._active_caches[session_id] = cache
        return cache

    def mark_cycle_done(self, session_id: int) -> None:
        """Record that one cycle of ``session_id`` finished; release the cache when done."""

        if session_id not in self._remaining_cycles:
            return
        self._remaining_cycles[session_id] -= 1
        if self._remaining_cycles[session_id] <= 0:
            cache = self._active_caches.pop(session_id, None)
            if cache is not None:
                cache.release()
