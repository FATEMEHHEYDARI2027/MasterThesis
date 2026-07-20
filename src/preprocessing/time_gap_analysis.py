"""Analyze time-gap distributions for large UUID-based measurement signals."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.preprocessing.session_detection import detect_recording_sessions
from src.utils.measurement_loader import load_uuid_signal

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNK_DURATION = pd.Timedelta(days=1)
SESSION_GAP_THRESHOLD_SECONDS = 5.0
THRESHOLDS_SECONDS: tuple[tuple[str, float], ...] = (
    ("gaps_gt_1_second", 1.0),
    ("gaps_gt_5_seconds", 5.0),
    ("gaps_gt_10_seconds", 10.0),
    ("gaps_gt_30_seconds", 30.0),
    ("gaps_gt_1_minute", 60.0),
    ("gaps_gt_5_minutes", 5.0 * 60.0),
    ("gaps_gt_10_minutes", 10.0 * 60.0),
    ("gaps_gt_30_minutes", 30.0 * 60.0),
    ("gaps_gt_1_hour", 60.0 * 60.0),
    ("gaps_gt_6_hours", 6.0 * 60.0 * 60.0),
    ("gaps_gt_24_hours", 24.0 * 60.0 * 60.0),
)


def _iter_time_windows(
    start_time: pd.Timestamp,
    end_time_exclusive: pd.Timestamp,
    chunk_duration: pd.Timedelta,
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    """Yield half-open time windows covering the requested interval."""

    current_start = start_time
    while current_start < end_time_exclusive:
        current_end = min(current_start + chunk_duration, end_time_exclusive)
        yield current_start, current_end
        current_start = current_end


def _update_gap_counts(
    gap_counts_us: dict[int, int],
    gap_us: np.ndarray,
) -> None:
    """Accumulate exact gap counts in integer microseconds."""

    unique_values, unique_counts = np.unique(gap_us, return_counts=True)
    for gap_value_us, count in zip(unique_values, unique_counts, strict=True):
        gap_counts_us[int(gap_value_us)] = gap_counts_us.get(int(gap_value_us), 0) + int(count)


def _rank_value_seconds(
    values_us: np.ndarray,
    cumulative_counts: np.ndarray,
    rank: int,
) -> float:
    """Return the value at a zero-based rank from compressed counts."""

    index = int(np.searchsorted(cumulative_counts, rank, side="right"))
    return float(values_us[index]) / 1_000_000.0


def _quantile_from_counts_seconds(
    gap_counts_us: dict[int, int],
    quantile: float,
) -> float:
    """Compute an exact linear-interpolated quantile from compressed gap counts."""

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1.")

    values_us = np.array(sorted(gap_counts_us), dtype=np.int64)
    counts = np.array([gap_counts_us[int(value)] for value in values_us], dtype=np.int64)
    cumulative_counts = np.cumsum(counts)
    total_count = int(cumulative_counts[-1])
    if total_count == 1:
        return float(values_us[0]) / 1_000_000.0

    position = (total_count - 1) * quantile
    lower_rank = int(np.floor(position))
    upper_rank = int(np.ceil(position))
    lower_value = _rank_value_seconds(values_us, cumulative_counts, lower_rank)
    upper_value = _rank_value_seconds(values_us, cumulative_counts, upper_rank)
    return lower_value + (upper_value - lower_value) * (position - lower_rank)


def analyze_time_gaps(
    base_dir: Path,
    signal_id_uuid: str,
    chunk_duration: pd.Timedelta = DEFAULT_CHUNK_DURATION,
) -> tuple[pd.DataFrame, dict[int, int]]:
    """Analyze consecutive timestamp gaps for one UUID-based signal.

    The full time range is processed incrementally through repeated calls to
    ``load_uuid_signal`` so that the complete signal is never held in memory at
    once.
    """

    if chunk_duration <= pd.Timedelta(0):
        raise ValueError("chunk_duration must be positive.")

    sessions_df = detect_recording_sessions(
        base_dir,
        signal_id_uuid,
        gap_threshold_seconds=SESSION_GAP_THRESHOLD_SECONDS,
    )
    if sessions_df.empty:
        raise ValueError(f"No samples found for UUID signal {signal_id_uuid}.")

    analysis_start = pd.Timestamp(sessions_df["start_time"].min())
    analysis_end_exclusive = pd.Timestamp(sessions_df["end_time"].max()) + pd.Timedelta(
        microseconds=1
    )

    gap_counts_us: dict[int, int] = {}
    previous_time: pd.Timestamp | None = None
    number_of_samples = 0
    total_gap_us = 0
    number_of_gaps = 0

    logger.info(
        "Analyzing time gaps for UUID signal %s from %s to %s in %s chunks",
        signal_id_uuid,
        analysis_start,
        analysis_end_exclusive,
        chunk_duration,
    )

    for window_start, window_end in _iter_time_windows(
        analysis_start,
        analysis_end_exclusive,
        chunk_duration,
    ):
        signal_df = load_uuid_signal(
            base_dir,
            signal_id_uuid,
            start_time=window_start,
            end_time=window_end,
        )
        if signal_df.empty:
            continue

        number_of_samples += len(signal_df)
        gaps = signal_df["time"].diff()
        if previous_time is not None:
            gaps.iat[0] = signal_df["time"].iat[0] - previous_time

        valid_gaps = gaps.dropna()
        if not valid_gaps.empty:
            gap_us = np.rint(valid_gaps.dt.total_seconds().to_numpy() * 1_000_000.0).astype(
                np.int64
            )
            _update_gap_counts(gap_counts_us, gap_us)
            total_gap_us += int(gap_us.sum())
            number_of_gaps += len(gap_us)

        previous_time = pd.Timestamp(signal_df["time"].iat[-1])

    if number_of_samples < 2 or not gap_counts_us:
        raise ValueError("At least two samples are required to analyze time gaps.")

    min_gap_seconds = float(min(gap_counts_us)) / 1_000_000.0
    max_gap_seconds = float(max(gap_counts_us)) / 1_000_000.0
    mean_gap_seconds = total_gap_us / number_of_gaps / 1_000_000.0

    statistics_rows: list[dict[str, object]] = [
        {"metric": "signal_id_uuid", "value": signal_id_uuid},
        {"metric": "number_of_samples", "value": number_of_samples},
        {"metric": "minimum_gap_seconds", "value": min_gap_seconds},
        {"metric": "median_gap_seconds", "value": _quantile_from_counts_seconds(gap_counts_us, 0.5)},
        {"metric": "mean_gap_seconds", "value": mean_gap_seconds},
        {
            "metric": "percentile_95_gap_seconds",
            "value": _quantile_from_counts_seconds(gap_counts_us, 0.95),
        },
        {
            "metric": "percentile_99_gap_seconds",
            "value": _quantile_from_counts_seconds(gap_counts_us, 0.99),
        },
        {"metric": "maximum_gap_seconds", "value": max_gap_seconds},
    ]

    values_us = np.array(sorted(gap_counts_us), dtype=np.int64)
    counts = np.array([gap_counts_us[int(value)] for value in values_us], dtype=np.int64)
    values_seconds = values_us / 1_000_000.0
    for metric_name, threshold_seconds in THRESHOLDS_SECONDS:
        threshold_count = int(counts[values_seconds > threshold_seconds].sum())
        statistics_rows.append({"metric": metric_name, "value": threshold_count})

    statistics_df = pd.DataFrame(statistics_rows)
    return statistics_df, gap_counts_us


def save_time_gap_statistics(statistics_df: pd.DataFrame, output_path: Path) -> None:
    """Save time-gap statistics to a CSV file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    statistics_df.to_csv(output_path, index=False)
    logger.info("Saved time-gap statistics to %s", output_path)


def plot_time_gap_histogram(
    gap_counts_us: dict[int, int],
    output_path: Path,
    max_gap_seconds: float | None = None,
) -> None:
    """Plot a weighted histogram from compressed gap counts."""

    values_us = np.array(sorted(gap_counts_us), dtype=np.int64)
    counts = np.array([gap_counts_us[int(value)] for value in values_us], dtype=np.int64)
    values_seconds = values_us / 1_000_000.0

    if max_gap_seconds is not None:
        mask = values_seconds < max_gap_seconds
        values_seconds = values_seconds[mask]
        counts = counts[mask]
        if len(values_seconds) == 0:
            raise ValueError("No gaps are available in the requested histogram range.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    if max_gap_seconds is None:
        if np.isclose(values_seconds.min(), values_seconds.max()):
            bins = np.array([values_seconds.min() * 0.9, values_seconds.max() * 1.1])
        else:
            bins = np.geomspace(values_seconds.min(), values_seconds.max(), num=80)
        ax.hist(values_seconds, bins=bins, weights=counts)
        ax.set_xscale("log")
        ax.set_title("Time Gap Histogram")
        ax.set_xlabel("Gap duration (seconds, log scale)")
    else:
        bins = np.linspace(0.0, max_gap_seconds, num=80)
        ax.hist(values_seconds, bins=bins, weights=counts)
        ax.set_title("Time Gap Histogram Under 1 Second")
        ax.set_xlabel("Gap duration (seconds)")

    ax.set_ylabel("Count")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved time-gap histogram to %s", output_path)


def print_time_gap_statistics(statistics_df: pd.DataFrame) -> None:
    """Print time-gap statistics to stdout."""

    for row in statistics_df.itertuples(index=False):
        print(f"{row.metric}: {row.value}")


if __name__ == "__main__":
    from src.utils.data_loader import build_uuid_signal_info, find_signals

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    base_dir = Path("/data/ERA/D63_Nr7_8")
    uuid_info = build_uuid_signal_info(base_dir)
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
    statistics_df, gap_counts_us = analyze_time_gaps(base_dir, signal_id_uuid)

    save_time_gap_statistics(
        statistics_df,
        PROJECT_ROOT / "outputs" / "statistics" / "time_gap_statistics.csv",
    )
    plot_time_gap_histogram(
        gap_counts_us,
        PROJECT_ROOT / "outputs" / "figures" / "time_gap_histogram.png",
    )
    plot_time_gap_histogram(
        gap_counts_us,
        PROJECT_ROOT / "outputs" / "figures" / "time_gap_histogram_under1s.png",
        max_gap_seconds=1.0,
    )
    print_time_gap_statistics(statistics_df)
