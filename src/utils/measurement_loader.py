"""Reusable loaders for reading targeted slices of large measurement datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds

logger = logging.getLogger(__name__)

SIGNAL_DATASET_NAME = "signal_data_point.parquet"
VIBRATION_DATASET_NAME = "vibration.parquet"
SIGNAL_COLUMNS = ["time", "value", "signal_id"]


def _normalize_timestamp(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    """Convert an optional timestamp-like value to ``pandas.Timestamp``."""

    if value is None:
        return None
    return pd.Timestamp(value)


def _build_interval_label(
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
) -> str:
    """Return a readable half-open interval label."""

    return f"[{start_ts if start_ts is not None else '-inf'}, {end_ts if end_ts is not None else '+inf'})"


def _load_signal_frame(
    dataset_path: Path,
    signal_label: str,
    start_time: str | pd.Timestamp | None,
    end_time: str | pd.Timestamp | None,
    allow_full_scan: bool,
    partitioning: str | None = "hive",
    signal_filter_value: str | int | None = None,
    coerce_signal_id_numeric: bool = False,
) -> pd.DataFrame:
    """Load one bounded signal slice from a parquet dataset."""

    if not dataset_path.exists():
        raise FileNotFoundError(f"Signal dataset not found: {dataset_path}")

    start_ts = _normalize_timestamp(start_time)
    end_ts = _normalize_timestamp(end_time)

    if start_ts is None and end_ts is None and not allow_full_scan:
        raise ValueError(
            "Loading the complete signal is disabled by default. Provide "
            "start_time or end_time, or set allow_full_scan=True."
        )

    if start_ts is not None and end_ts is not None and start_ts >= end_ts:
        raise ValueError("start_time must be earlier than end_time.")

    interval_label = _build_interval_label(start_ts, end_ts)
    logger.info("Loading %s from %s for interval %s", signal_label, dataset_path, interval_label)

    dataset_kwargs: dict[str, Any] = {"format": "parquet"}
    if partitioning is not None:
        dataset_kwargs["partitioning"] = partitioning

    dataset = ds.dataset(dataset_path, **dataset_kwargs)
    filter_expression = None
    if signal_filter_value is not None:
        filter_expression = ds.field("signal_id") == signal_filter_value
    if start_ts is not None:
        time_filter = ds.field("time") >= start_ts.to_pydatetime()
        filter_expression = time_filter if filter_expression is None else filter_expression & time_filter
    if end_ts is not None:
        time_filter = ds.field("time") < end_ts.to_pydatetime()
        filter_expression = time_filter if filter_expression is None else filter_expression & time_filter

    scanner = dataset.scanner(columns=SIGNAL_COLUMNS, filter=filter_expression)
    table = scanner.to_table()

    signal_df = table.to_pandas().sort_values("time").reset_index(drop=True)
    if coerce_signal_id_numeric and not signal_df.empty:
        signal_df["signal_id"] = pd.to_numeric(signal_df["signal_id"], errors="coerce").astype("Int64")

    logger.info("Loaded %d rows for %s in interval %s", len(signal_df), signal_label, interval_label)
    return signal_df


def load_uuid_signal(
    base_dir: Path,
    signal_id_uuid: str,
    start_time: str | pd.Timestamp | None = None,
    end_time: str | pd.Timestamp | None = None,
    allow_full_scan: bool = False,
) -> pd.DataFrame:
    """Load one UUID-based signal from the partitioned measurement dataset.

    Parameters
    ----------
    base_dir:
        Base ERA dataset directory containing ``signal_data_point.parquet``.
    signal_id_uuid:
        UUID string identifying the signal to load.
    start_time:
        Inclusive lower time bound for the selected rows.
    end_time:
        Exclusive upper time bound for the selected rows.
    allow_full_scan:
        Whether to allow loading the full lifetime of the signal when no time
        bounds are provided.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing only ``time``, ``value``, and ``signal_id`` for the
        requested signal, sorted by ``time`` with a reset index.
    """

    dataset_root = Path(base_dir).expanduser().resolve() / SIGNAL_DATASET_NAME
    return _load_signal_frame(
        dataset_path=dataset_root,
        signal_label=f"UUID signal {signal_id_uuid}",
        start_time=start_time,
        end_time=end_time,
        allow_full_scan=allow_full_scan,
        partitioning="hive",
        signal_filter_value=signal_id_uuid,
    )


def load_int_signal(
    base_dir: Path,
    signal_id: int,
    start_time: str | pd.Timestamp | None = None,
    end_time: str | pd.Timestamp | None = None,
    allow_full_scan: bool = False,
) -> pd.DataFrame:
    """Load one integer-ID vibration signal from the partitioned measurement dataset.

    Parameters
    ----------
    base_dir:
        Base ERA dataset directory containing ``vibration.parquet``.
    signal_id:
        Integer signal identifier for the vibration dataset.
    start_time:
        Inclusive lower time bound for the selected rows.
    end_time:
        Exclusive upper time bound for the selected rows.
    allow_full_scan:
        Whether to allow loading the full lifetime of the signal when no time
        bounds are provided.
    """

    dataset_root = Path(base_dir).expanduser().resolve() / VIBRATION_DATASET_NAME / f"signal_id={signal_id}"
    return _load_signal_frame(
        dataset_path=dataset_root,
        signal_label=f"INT signal {signal_id}",
        start_time=start_time,
        end_time=end_time,
        allow_full_scan=allow_full_scan,
        partitioning=None,
        signal_filter_value=None,
        coerce_signal_id_numeric=True,
    )
