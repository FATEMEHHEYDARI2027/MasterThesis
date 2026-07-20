"""Interactive multi-sensor cycle visualization built with Plotly."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

MAX_POINTS_PER_SIGNAL = 20_000


def _evenly_spaced_indices(length: int, max_count: int) -> np.ndarray:
    """Return evenly spaced row indices, always keeping the first and last row."""

    if length <= max_count:
        return np.arange(length)
    return np.unique(np.linspace(0, length - 1, num=max_count, dtype=np.int64))


def _lookup_unit_symbol(signal_name: str, signal_descriptors: pd.DataFrame) -> str:
    """Return the physical unit symbol for one signal, defaulting to ``unknown``."""

    if signal_descriptors is None or signal_descriptors.empty:
        return "unknown"
    if "signal_name" not in signal_descriptors.columns or "unit_symbol" not in signal_descriptors.columns:
        return "unknown"

    matches = signal_descriptors.loc[
        signal_descriptors["signal_name"] == signal_name, "unit_symbol"
    ]
    if matches.empty:
        return "unknown"

    unit_symbol = matches.iloc[0]
    if pd.isna(unit_symbol) or str(unit_symbol).strip() == "":
        return "unknown"
    return str(unit_symbol)


def _build_title(experiment: str, cycle_id: int, session_id: int | None) -> str:
    """Build one readable title identifying the experiment, cycle, and session."""

    title_parts = [experiment, f"Multi-sensor cycle {cycle_id}"]
    if session_id is not None:
        title_parts.append(f"Session {session_id}")
    return " \u2013 ".join(title_parts)


def plot_multi_sensor_cycle(
    extracted_signals: dict[str, pd.DataFrame],
    signal_descriptors: pd.DataFrame,
    output_path: Path,
    experiment: str,
    cycle_id: int,
    session_id: int | None = None,
) -> Path | None:
    """Save one interactive Plotly visualization for all sensors of one cycle.

    Every extracted signal receives its own vertically stacked subplot sharing
    the same time axis, so high-frequency signals (e.g. vibration_x/y/z) are
    never merged onto the same y-axis as signals using a different unit.
    """

    if not extracted_signals:
        return None

    signal_names = list(extracted_signals.keys())
    num_signals = len(signal_names)

    units = {name: _lookup_unit_symbol(name, signal_descriptors) for name in signal_names}
    subplot_titles = [
        f"{name} ({units[name]})" if units[name] not in ("", "unknown") else name
        for name in signal_names
    ]

    fig = make_subplots(
        rows=num_signals,
        cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=min(0.3 / max(num_signals, 1), 0.08),
    )

    for row_index, signal_name in enumerate(signal_names, start=1):
        signal_df = extracted_signals[signal_name]
        unit = units[signal_name]

        if signal_df is None or signal_df.empty:
            fig.add_annotation(
                text="No samples available",
                showarrow=False,
                row=row_index,
                col=1,
            )
            logger.info(
                "multi-sensor cycle plot: signal %s has no samples for cycle %d",
                signal_name,
                cycle_id,
            )
            continue

        total_rows = len(signal_df)
        if total_rows > MAX_POINTS_PER_SIGNAL:
            plot_indices = _evenly_spaced_indices(total_rows, MAX_POINTS_PER_SIGNAL)
            plot_df = signal_df.iloc[plot_indices]
        else:
            plot_df = signal_df
        logger.info(
            "multi-sensor cycle plot: plotting %d of %d rows for signal %s (cycle %d)",
            len(plot_df),
            total_rows,
            signal_name,
            cycle_id,
        )

        custom_data = np.column_stack(
            [
                np.full(len(plot_df), signal_name),
                plot_df["time"].astype(str).to_numpy(),
                plot_df["value"].to_numpy(),
                np.full(len(plot_df), unit),
            ]
        )
        fig.add_trace(
            go.Scattergl(
                x=plot_df["time"],
                y=plot_df["value"],
                mode="lines+markers",
                marker=dict(size=3),
                line=dict(width=1),
                name=signal_name,
                customdata=custom_data,
                hovertemplate=(
                    "signal: %{customdata[0]}<br>"
                    "timestamp: %{customdata[1]}<br>"
                    "value: %{customdata[2]}<br>"
                    "unit: %{customdata[3]}<extra></extra>"
                ),
            ),
            row=row_index,
            col=1,
        )
        fig.update_yaxes(title_text=unit, row=row_index, col=1)

    fig.update_xaxes(title_text="Time", row=num_signals, col=1)
    fig.update_layout(
        title=_build_title(experiment, cycle_id, session_id),
        hovermode="x unified",
        dragmode="zoom",
        showlegend=False,
        height=max(250 * num_signals, 400),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
    )
    return output_path
