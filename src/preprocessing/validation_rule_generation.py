"""Robust, data-driven validation rule generation.

This module implements the ``validation_rule_generation`` pipeline stage.
It runs **after** extraction and quality profiling and **before** dataset
validation, and it derives statistical validation thresholds from the
observed, non-rejected profiling population instead of from manually fixed
per-signal sample-count thresholds.

Methodology
-----------
Two kinds of rules are produced:

``hard rules``
    Fixed, logical rules that are never learned from data (for example:
    ``start_time`` must be earlier than ``end_time``, a required signal must
    have at least one finite value, Position must exist for a detected
    cycle).

``learned rules``
    Robust statistical bounds derived per ``(signal, metric)`` pair from the
    profiling population, using either:

    * ``median_mad`` -- robust z-score bounds using the median and the
      median absolute deviation (MAD):
      ``robust_z = 0.6745 * (x - median) / MAD``. Solving for ``x`` at the
      configured ``mad_z_limit`` gives the lower/upper bound.
    * ``quantile`` -- bounds derived directly from configurable quantiles,
      used whenever MAD is zero (a degenerate/near-constant metric) or as an
      explicit ``fallback_method``.

Because the vibration signals (``vibration_x``/``y``/``z``) are
intentionally duty-cycled, a global sample-count threshold computed over
*all* cycles would mix "no vibration recorded" cycles with "full vibration
burst" cycles and produce a meaningless bound. Every cycle is therefore
first classified into a vibration-availability class
(``vibration_unavailable`` / ``vibration_partial`` / ``vibration_complete``)
and vibration thresholds are derived **only** from the ``vibration_complete``
reference subset.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_SIGNAL_ROLES: dict[str, tuple[str, ...]] = {
    "cycle_reference": ("position",),
    "core_required": ("position", "velocity", "current", "pressure", "temperature"),
    "optional_duty_cycled": ("vibration_x", "vibration_y", "vibration_z"),
}

DEFAULT_VALIDATION_RULE_GENERATION_CONFIG: dict[str, object] = {
    "enabled": True,
    "reference_population": "all_profiled_cycles",
    "default_method": "median_mad",
    "mad_z_limit": 3.5,
    "lower_quantile": 0.01,
    "upper_quantile": 0.99,
    "minimum_reference_count": 20,
    "fallback_method": "quantile",
    "freeze_rules": True,
    "mark_small_or_limited_population_as_provisional": True,
}

# Metrics eligible for automatic, data-driven threshold generation for every
# per-signal quality row produced by ``cycle_quality_profiler``.
SIGNAL_LEVEL_LEARNABLE_METRICS: tuple[str, ...] = (
    "sample_count",
    "finite_sample_count",
    "estimated_sampling_rate",
    "median_sampling_interval",
    "maximum_timestamp_gap",
    "coverage_ratio",
    "signal_range",
    "standard_deviation",
)

# Cycle-level metrics (one value per cycle, not per signal).
CYCLE_LEVEL_LEARNABLE_METRICS: tuple[str, ...] = (
    "cycle_duration_seconds",
    "position_stroke_range",
)

VIBRATION_UNAVAILABLE = "vibration_unavailable"
VIBRATION_PARTIAL = "vibration_partial"
VIBRATION_COMPLETE = "vibration_complete"

RULE_TYPE_HARD_INTERVAL_ORDER = "hard_interval_order"
RULE_TYPE_HARD_POSITIVE_DURATION = "hard_positive_duration"
RULE_TYPE_HARD_SIGNAL_PRESENCE = "hard_signal_presence"
RULE_TYPE_HARD_FINITE_REQUIRED = "hard_finite_required"
RULE_TYPE_MEDIAN_MAD = "median_mad"
RULE_TYPE_QUANTILE = "quantile"


@dataclass(slots=True)
class RuleGenerationConfig:
    """Validated configuration for the ``validation_rule_generation`` stage."""

    enabled: bool = True
    reference_population: str = "all_profiled_cycles"
    default_method: str = "median_mad"
    mad_z_limit: float = 3.5
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99
    minimum_reference_count: int = 20
    fallback_method: str = "quantile"
    freeze_rules: bool = True
    mark_small_or_limited_population_as_provisional: bool = True

    @classmethod
    def from_mapping(cls, mapping: dict[str, object] | None) -> "RuleGenerationConfig":
        merged = dict(DEFAULT_VALIDATION_RULE_GENERATION_CONFIG)
        if mapping:
            merged.update(mapping)
        return cls(
            enabled=bool(merged["enabled"]),
            reference_population=str(merged["reference_population"]),
            default_method=str(merged["default_method"]),
            mad_z_limit=float(merged["mad_z_limit"]),
            lower_quantile=float(merged["lower_quantile"]),
            upper_quantile=float(merged["upper_quantile"]),
            minimum_reference_count=int(merged["minimum_reference_count"]),
            fallback_method=str(merged["fallback_method"]),
            freeze_rules=bool(merged["freeze_rules"]),
            mark_small_or_limited_population_as_provisional=bool(
                merged["mark_small_or_limited_population_as_provisional"]
            ),
        )


def normalize_signal_roles(
    signal_roles: dict[str, object] | None,
) -> dict[str, tuple[str, ...]]:
    """Merge user-configured signal roles over the documented defaults."""

    merged: dict[str, tuple[str, ...]] = {
        key: tuple(value) for key, value in DEFAULT_SIGNAL_ROLES.items()
    }
    if signal_roles:
        for key, value in signal_roles.items():
            merged[str(key)] = tuple(str(item) for item in value)
    return merged


@dataclass(slots=True)
class RuleGenerationResult:
    """Outcome of one validation-rule-generation run."""

    validation_thresholds: pd.DataFrame = field(default_factory=pd.DataFrame)
    threshold_derivation_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    skipped_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    rule_generation_summary: dict[str, object] = field(default_factory=dict)
    vibration_classification: pd.DataFrame = field(default_factory=pd.DataFrame)


def sanitize_json_value(value: Any) -> Any:
    """Recursively replace NaN/Infinity with ``None`` so JSON stays standards-compliant."""

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (np.floating,)):
        return sanitize_json_value(float(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): sanitize_json_value(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json_value(item) for item in value]
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(payload: dict[str, object], output_path: Path) -> Path:
    """Write ``payload`` as a NaN/Infinity-free JSON document."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(sanitize_json_value(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return output_path


def _axis_completeness_thresholds(
    signal_quality_df: pd.DataFrame, vibration_signals: tuple[str, ...]
) -> dict[str, float]:
    """Derive a per-axis "complete burst" sample-count threshold from the data.

    A vibration axis is only considered part of a *complete* burst overlap
    when its sample count reaches a meaningful fraction of the typical,
    actually-observed burst size for that axis. Using half of the median
    non-zero sample count keeps the definition purely data-driven instead of
    hard-coding an assumed burst duration or sampling rate.
    """

    thresholds: dict[str, float] = {}
    for signal_name in vibration_signals:
        axis_df = signal_quality_df[signal_quality_df["signal_name"] == signal_name]
        nonzero_counts = axis_df.loc[axis_df["sample_count"] > 0, "sample_count"]
        if nonzero_counts.empty:
            thresholds[signal_name] = 1.0
        else:
            thresholds[signal_name] = max(1.0, float(nonzero_counts.median()) * 0.5)
    return thresholds


def classify_vibration_availability(
    signal_quality_df: pd.DataFrame,
    cycle_ids: pd.Series,
    vibration_signals: tuple[str, ...] = DEFAULT_SIGNAL_ROLES["optional_duty_cycled"],
) -> pd.DataFrame:
    """Classify every cycle into a vibration-availability class.

    Returns a DataFrame with columns ``cycle_id`` and ``vibration_class``.
    """

    if signal_quality_df.empty or not vibration_signals:
        return pd.DataFrame(
            {
                "cycle_id": cycle_ids,
                "vibration_class": VIBRATION_UNAVAILABLE,
            }
        )

    vibration_df = signal_quality_df[signal_quality_df["signal_name"].isin(vibration_signals)]
    completeness_thresholds = _axis_completeness_thresholds(vibration_df, vibration_signals)

    rows: list[dict[str, object]] = []
    grouped = vibration_df.groupby("cycle_id")
    for cycle_id in cycle_ids:
        try:
            cycle_axes = grouped.get_group(int(cycle_id))
        except KeyError:
            cycle_axes = pd.DataFrame(columns=vibration_df.columns)

        axis_present = {
            axis: cycle_axes[cycle_axes["signal_name"] == axis] for axis in vibration_signals
        }
        total_samples = sum(
            int(frame["sample_count"].iloc[0]) if not frame.empty else 0
            for frame in axis_present.values()
        )
        if total_samples == 0:
            vibration_class = VIBRATION_UNAVAILABLE
        else:
            all_complete = True
            for axis, frame in axis_present.items():
                if frame.empty:
                    all_complete = False
                    break
                sample_count = float(frame["sample_count"].iloc[0])
                finite_count = float(frame["finite_sample_count"].iloc[0])
                if sample_count <= 0 or finite_count <= 0:
                    all_complete = False
                    break
                if sample_count < completeness_thresholds.get(axis, 1.0):
                    all_complete = False
                    break
            vibration_class = VIBRATION_COMPLETE if all_complete else VIBRATION_PARTIAL

        rows.append({"cycle_id": int(cycle_id), "vibration_class": vibration_class})

    return pd.DataFrame(rows)


def _median_mad(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, mad


def _derive_metric_rule(
    values: pd.Series,
    config: RuleGenerationConfig,
) -> dict[str, object]:
    """Derive one robust threshold for a single metric's observed values."""

    numeric_values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite_values = numeric_values[np.isfinite(numeric_values)]
    reference_count = int(finite_values.size)

    base: dict[str, object] = {
        "reference_count": reference_count,
        "median": None,
        "mad": None,
        "lower_quantile_value": None,
        "upper_quantile_value": None,
        "lower_bound": None,
        "upper_bound": None,
        "fallback_used": False,
    }

    if reference_count == 0:
        base.update({"status": "skipped_insufficient_data", "method": None})
        return base
    if reference_count < config.minimum_reference_count:
        base.update({"status": "skipped_insufficient_data", "method": None})
        return base

    median, mad = _median_mad(finite_values)
    base["median"] = median
    base["mad"] = mad

    if mad > 0:
        half_width = (config.mad_z_limit / 0.6745) * mad
        base.update(
            {
                "status": "learned",
                "method": RULE_TYPE_MEDIAN_MAD,
                "lower_bound": median - half_width,
                "upper_bound": median + half_width,
            }
        )
        return base

    lower_q = float(np.quantile(finite_values, config.lower_quantile))
    upper_q = float(np.quantile(finite_values, config.upper_quantile))
    base["lower_quantile_value"] = lower_q
    base["upper_quantile_value"] = upper_q
    if lower_q == upper_q:
        base.update({"status": "skipped_constant", "method": None})
        return base

    base.update(
        {
            "status": "learned",
            "method": config.fallback_method or RULE_TYPE_QUANTILE,
            "lower_bound": lower_q,
            "upper_bound": upper_q,
            "fallback_used": True,
        }
    )
    return base


def _hard_rules(core_required_signals: tuple[str, ...]) -> list[dict[str, object]]:
    """Fixed, logical rules that are never learned from the profiling data."""

    hard_rules: list[dict[str, object]] = [
        {
            "signal": "cycle",
            "metric": "interval_order",
            "rule_type": RULE_TYPE_HARD_INTERVAL_ORDER,
            "method": "hard_rule",
            "reference_population": None,
            "reference_count": None,
            "median": None,
            "mad": None,
            "lower_quantile": None,
            "upper_quantile": None,
            "lower_bound": None,
            "upper_bound": None,
            "hard_rule": True,
            "provisional": False,
            "fallback_used": False,
        },
        {
            "signal": "cycle",
            "metric": "duration_seconds",
            "rule_type": RULE_TYPE_HARD_POSITIVE_DURATION,
            "method": "hard_rule",
            "reference_population": None,
            "reference_count": None,
            "median": None,
            "mad": None,
            "lower_quantile": None,
            "upper_quantile": None,
            "lower_bound": 0.0,
            "upper_bound": None,
            "hard_rule": True,
            "provisional": False,
            "fallback_used": False,
        },
        {
            "signal": "position",
            "metric": "presence",
            "rule_type": RULE_TYPE_HARD_SIGNAL_PRESENCE,
            "method": "hard_rule",
            "reference_population": None,
            "reference_count": None,
            "median": None,
            "mad": None,
            "lower_quantile": None,
            "upper_quantile": None,
            "lower_bound": None,
            "upper_bound": None,
            "hard_rule": True,
            "provisional": False,
            "fallback_used": False,
        },
    ]
    for signal_name in core_required_signals:
        hard_rules.append(
            {
                "signal": signal_name,
                "metric": "finite_values",
                "rule_type": RULE_TYPE_HARD_FINITE_REQUIRED,
                "method": "hard_rule",
                "reference_population": None,
                "reference_count": None,
                "median": None,
                "mad": None,
                "lower_quantile": None,
                "upper_quantile": None,
                "lower_bound": None,
                "upper_bound": None,
                "hard_rule": True,
                "provisional": False,
                "fallback_used": False,
            }
        )
    return hard_rules


def _representative_population_warning(
    cycle_quality_profile_df: pd.DataFrame, minimum_reference_count: int
) -> str | None:
    if cycle_quality_profile_df.empty:
        return "No profiled cycles were available; no reference population exists."

    unique_sessions = (
        cycle_quality_profile_df["session_id"].nunique()
        if "session_id" in cycle_quality_profile_df.columns
        else 0
    )
    cycle_count = len(cycle_quality_profile_df)
    if unique_sessions <= 1 or cycle_count < max(minimum_reference_count * 5, 200):
        return (
            "The profiling population contains "
            f"{cycle_count} cycle(s) spanning {unique_sessions} session(s). "
            "This may not represent all sessions, the full experiment timeline, "
            "all operating conditions, or degradation states observed over the "
            "complete dataset. Generated rules are marked provisional."
        )
    return None


def generate_validation_rules(
    signal_quality_metrics: pd.DataFrame,
    cycle_quality_profile: pd.DataFrame,
    signal_roles: dict[str, tuple[str, ...]],
    config: RuleGenerationConfig,
    output_directory: Path,
    dataset_name: str = "",
    experiment: str = "",
) -> RuleGenerationResult:
    """Derive robust validation thresholds from the profiled cycle population.

    This function must be given the *profiled* population (never cycles
    that have already been validated/rejected), so that the resulting rules
    are not circularly derived from an already-filtered subset.
    """

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    core_required = signal_roles.get("core_required", ())
    vibration_signals = signal_roles.get("optional_duty_cycled", ())

    cycle_ids = (
        cycle_quality_profile["cycle_id"]
        if "cycle_id" in cycle_quality_profile.columns
        else pd.Series(dtype=int)
    )
    vibration_classification = classify_vibration_availability(
        signal_quality_metrics, cycle_ids, vibration_signals
    )

    representative_warning = _representative_population_warning(
        cycle_quality_profile, config.minimum_reference_count
    )
    provisional = bool(
        config.mark_small_or_limited_population_as_provisional and representative_warning is not None
    )

    threshold_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    generation_timestamp = datetime.now(timezone.utc).isoformat()

    def _record(
        signal_name: str,
        metric: str,
        reference_population_label: str,
        derived: dict[str, object],
    ) -> None:
        row = {
            "signal": signal_name,
            "metric": metric,
            "rule_type": derived["method"],
            "method": derived["method"],
            "reference_population": reference_population_label,
            "reference_count": derived["reference_count"],
            "median": derived["median"],
            "mad": derived["mad"],
            "lower_quantile": config.lower_quantile if derived.get("lower_quantile_value") is not None else None,
            "upper_quantile": config.upper_quantile if derived.get("upper_quantile_value") is not None else None,
            "lower_quantile_value": derived.get("lower_quantile_value"),
            "upper_quantile_value": derived.get("upper_quantile_value"),
            "lower_bound": derived["lower_bound"],
            "upper_bound": derived["upper_bound"],
            "hard_rule": False,
            "provisional": provisional,
            "fallback_used": derived["fallback_used"],
            "generation_timestamp": generation_timestamp,
            "status": derived["status"],
        }
        if derived["status"] == "learned":
            threshold_rows.append(row)
        else:
            skipped_rows.append(row)

    # Core-required, non-vibration signal-level metrics: reference population
    # is every profiled cycle (never restricted).
    for signal_name in core_required:
        signal_df = signal_quality_metrics[signal_quality_metrics["signal_name"] == signal_name]
        for metric in SIGNAL_LEVEL_LEARNABLE_METRICS:
            if metric not in signal_df.columns:
                continue
            derived = _derive_metric_rule(signal_df[metric], config)
            _record(signal_name, metric, "all_profiled_cycles", derived)

    # Vibration signal-level metrics: reference population restricted to
    # vibration_complete cycles only, to avoid mixing zero-sample and
    # complete-burst cycles into one meaningless global threshold.
    complete_cycle_ids = set(
        vibration_classification.loc[
            vibration_classification["vibration_class"] == VIBRATION_COMPLETE, "cycle_id"
        ].tolist()
    )
    for signal_name in vibration_signals:
        signal_df = signal_quality_metrics[
            (signal_quality_metrics["signal_name"] == signal_name)
            & (signal_quality_metrics["cycle_id"].isin(complete_cycle_ids))
        ]
        for metric in SIGNAL_LEVEL_LEARNABLE_METRICS:
            if metric not in signal_df.columns:
                continue
            derived = _derive_metric_rule(signal_df[metric], config)
            _record(signal_name, metric, "vibration_complete_cycles", derived)

    # Cycle-level metrics.
    for metric in CYCLE_LEVEL_LEARNABLE_METRICS:
        if metric not in cycle_quality_profile.columns:
            continue
        derived = _derive_metric_rule(cycle_quality_profile[metric], config)
        _record("cycle", metric, "all_profiled_cycles", derived)

    hard_rule_rows = _hard_rules(tuple(core_required))
    for row in hard_rule_rows:
        row["generation_timestamp"] = generation_timestamp
        row["lower_quantile"] = None
        row["upper_quantile"] = None
        row["lower_quantile_value"] = None
        row["upper_quantile_value"] = None
        row["status"] = "hard_rule"

    all_learned_and_hard = threshold_rows + hard_rule_rows
    validation_thresholds_df = pd.DataFrame(all_learned_and_hard)
    threshold_derivation_summary_df = pd.DataFrame(threshold_rows + skipped_rows + hard_rule_rows)
    skipped_metrics_df = pd.DataFrame(
        [
            {
                "signal": row["signal"],
                "metric": row["metric"],
                "reason": row["status"],
                "reference_count": row["reference_count"],
            }
            for row in skipped_rows
        ]
    )

    vibration_class_counts = (
        vibration_classification["vibration_class"].value_counts().to_dict()
        if not vibration_classification.empty
        else {}
    )

    summary: dict[str, object] = {
        "dataset": dataset_name,
        "experiment": experiment,
        "generated_at": generation_timestamp,
        "selection_mode": config.reference_population,
        "reference_cycles": int(len(cycle_quality_profile)),
        "rules_generated": int(len(threshold_rows)),
        "hard_rules": int(len(hard_rule_rows)),
        "metrics_skipped": int(len(skipped_rows)),
        "provisional_rules": int(len(threshold_rows)) if provisional else 0,
        "representative_population_warning": representative_warning,
        "vibration_classification_counts": {
            VIBRATION_UNAVAILABLE: int(vibration_class_counts.get(VIBRATION_UNAVAILABLE, 0)),
            VIBRATION_PARTIAL: int(vibration_class_counts.get(VIBRATION_PARTIAL, 0)),
            VIBRATION_COMPLETE: int(vibration_class_counts.get(VIBRATION_COMPLETE, 0)),
        },
        "config": {
            "default_method": config.default_method,
            "mad_z_limit": config.mad_z_limit,
            "lower_quantile": config.lower_quantile,
            "upper_quantile": config.upper_quantile,
            "minimum_reference_count": config.minimum_reference_count,
            "fallback_method": config.fallback_method,
            "freeze_rules": config.freeze_rules,
        },
    }

    write_json(
        {"thresholds": sanitize_json_value(validation_thresholds_df.to_dict(orient="records"))},
        output_directory / "validation_thresholds.json",
    )
    threshold_derivation_summary_df.to_csv(
        output_directory / "threshold_derivation_summary.csv", index=False
    )
    skipped_metrics_df.to_csv(output_directory / "skipped_metrics.csv", index=False)
    write_json(summary, output_directory / "rule_generation_summary.json")

    logger.info(
        "Generated %d validation rule(s) (%d hard, %d learned, %d skipped) "
        "from %d reference cycle(s); provisional=%s",
        len(all_learned_and_hard),
        len(hard_rule_rows),
        len(threshold_rows),
        len(skipped_rows),
        len(cycle_quality_profile),
        provisional,
    )

    return RuleGenerationResult(
        validation_thresholds=validation_thresholds_df,
        threshold_derivation_summary=threshold_derivation_summary_df,
        skipped_metrics=skipped_metrics_df,
        rule_generation_summary=summary,
        vibration_classification=vibration_classification,
    )


def load_frozen_thresholds(output_directory: Path) -> pd.DataFrame:
    """Load a previously generated and frozen ``validation_thresholds.json``."""

    thresholds_path = Path(output_directory) / "validation_thresholds.json"
    if not thresholds_path.exists():
        return pd.DataFrame()
    payload = json.loads(thresholds_path.read_text(encoding="utf-8"))
    return pd.DataFrame(payload.get("thresholds", []))
