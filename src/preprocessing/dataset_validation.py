"""Frozen-rule dataset validation.

This module implements the ``dataset_validation`` pipeline stage. It runs
**after** ``validation_rule_generation`` and applies the frozen, previously
generated hard and learned rules consistently to every profiled cycle. It
never re-derives thresholds -- that circular dependency is exactly what the
profile -> generate rules -> freeze -> validate sequence is designed to
avoid.

Classification
---------------
Every cycle receives exactly one final class:

``valid_core_cycle``
    Every configured ``core_required`` signal satisfies its hard and
    learned rules. Vibration may be unavailable or only partially available
    -- the *absence* of vibration alone never invalidates a core cycle,
    because the ESP32 vibration acquisition is intentionally duty-cycled.

``valid_complete_multisensor_cycle``
    Everything required for ``valid_core_cycle`` holds, and in addition the
    cycle's vibration class is ``vibration_complete`` and every applicable
    vibration quality rule is satisfied.

``invalid_cycle``
    One or more ``core_required`` signals fail a hard or learned rule, a
    cycle-level hard rule fails, or the cycle could not be reliably used
    because of an actual technical corruption.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.preprocessing.validation_rule_generation import (
    VIBRATION_COMPLETE,
    VIBRATION_PARTIAL,
    VIBRATION_UNAVAILABLE,
    classify_vibration_availability,
    normalize_signal_roles,
    write_json,
)

logger = logging.getLogger(__name__)

DEFAULT_DATASET_VALIDATION_CONFIG: dict[str, object] = {
    "enabled": True,
    "fail_on_missing_generated_rule": False,
    "allow_provisional_rules": True,
    "write_signal_level_results": True,
}

VALID_CORE_CYCLE = "valid_core_cycle"
VALID_COMPLETE_MULTISENSOR_CYCLE = "valid_complete_multisensor_cycle"
INVALID_CYCLE = "invalid_cycle"


@dataclass(slots=True)
class DatasetValidationConfig:
    """Validated configuration for the ``dataset_validation`` stage."""

    enabled: bool = True
    fail_on_missing_generated_rule: bool = False
    allow_provisional_rules: bool = True
    write_signal_level_results: bool = True

    @classmethod
    def from_mapping(cls, mapping: dict[str, object] | None) -> "DatasetValidationConfig":
        merged = dict(DEFAULT_DATASET_VALIDATION_CONFIG)
        if mapping:
            merged.update(mapping)
        return cls(
            enabled=bool(merged["enabled"]),
            fail_on_missing_generated_rule=bool(merged["fail_on_missing_generated_rule"]),
            allow_provisional_rules=bool(merged["allow_provisional_rules"]),
            write_signal_level_results=bool(merged["write_signal_level_results"]),
        )


@dataclass(slots=True)
class DatasetValidationResult:
    """Outcome of one dataset-validation run."""

    cycle_validation_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    signal_validation_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation_reason_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation_summary: dict[str, object] = field(default_factory=dict)
    valid_core_cycles: pd.DataFrame = field(default_factory=pd.DataFrame)
    valid_complete_multisensor_cycles: pd.DataFrame = field(default_factory=pd.DataFrame)
    invalid_cycles: pd.DataFrame = field(default_factory=pd.DataFrame)


def _bounds_for(
    thresholds_df: pd.DataFrame, signal_name: str, metric: str
) -> pd.Series | None:
    matches = thresholds_df[
        (thresholds_df["signal"] == signal_name) & (thresholds_df["metric"] == metric)
    ]
    if matches.empty:
        return None
    return matches.iloc[0]


def _evaluate_signal(
    signal_row: pd.Series,
    thresholds_df: pd.DataFrame,
    signal_name: str,
    config: DatasetValidationConfig,
) -> tuple[bool, list[str]]:
    """Evaluate one (cycle, signal) row against hard and learned rules."""

    reasons: list[str] = []

    missing_signal = bool(signal_row.get("missing_signal", signal_row.get("is_missing", False)))
    if missing_signal:
        reasons.append("missing_required_signal")
        return False, reasons

    finite_count = signal_row.get("finite_sample_count", signal_row.get("finite_count"))
    if finite_count is None or pd.isna(finite_count) or float(finite_count) <= 0:
        reasons.append("no_finite_values")
        return False, reasons

    for metric, lower_reason, upper_reason in (
        ("sample_count", "sample_count_below_lower_bound", "sample_count_above_upper_bound"),
        (
            "finite_sample_count",
            "sample_count_below_lower_bound",
            "sample_count_above_upper_bound",
        ),
        ("coverage_ratio", "coverage_below_lower_bound", None),
        ("maximum_timestamp_gap", None, "maximum_gap_above_upper_bound"),
        ("signal_range", "signal_range_below_lower_bound", "signal_range_above_upper_bound"),
    ):
        if metric not in signal_row.index:
            continue
        value = signal_row.get(metric)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        rule_row = _bounds_for(thresholds_df, signal_name, metric)
        if rule_row is None:
            if config.fail_on_missing_generated_rule:
                reasons.append("rule_not_generated")
            continue
        if bool(rule_row.get("provisional", False)) and not config.allow_provisional_rules:
            continue
        lower_bound = rule_row.get("lower_bound")
        upper_bound = rule_row.get("upper_bound")
        if lower_reason and lower_bound is not None and not pd.isna(lower_bound):
            if float(value) < float(lower_bound):
                reasons.append(lower_reason)
        if upper_reason and upper_bound is not None and not pd.isna(upper_bound):
            if float(value) > float(upper_bound):
                reasons.append(upper_reason)

    return (len(reasons) == 0), reasons


def _evaluate_cycle_level_hard_rules(cycle_row: pd.Series) -> list[str]:
    reasons: list[str] = []
    start_time = cycle_row.get("start_time")
    end_time = cycle_row.get("end_time")
    if start_time is not None and end_time is not None:
        try:
            if pd.Timestamp(start_time) >= pd.Timestamp(end_time):
                reasons.append("invalid_cycle_interval")
        except (TypeError, ValueError):
            reasons.append("invalid_cycle_interval")
    duration = cycle_row.get("cycle_duration_seconds")
    if duration is not None and not pd.isna(duration) and float(duration) <= 0:
        reasons.append("duration_below_lower_bound")
    return reasons


def _evaluate_cycle_level_learned_rules(
    cycle_row: pd.Series, thresholds_df: pd.DataFrame
) -> list[str]:
    reasons: list[str] = []
    duration = cycle_row.get("cycle_duration_seconds")
    if duration is not None and not pd.isna(duration):
        rule_row = _bounds_for(thresholds_df, "cycle", "cycle_duration_seconds")
        if rule_row is not None:
            lower_bound, upper_bound = rule_row.get("lower_bound"), rule_row.get("upper_bound")
            if lower_bound is not None and not pd.isna(lower_bound) and float(duration) < float(lower_bound):
                reasons.append("duration_below_lower_bound")
            if upper_bound is not None and not pd.isna(upper_bound) and float(duration) > float(upper_bound):
                reasons.append("duration_above_upper_bound")

    stroke_range = cycle_row.get("position_stroke_range")
    if stroke_range is not None and not pd.isna(stroke_range):
        rule_row = _bounds_for(thresholds_df, "cycle", "position_stroke_range")
        if rule_row is not None:
            lower_bound, upper_bound = rule_row.get("lower_bound"), rule_row.get("upper_bound")
            if lower_bound is not None and not pd.isna(lower_bound) and float(stroke_range) < float(lower_bound):
                reasons.append("position_stroke_out_of_range")
            if upper_bound is not None and not pd.isna(upper_bound) and float(stroke_range) > float(upper_bound):
                reasons.append("position_stroke_out_of_range")
    return reasons


def validate_dataset(
    signal_quality_metrics: pd.DataFrame,
    cycle_quality_profile: pd.DataFrame,
    validation_thresholds: pd.DataFrame,
    signal_roles: dict[str, tuple[str, ...]] | None,
    config: DatasetValidationConfig,
    output_directory: Path,
    dataset_name: str = "",
    experiment: str = "",
) -> DatasetValidationResult:
    """Apply the frozen validation rules to every profiled cycle."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    roles = normalize_signal_roles(signal_roles)
    core_required = roles.get("core_required", ())
    vibration_signals = roles.get("optional_duty_cycled", ())

    cycle_ids = (
        cycle_quality_profile["cycle_id"]
        if "cycle_id" in cycle_quality_profile.columns
        else pd.Series(dtype=int)
    )
    vibration_classification = classify_vibration_availability(
        signal_quality_metrics, cycle_ids, vibration_signals
    )
    vibration_class_by_cycle = dict(
        zip(
            vibration_classification["cycle_id"],
            vibration_classification["vibration_class"],
        )
    )

    signal_rows: list[dict[str, object]] = []
    cycle_rows: list[dict[str, object]] = []

    signal_quality_by_cycle = (
        signal_quality_metrics.groupby("cycle_id")
        if not signal_quality_metrics.empty
        else None
    )

    for cycle_row in cycle_quality_profile.itertuples(index=False):
        cycle_dict = cycle_row._asdict()
        cycle_id = int(cycle_dict["cycle_id"])
        session_id = cycle_dict.get("session_id")

        reasons: list[str] = []
        reasons.extend(_evaluate_cycle_level_hard_rules(pd.Series(cycle_dict)))
        reasons.extend(_evaluate_cycle_level_learned_rules(pd.Series(cycle_dict), validation_thresholds))

        cycle_signal_df = (
            signal_quality_by_cycle.get_group(cycle_id)
            if signal_quality_by_cycle is not None and cycle_id in getattr(
                signal_quality_by_cycle, "groups", {}
            )
            else pd.DataFrame()
        )

        core_valid = True
        for signal_name in core_required:
            matches = (
                cycle_signal_df[cycle_signal_df["signal_name"] == signal_name]
                if not cycle_signal_df.empty
                else pd.DataFrame()
            )
            if matches.empty:
                signal_valid = False
                signal_reasons = ["missing_required_signal"]
                if signal_name == "position":
                    signal_reasons = ["missing_required_signal"]
            else:
                signal_valid, signal_reasons = _evaluate_signal(
                    matches.iloc[0], validation_thresholds, signal_name, config
                )
            if not signal_valid:
                core_valid = False
                reasons.extend(f"{signal_name}:{reason}" for reason in signal_reasons)

            signal_rows.append(
                {
                    "cycle_id": cycle_id,
                    "session_id": session_id,
                    "signal_name": signal_name,
                    "role": "core_required",
                    "signal_valid": signal_valid,
                    "reasons": ",".join(signal_reasons) if not matches.empty else "missing_required_signal",
                }
            )

        vibration_class = vibration_class_by_cycle.get(cycle_id, VIBRATION_UNAVAILABLE)
        vibration_reasons: list[str] = []
        vibration_quality_ok = True
        for signal_name in vibration_signals:
            matches = (
                cycle_signal_df[cycle_signal_df["signal_name"] == signal_name]
                if not cycle_signal_df.empty
                else pd.DataFrame()
            )
            if vibration_class == VIBRATION_COMPLETE and not matches.empty:
                signal_valid, signal_reasons = _evaluate_signal(
                    matches.iloc[0], validation_thresholds, signal_name, config
                )
                if not signal_valid:
                    vibration_quality_ok = False
                    vibration_reasons.extend(signal_reasons)
                signal_rows.append(
                    {
                        "cycle_id": cycle_id,
                        "session_id": session_id,
                        "signal_name": signal_name,
                        "role": "optional_duty_cycled",
                        "signal_valid": signal_valid,
                        "reasons": ",".join(signal_reasons),
                    }
                )
            else:
                signal_rows.append(
                    {
                        "cycle_id": cycle_id,
                        "session_id": session_id,
                        "signal_name": signal_name,
                        "role": "optional_duty_cycled",
                        "signal_valid": None,
                        "reasons": vibration_class,
                    }
                )

        if vibration_class == VIBRATION_UNAVAILABLE:
            reasons.append("vibration_unavailable")
        elif vibration_class == VIBRATION_PARTIAL:
            reasons.append("vibration_partial")

        has_hard_or_core_failure = bool(
            core_valid is False
            or any(
                reason
                for reason in reasons
                if reason
                not in (
                    "vibration_unavailable",
                    "vibration_partial",
                )
            )
        )

        if has_hard_or_core_failure:
            final_class = INVALID_CYCLE
        elif vibration_class == VIBRATION_COMPLETE and vibration_quality_ok:
            final_class = VALID_COMPLETE_MULTISENSOR_CYCLE
        else:
            final_class = VALID_CORE_CYCLE

        cycle_rows.append(
            {
                "cycle_id": cycle_id,
                "session_id": session_id,
                "final_class": final_class,
                "vibration_class": vibration_class,
                "core_valid": core_valid,
                "reasons": ",".join(reasons) if reasons else "",
            }
        )

    cycle_validation_df = pd.DataFrame(cycle_rows)
    signal_validation_df = pd.DataFrame(signal_rows)

    reason_counter: dict[str, int] = {}
    for reasons_str in cycle_validation_df.get("reasons", pd.Series(dtype=str)):
        for reason in str(reasons_str).split(","):
            reason = reason.strip()
            if reason:
                reason_counter[reason] = reason_counter.get(reason, 0) + 1
    validation_reason_summary_df = pd.DataFrame(
        [{"reason": reason, "count": count} for reason, count in sorted(reason_counter.items())]
    )

    valid_core_df = cycle_validation_df[cycle_validation_df["final_class"] == VALID_CORE_CYCLE]
    valid_complete_df = cycle_validation_df[
        cycle_validation_df["final_class"] == VALID_COMPLETE_MULTISENSOR_CYCLE
    ]
    invalid_df = cycle_validation_df[cycle_validation_df["final_class"] == INVALID_CYCLE]

    vibration_counts = (
        cycle_validation_df["vibration_class"].value_counts().to_dict()
        if not cycle_validation_df.empty
        else {}
    )

    summary: dict[str, object] = {
        "dataset": dataset_name,
        "experiment": experiment,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cycles_evaluated": int(len(cycle_validation_df)),
        "valid_core_cycles": int(len(valid_core_df)),
        "valid_complete_multisensor_cycles": int(len(valid_complete_df)),
        "invalid_cycles": int(len(invalid_df)),
        "vibration_unavailable_cycles": int(vibration_counts.get(VIBRATION_UNAVAILABLE, 0)),
        "vibration_partial_cycles": int(vibration_counts.get(VIBRATION_PARTIAL, 0)),
        "vibration_complete_cycles": int(vibration_counts.get(VIBRATION_COMPLETE, 0)),
        "reason_counts": reason_counter,
    }

    cycle_validation_df.to_parquet(output_directory / "cycle_validation_results.parquet", index=False)
    if config.write_signal_level_results:
        signal_validation_df.to_parquet(
            output_directory / "signal_validation_results.parquet", index=False
        )
    else:
        pd.DataFrame().to_parquet(output_directory / "signal_validation_results.parquet", index=False)
    validation_reason_summary_df.to_csv(
        output_directory / "validation_reason_summary.csv", index=False
    )
    write_json(summary, output_directory / "validation_summary.json")
    valid_core_df.to_parquet(output_directory / "valid_core_cycles.parquet", index=False)
    valid_complete_df.to_parquet(
        output_directory / "valid_complete_multisensor_cycles.parquet", index=False
    )
    invalid_df.to_parquet(output_directory / "invalid_cycles.parquet", index=False)

    logger.info(
        "Dataset validation: %d evaluated, %d valid_core, %d valid_complete_multisensor, "
        "%d invalid (vibration unavailable=%d partial=%d complete=%d)",
        summary["cycles_evaluated"],
        summary["valid_core_cycles"],
        summary["valid_complete_multisensor_cycles"],
        summary["invalid_cycles"],
        summary["vibration_unavailable_cycles"],
        summary["vibration_partial_cycles"],
        summary["vibration_complete_cycles"],
    )

    return DatasetValidationResult(
        cycle_validation_results=cycle_validation_df,
        signal_validation_results=signal_validation_df,
        validation_reason_summary=validation_reason_summary_df,
        validation_summary=summary,
        valid_core_cycles=valid_core_df,
        valid_complete_multisensor_cycles=valid_complete_df,
        invalid_cycles=invalid_df,
    )
