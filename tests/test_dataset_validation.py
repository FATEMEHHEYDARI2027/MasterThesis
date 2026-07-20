"""Behavior tests for the ``dataset_validation`` stage."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.preprocessing.dataset_validation import (
    INVALID_CYCLE,
    VALID_COMPLETE_MULTISENSOR_CYCLE,
    VALID_CORE_CYCLE,
    DatasetValidationConfig,
    validate_dataset,
)
from src.preprocessing.validation_rule_generation import (
    DEFAULT_SIGNAL_ROLES,
    RuleGenerationConfig,
    generate_validation_rules,
)
from tests.test_validation_rule_generation import _build_basic_population


def _generate_thresholds(signal_df: pd.DataFrame, cycle_df: pd.DataFrame, temp_dir: str) -> pd.DataFrame:
    config = RuleGenerationConfig.from_mapping({"minimum_reference_count": 4})
    result = generate_validation_rules(
        signal_df, cycle_df, DEFAULT_SIGNAL_ROLES, config, Path(temp_dir)
    )
    return result.validation_thresholds


class ValidCoreCycleWithoutVibrationTests(unittest.TestCase):
    def test_core_cycle_without_vibration_is_valid_core(self) -> None:
        signal_df, cycle_df = _build_basic_population(cycle_count=60, session_count=3)
        with tempfile.TemporaryDirectory() as temp_dir:
            thresholds = _generate_thresholds(signal_df, cycle_df, temp_dir)
            result = validate_dataset(
                signal_df,
                cycle_df,
                thresholds,
                DEFAULT_SIGNAL_ROLES,
                DatasetValidationConfig(),
                Path(temp_dir) / "validation",
            )

        # Cycle 10 has no vibration at all in the fixture.
        row = result.cycle_validation_results[result.cycle_validation_results["cycle_id"] == 10].iloc[0]
        self.assertEqual(row["final_class"], VALID_CORE_CYCLE)
        self.assertIn("vibration_unavailable", row["reasons"])


class ValidCompleteMultisensorCycleTests(unittest.TestCase):
    def test_complete_vibration_cycle_is_valid_complete_multisensor(self) -> None:
        signal_df, cycle_df = _build_basic_population(cycle_count=60, session_count=3)
        with tempfile.TemporaryDirectory() as temp_dir:
            thresholds = _generate_thresholds(signal_df, cycle_df, temp_dir)
            result = validate_dataset(
                signal_df,
                cycle_df,
                thresholds,
                DEFAULT_SIGNAL_ROLES,
                DatasetValidationConfig(),
                Path(temp_dir) / "validation",
            )

        # Cycles 0-3 carry a complete vibration burst in the fixture.
        row = result.cycle_validation_results[result.cycle_validation_results["cycle_id"] == 0].iloc[0]
        self.assertEqual(row["final_class"], VALID_COMPLETE_MULTISENSOR_CYCLE)
        self.assertEqual(row["vibration_class"], "vibration_complete")


class VibrationAvailabilityHandlingTests(unittest.TestCase):
    def test_vibration_unavailable_does_not_invalidate_core_cycle(self) -> None:
        signal_df, cycle_df = _build_basic_population(cycle_count=60, session_count=3)
        with tempfile.TemporaryDirectory() as temp_dir:
            thresholds = _generate_thresholds(signal_df, cycle_df, temp_dir)
            result = validate_dataset(
                signal_df,
                cycle_df,
                thresholds,
                DEFAULT_SIGNAL_ROLES,
                DatasetValidationConfig(),
                Path(temp_dir) / "validation",
            )

        vibration_unavailable_rows = result.cycle_validation_results[
            result.cycle_validation_results["vibration_class"] == "vibration_unavailable"
        ]
        self.assertFalse(vibration_unavailable_rows.empty)
        # A cycle may still be invalid for an unrelated reason (e.g. a core
        # signal outlier), but it must never be invalid *solely* because
        # vibration was unavailable.
        for _, row in vibration_unavailable_rows.iterrows():
            reasons = [reason for reason in row["reasons"].split(",") if reason]
            other_reasons = [reason for reason in reasons if reason != "vibration_unavailable"]
            if row["final_class"] == INVALID_CYCLE:
                self.assertTrue(other_reasons, f"cycle {row['cycle_id']} invalid with reasons={reasons}")

    def test_partial_vibration_classified_correctly(self) -> None:
        signal_df, cycle_df = _build_basic_population(cycle_count=60, session_count=3)
        # Make cycle 0 have only one vibration axis present -> partial.
        mask = (signal_df["cycle_id"] == 0) & (signal_df["signal_name"].isin(["vibration_y", "vibration_z"]))
        signal_df.loc[mask, "sample_count"] = 0
        signal_df.loc[mask, "finite_sample_count"] = 0

        with tempfile.TemporaryDirectory() as temp_dir:
            thresholds = _generate_thresholds(signal_df, cycle_df, temp_dir)
            result = validate_dataset(
                signal_df,
                cycle_df,
                thresholds,
                DEFAULT_SIGNAL_ROLES,
                DatasetValidationConfig(),
                Path(temp_dir) / "validation",
            )
        row = result.cycle_validation_results[result.cycle_validation_results["cycle_id"] == 0].iloc[0]
        self.assertEqual(row["vibration_class"], "vibration_partial")
        self.assertEqual(row["final_class"], VALID_CORE_CYCLE)


class MissingCoreSignalTests(unittest.TestCase):
    def test_missing_position_invalidates_cycle(self) -> None:
        signal_df, cycle_df = _build_basic_population(cycle_count=60, session_count=3)
        signal_df = signal_df[~((signal_df["cycle_id"] == 5) & (signal_df["signal_name"] == "position"))]

        with tempfile.TemporaryDirectory() as temp_dir:
            thresholds = _generate_thresholds(signal_df, cycle_df, temp_dir)
            result = validate_dataset(
                signal_df,
                cycle_df,
                thresholds,
                DEFAULT_SIGNAL_ROLES,
                DatasetValidationConfig(),
                Path(temp_dir) / "validation",
            )
        row = result.cycle_validation_results[result.cycle_validation_results["cycle_id"] == 5].iloc[0]
        self.assertEqual(row["final_class"], INVALID_CYCLE)
        self.assertIn("position:missing_required_signal", row["reasons"])

    def test_missing_configurable_core_signal_invalidates_cycle(self) -> None:
        signal_df, cycle_df = _build_basic_population(cycle_count=60, session_count=3)
        signal_df = signal_df[~((signal_df["cycle_id"] == 6) & (signal_df["signal_name"] == "pressure"))]

        with tempfile.TemporaryDirectory() as temp_dir:
            thresholds = _generate_thresholds(signal_df, cycle_df, temp_dir)
            result = validate_dataset(
                signal_df,
                cycle_df,
                thresholds,
                DEFAULT_SIGNAL_ROLES,
                DatasetValidationConfig(),
                Path(temp_dir) / "validation",
            )
        row = result.cycle_validation_results[result.cycle_validation_results["cycle_id"] == 6].iloc[0]
        self.assertEqual(row["final_class"], INVALID_CYCLE)


class LearnedBoundViolationTests(unittest.TestCase):
    def test_lower_bound_violation_invalidates_cycle(self) -> None:
        signal_df, cycle_df = _build_basic_population(cycle_count=60, session_count=3)
        with tempfile.TemporaryDirectory() as temp_dir:
            thresholds = _generate_thresholds(signal_df, cycle_df, temp_dir)
            mask = (signal_df["cycle_id"] == 7) & (signal_df["signal_name"] == "current")
            signal_df.loc[mask, "sample_count"] = 1
            signal_df.loc[mask, "finite_sample_count"] = 1
            result = validate_dataset(
                signal_df,
                cycle_df,
                thresholds,
                DEFAULT_SIGNAL_ROLES,
                DatasetValidationConfig(),
                Path(temp_dir) / "validation",
            )
        row = result.cycle_validation_results[result.cycle_validation_results["cycle_id"] == 7].iloc[0]
        self.assertEqual(row["final_class"], INVALID_CYCLE)
        self.assertIn("current:sample_count_below_lower_bound", row["reasons"])

    def test_upper_bound_violation_invalidates_cycle(self) -> None:
        signal_df, cycle_df = _build_basic_population(cycle_count=60, session_count=3)
        with tempfile.TemporaryDirectory() as temp_dir:
            thresholds = _generate_thresholds(signal_df, cycle_df, temp_dir)
            mask = (signal_df["cycle_id"] == 8) & (signal_df["signal_name"] == "current")
            signal_df.loc[mask, "sample_count"] = 100000
            signal_df.loc[mask, "finite_sample_count"] = 100000
            result = validate_dataset(
                signal_df,
                cycle_df,
                thresholds,
                DEFAULT_SIGNAL_ROLES,
                DatasetValidationConfig(),
                Path(temp_dir) / "validation",
            )
        row = result.cycle_validation_results[result.cycle_validation_results["cycle_id"] == 8].iloc[0]
        self.assertEqual(row["final_class"], INVALID_CYCLE)
        self.assertIn("current:sample_count_above_upper_bound", row["reasons"])


class HardRuleViolationTests(unittest.TestCase):
    def test_invalid_interval_invalidates_cycle(self) -> None:
        signal_df, cycle_df = _build_basic_population(cycle_count=60, session_count=3)
        cycle_df = cycle_df.copy()
        cycle_df["start_time"] = pd.Timestamp("2026-01-01T00:00:10")
        cycle_df["end_time"] = pd.Timestamp("2026-01-01T00:00:00")

        with tempfile.TemporaryDirectory() as temp_dir:
            thresholds = _generate_thresholds(signal_df, cycle_df, temp_dir)
            result = validate_dataset(
                signal_df,
                cycle_df,
                thresholds,
                DEFAULT_SIGNAL_ROLES,
                DatasetValidationConfig(),
                Path(temp_dir) / "validation",
            )
        self.assertTrue((result.cycle_validation_results["final_class"] == INVALID_CYCLE).all())
        self.assertTrue(
            result.cycle_validation_results["reasons"].str.contains("invalid_cycle_interval").all()
        )

    def test_no_finite_values_invalidates_cycle(self) -> None:
        signal_df, cycle_df = _build_basic_population(cycle_count=60, session_count=3)
        mask = (signal_df["cycle_id"] == 9) & (signal_df["signal_name"] == "velocity")
        signal_df.loc[mask, "finite_sample_count"] = 0

        with tempfile.TemporaryDirectory() as temp_dir:
            thresholds = _generate_thresholds(signal_df, cycle_df, temp_dir)
            result = validate_dataset(
                signal_df,
                cycle_df,
                thresholds,
                DEFAULT_SIGNAL_ROLES,
                DatasetValidationConfig(),
                Path(temp_dir) / "validation",
            )
        row = result.cycle_validation_results[result.cycle_validation_results["cycle_id"] == 9].iloc[0]
        self.assertEqual(row["final_class"], INVALID_CYCLE)
        self.assertIn("velocity:no_finite_values", row["reasons"])


class MissingGeneratedRuleTests(unittest.TestCase):
    def test_missing_generated_rule_is_ignored_by_default(self) -> None:
        signal_df, cycle_df = _build_basic_population(cycle_count=60, session_count=3)
        with tempfile.TemporaryDirectory() as temp_dir:
            thresholds = _generate_thresholds(signal_df, cycle_df, temp_dir)
            thresholds = thresholds[thresholds["metric"] != "signal_range"]
            result = validate_dataset(
                signal_df,
                cycle_df,
                thresholds,
                DEFAULT_SIGNAL_ROLES,
                DatasetValidationConfig(fail_on_missing_generated_rule=False),
                Path(temp_dir) / "validation",
            )
        self.assertFalse((result.cycle_validation_results["reasons"].str.contains("rule_not_generated")).any())

    def test_missing_generated_rule_can_be_flagged(self) -> None:
        signal_df, cycle_df = _build_basic_population(cycle_count=60, session_count=3)
        with tempfile.TemporaryDirectory() as temp_dir:
            thresholds = _generate_thresholds(signal_df, cycle_df, temp_dir)
            thresholds = thresholds[thresholds["metric"] != "signal_range"]
            result = validate_dataset(
                signal_df,
                cycle_df,
                thresholds,
                DEFAULT_SIGNAL_ROLES,
                DatasetValidationConfig(fail_on_missing_generated_rule=True),
                Path(temp_dir) / "validation",
            )
        self.assertTrue((result.cycle_validation_results["reasons"].str.contains("rule_not_generated")).any())


class ProvisionalRulesAllowedTests(unittest.TestCase):
    def test_provisional_rules_are_applied_when_allowed(self) -> None:
        signal_df, cycle_df = _build_basic_population(cycle_count=30, session_count=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            thresholds = _generate_thresholds(signal_df, cycle_df, temp_dir)
            self.assertTrue(thresholds[~thresholds["hard_rule"]]["provisional"].all())
            result = validate_dataset(
                signal_df,
                cycle_df,
                thresholds,
                DEFAULT_SIGNAL_ROLES,
                DatasetValidationConfig(allow_provisional_rules=True),
                Path(temp_dir) / "validation",
            )
        self.assertFalse(result.cycle_validation_results.empty)


class SummaryAndReasonCodeTests(unittest.TestCase):
    def test_summary_counts_and_reason_summary(self) -> None:
        signal_df, cycle_df = _build_basic_population(cycle_count=60, session_count=3)
        with tempfile.TemporaryDirectory() as temp_dir:
            thresholds = _generate_thresholds(signal_df, cycle_df, temp_dir)
            result = validate_dataset(
                signal_df,
                cycle_df,
                thresholds,
                DEFAULT_SIGNAL_ROLES,
                DatasetValidationConfig(),
                Path(temp_dir) / "validation",
            )

        summary = result.validation_summary
        self.assertEqual(summary["cycles_evaluated"], 60)
        self.assertEqual(
            summary["valid_core_cycles"] + summary["valid_complete_multisensor_cycles"] + summary["invalid_cycles"],
            60,
        )
        self.assertGreater(len(result.validation_reason_summary), 0)


class OriginalIdentifierPreservationTests(unittest.TestCase):
    def test_cycle_id_and_session_id_are_preserved(self) -> None:
        signal_df, cycle_df = _build_basic_population(cycle_count=10, session_count=2)
        with tempfile.TemporaryDirectory() as temp_dir:
            thresholds = _generate_thresholds(signal_df, cycle_df, temp_dir)
            result = validate_dataset(
                signal_df,
                cycle_df,
                thresholds,
                DEFAULT_SIGNAL_ROLES,
                DatasetValidationConfig(),
                Path(temp_dir) / "validation",
            )
        self.assertEqual(
            sorted(result.cycle_validation_results["cycle_id"].tolist()),
            sorted(cycle_df["cycle_id"].tolist()),
        )
        merged = result.cycle_validation_results.merge(cycle_df, on="cycle_id", suffixes=("", "_orig"))
        self.assertTrue((merged["session_id"] == merged["session_id_orig"]).all())


if __name__ == "__main__":
    unittest.main()
