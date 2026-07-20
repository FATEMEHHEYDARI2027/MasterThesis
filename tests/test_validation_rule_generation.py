"""Behavior tests for the ``validation_rule_generation`` stage."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.validation_rule_generation import (
    DEFAULT_SIGNAL_ROLES,
    RuleGenerationConfig,
    classify_vibration_availability,
    generate_validation_rules,
)

CORE_SIGNALS = ("position", "velocity", "current", "pressure", "temperature")
VIBRATION_SIGNALS = ("vibration_x", "vibration_y", "vibration_z")


def _signal_row(
    cycle_id: int,
    session_id: int,
    signal_name: str,
    sample_count: int,
    finite_sample_count: int | None = None,
    coverage_ratio: float = 0.9,
    signal_range: float = 10.0,
    standard_deviation: float = 2.0,
    estimated_sampling_rate: float = 50.0,
    median_sampling_interval: float = 0.02,
    maximum_timestamp_gap: float = 0.5,
) -> dict[str, object]:
    return {
        "cycle_id": cycle_id,
        "session_id": session_id,
        "signal_name": signal_name,
        "sample_count": sample_count,
        "finite_sample_count": finite_sample_count if finite_sample_count is not None else sample_count,
        "coverage_ratio": coverage_ratio,
        "signal_range": signal_range,
        "standard_deviation": standard_deviation,
        "estimated_sampling_rate": estimated_sampling_rate,
        "median_sampling_interval": median_sampling_interval,
        "maximum_timestamp_gap": maximum_timestamp_gap,
    }


def _build_basic_population(
    cycle_count: int = 60, session_count: int = 3, rng_seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a synthetic profiling population with variance in every metric."""

    rng = np.random.default_rng(rng_seed)
    signal_rows: list[dict[str, object]] = []
    cycle_rows: list[dict[str, object]] = []

    for cycle_id in range(cycle_count):
        session_id = cycle_id % session_count
        for signal_name in CORE_SIGNALS:
            signal_rows.append(
                _signal_row(
                    cycle_id,
                    session_id,
                    signal_name,
                    sample_count=int(100 + rng.normal(0, 5)),
                    coverage_ratio=float(np.clip(0.9 + rng.normal(0, 0.02), 0, 1)),
                    signal_range=float(10 + rng.normal(0, 1)),
                    standard_deviation=float(2 + rng.normal(0, 0.2)),
                )
            )
        # Only the first four cycles carry a full vibration burst; this
        # mirrors the duty-cycled acquisition behavior observed in the real
        # dataset (short bursts every ~10 minutes).
        if cycle_id < 4:
            for signal_name in VIBRATION_SIGNALS:
                signal_rows.append(
                    _signal_row(
                        cycle_id,
                        session_id,
                        signal_name,
                        sample_count=int(200 + rng.normal(0, 10)),
                        coverage_ratio=0.95,
                        signal_range=float(1 + rng.normal(0, 0.1)),
                        standard_deviation=float(0.3 + rng.normal(0, 0.02)),
                    )
                )
        else:
            for signal_name in VIBRATION_SIGNALS:
                signal_rows.append(_signal_row(cycle_id, session_id, signal_name, sample_count=0))

        cycle_rows.append(
            {
                "cycle_id": cycle_id,
                "session_id": session_id,
                "cycle_duration_seconds": float(4 + rng.normal(0, 0.3)),
                "position_stroke_range": float(50 + rng.normal(0, 2)),
            }
        )

    return pd.DataFrame(signal_rows), pd.DataFrame(cycle_rows)


class MedianMadDerivationTests(unittest.TestCase):
    def test_median_mad_rule_uses_configured_z_limit(self) -> None:
        signal_df, cycle_df = _build_basic_population(cycle_count=50)
        config = RuleGenerationConfig.from_mapping({"mad_z_limit": 3.5, "minimum_reference_count": 5})
        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_validation_rules(
                signal_df, cycle_df, DEFAULT_SIGNAL_ROLES, config, Path(temp_dir)
            )

        rows = result.validation_thresholds
        row = rows[
            (rows["signal"] == "position") & (rows["metric"] == "signal_range")
        ].iloc[0]
        self.assertEqual(row["method"], "median_mad")
        median = row["median"]
        mad = row["mad"]
        expected_lower = median - (3.5 / 0.6745) * mad
        expected_upper = median + (3.5 / 0.6745) * mad
        self.assertAlmostEqual(row["lower_bound"], expected_lower, places=6)
        self.assertAlmostEqual(row["upper_bound"], expected_upper, places=6)


class QuantileFallbackTests(unittest.TestCase):
    def test_quantile_fallback_used_when_mad_is_zero(self) -> None:
        signal_df, cycle_df = _build_basic_population(cycle_count=30)
        # Force a near-constant metric for one signal/metric pair with a few
        # outliers so the median is repeated (MAD == 0) but the configured
        # quantiles still differ.
        mask = signal_df["signal_name"] == "pressure"
        signal_df.loc[mask, "standard_deviation"] = 1.0
        signal_df.loc[signal_df[mask].index[:2], "standard_deviation"] = [0.1, 5.0]

        config = RuleGenerationConfig.from_mapping({"minimum_reference_count": 5})
        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_validation_rules(
                signal_df, cycle_df, DEFAULT_SIGNAL_ROLES, config, Path(temp_dir)
            )

        rows = result.validation_thresholds
        row = rows[
            (rows["signal"] == "pressure") & (rows["metric"] == "standard_deviation")
        ]
        self.assertFalse(row.empty)
        self.assertEqual(row.iloc[0]["method"], "quantile")
        self.assertTrue(bool(row.iloc[0]["fallback_used"]))


class ConstantAndInsufficientDataTests(unittest.TestCase):
    def test_constant_metric_is_skipped(self) -> None:
        signal_df, cycle_df = _build_basic_population(cycle_count=30)
        signal_df.loc[signal_df["signal_name"] == "current", "standard_deviation"] = 1.0

        config = RuleGenerationConfig.from_mapping({"minimum_reference_count": 5})
        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_validation_rules(
                signal_df, cycle_df, DEFAULT_SIGNAL_ROLES, config, Path(temp_dir)
            )

        skipped = result.skipped_metrics
        matches = skipped[(skipped["signal"] == "current") & (skipped["metric"] == "standard_deviation")]
        self.assertFalse(matches.empty)
        self.assertEqual(matches.iloc[0]["reason"], "skipped_constant")

    def test_insufficient_reference_population_is_skipped(self) -> None:
        signal_df, cycle_df = _build_basic_population(cycle_count=5)
        config = RuleGenerationConfig.from_mapping({"minimum_reference_count": 20})
        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_validation_rules(
                signal_df, cycle_df, DEFAULT_SIGNAL_ROLES, config, Path(temp_dir)
            )

        self.assertTrue(result.validation_thresholds.empty or (result.validation_thresholds["hard_rule"]).all())
        self.assertFalse(result.skipped_metrics.empty)
        self.assertTrue((result.skipped_metrics["reason"] == "skipped_insufficient_data").all())

    def test_all_null_metric_is_skipped(self) -> None:
        signal_df, cycle_df = _build_basic_population(cycle_count=30)
        signal_df.loc[signal_df["signal_name"] == "velocity", "coverage_ratio"] = np.nan
        config = RuleGenerationConfig.from_mapping({"minimum_reference_count": 5})
        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_validation_rules(
                signal_df, cycle_df, DEFAULT_SIGNAL_ROLES, config, Path(temp_dir)
            )

        skipped = result.skipped_metrics
        matches = skipped[(skipped["signal"] == "velocity") & (skipped["metric"] == "coverage_ratio")]
        self.assertFalse(matches.empty)
        self.assertEqual(matches.iloc[0]["reason"], "skipped_insufficient_data")

    def test_non_finite_values_are_excluded_from_reference_count(self) -> None:
        signal_df, cycle_df = _build_basic_population(cycle_count=30)
        idx = signal_df[signal_df["signal_name"] == "temperature"].index[:3]
        signal_df.loc[idx, "signal_range"] = np.inf
        config = RuleGenerationConfig.from_mapping({"minimum_reference_count": 5})
        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_validation_rules(
                signal_df, cycle_df, DEFAULT_SIGNAL_ROLES, config, Path(temp_dir)
            )
        rows = result.validation_thresholds
        row = rows[(rows["signal"] == "temperature") & (rows["metric"] == "signal_range")]
        self.assertFalse(row.empty)
        self.assertEqual(row.iloc[0]["reference_count"], 27)


class JsonSafetyTests(unittest.TestCase):
    def test_validation_thresholds_json_has_no_nan_or_infinity(self) -> None:
        signal_df, cycle_df = _build_basic_population(cycle_count=30)
        config = RuleGenerationConfig.from_mapping({"minimum_reference_count": 5})
        with tempfile.TemporaryDirectory() as temp_dir:
            generate_validation_rules(signal_df, cycle_df, DEFAULT_SIGNAL_ROLES, config, Path(temp_dir))
            thresholds_text = (Path(temp_dir) / "validation_thresholds.json").read_text(encoding="utf-8")
            summary_text = (Path(temp_dir) / "rule_generation_summary.json").read_text(encoding="utf-8")

        for text in (thresholds_text, summary_text):
            self.assertNotIn("NaN", text)
            self.assertNotIn("Infinity", text)
            json.loads(text)  # must parse as standard JSON


class ProvisionalRuleTests(unittest.TestCase):
    def test_small_population_marks_rules_provisional(self) -> None:
        signal_df, cycle_df = _build_basic_population(cycle_count=30, session_count=1)
        config = RuleGenerationConfig.from_mapping(
            {"minimum_reference_count": 5, "mark_small_or_limited_population_as_provisional": True}
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_validation_rules(
                signal_df, cycle_df, DEFAULT_SIGNAL_ROLES, config, Path(temp_dir)
            )

        learned = result.validation_thresholds[~result.validation_thresholds["hard_rule"]]
        self.assertFalse(learned.empty)
        self.assertTrue(learned["provisional"].all())
        self.assertIsNotNone(result.rule_generation_summary["representative_population_warning"])


class VibrationThresholdIsolationTests(unittest.TestCase):
    def test_vibration_thresholds_use_only_complete_cycles(self) -> None:
        signal_df, cycle_df = _build_basic_population(cycle_count=200, session_count=6)
        config = RuleGenerationConfig.from_mapping({"minimum_reference_count": 4})
        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_validation_rules(
                signal_df, cycle_df, DEFAULT_SIGNAL_ROLES, config, Path(temp_dir)
            )

        rows = result.validation_thresholds
        vibration_rows = rows[rows["signal"] == "vibration_x"]
        self.assertFalse(vibration_rows.empty)
        for _, row in vibration_rows.iterrows():
            self.assertEqual(row["reference_population"], "vibration_complete_cycles")
            # Only cycles 0-3 have a complete vibration burst in the fixture.
            self.assertLessEqual(row["reference_count"], 4)

    def test_no_vibration_threshold_from_zero_sample_cycles(self) -> None:
        signal_df, cycle_df = _build_basic_population(cycle_count=200, session_count=6)
        cycle_ids = cycle_df["cycle_id"]
        classification = classify_vibration_availability(signal_df, cycle_ids, VIBRATION_SIGNALS)
        unavailable = classification[classification["vibration_class"] == "vibration_unavailable"]
        self.assertGreater(len(unavailable), 0)

        config = RuleGenerationConfig.from_mapping({"minimum_reference_count": 4})
        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_validation_rules(
                signal_df, cycle_df, DEFAULT_SIGNAL_ROLES, config, Path(temp_dir)
            )
        rows = result.validation_thresholds
        vibration_rows = rows[rows["signal"].isin(VIBRATION_SIGNALS)]
        # None of the reference counts may include the zero-sample cycles.
        for _, row in vibration_rows.iterrows():
            self.assertLessEqual(row["reference_count"], 4)


class DeterminismTests(unittest.TestCase):
    def test_generation_is_deterministic(self) -> None:
        signal_df, cycle_df = _build_basic_population(cycle_count=60)
        config = RuleGenerationConfig.from_mapping({"minimum_reference_count": 5})
        with tempfile.TemporaryDirectory() as temp_dir_a, tempfile.TemporaryDirectory() as temp_dir_b:
            result_a = generate_validation_rules(
                signal_df, cycle_df, DEFAULT_SIGNAL_ROLES, config, Path(temp_dir_a)
            )
            result_b = generate_validation_rules(
                signal_df, cycle_df, DEFAULT_SIGNAL_ROLES, config, Path(temp_dir_b)
            )

        pd.testing.assert_frame_equal(
            result_a.validation_thresholds.drop(columns=["generation_timestamp"]).reset_index(drop=True),
            result_b.validation_thresholds.drop(columns=["generation_timestamp"]).reset_index(drop=True),
        )


if __name__ == "__main__":
    unittest.main()
