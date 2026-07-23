"""Behavior tests for the padding-based ``cycle_tensor_generation`` stage."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_engineering.cycle_tensor_generation import (
    DEFAULT_REQUIRED_SIGNALS,
    CycleTensorGenerationConfig,
    MissingRequiredSignalError,
    build_cycle_length_statistics,
    build_cycle_tensor,
    build_padding_mask,
    compute_cycle_lengths,
    determine_target_length,
    generate_cycle_tensor_dataset,
    pad_signal,
)

CYCLE_START = pd.Timestamp("2026-01-01 00:00:00")


def _write_measurement_dataset(root_dir: Path, measurements_df: pd.DataFrame) -> Path:
    """Write one long-format measurement Parquet dataset for ``ds.dataset`` reads."""

    measurement_dir = root_dir / "measurements"
    measurement_dir.mkdir(parents=True, exist_ok=True)
    measurements_df.to_parquet(measurement_dir / "part-0.parquet", index=False)
    return measurement_dir


def _measurement_row(cycle_id: int, signal_name: str, time: pd.Timestamp, value: float) -> dict:
    return {"cycle_id": cycle_id, "signal_name": signal_name, "time": time, "value": value}


def _build_cycle_measurements(
    cycle_id: int, position_sample_count: int, start: pd.Timestamp = CYCLE_START
) -> pd.DataFrame:
    """Build one cycle's raw, irregularly sampled measurements.

    Every required signal keeps its own native sample count -- exactly the
    situation the padding methodology (as opposed to interpolation) must
    reconcile without fabricating values.
    """

    rows: list[dict] = []
    for index in range(position_sample_count):
        rows.append(
            _measurement_row(
                cycle_id, "position", start + pd.Timedelta(seconds=index), float(index) * 10.0
            )
        )
    # velocity: always exactly 2 raw samples, regardless of position's length.
    for index, value in enumerate((1.0, 2.0)):
        rows.append(
            _measurement_row(cycle_id, "velocity", start + pd.Timedelta(seconds=index), value)
        )
    # current: always exactly 3 raw samples.
    for index, value in enumerate((0.1, 0.2, 0.3)):
        rows.append(
            _measurement_row(cycle_id, "current", start + pd.Timedelta(seconds=index), value)
        )
    # pressure/temperature: single raw sample (constant sensors).
    rows.append(_measurement_row(cycle_id, "pressure", start, 42.0))
    rows.append(_measurement_row(cycle_id, "temperature", start, 21.5))
    return pd.DataFrame(rows)


class PadSignalTests(unittest.TestCase):
    def test_edge_padding_repeats_last_value_at_the_end(self) -> None:
        values = np.array([0.0, 10.0, 40.0, 80.0])
        padded, padded_count, truncated_count = pad_signal(values, 6, "edge")
        np.testing.assert_array_equal(padded, [0.0, 10.0, 40.0, 80.0, 80.0, 80.0])
        self.assertEqual(padded_count, 2)
        self.assertEqual(truncated_count, 0)

    def test_never_pads_with_zero(self) -> None:
        values = np.array([5.0, 6.0])
        padded, _, _ = pad_signal(values, 4, "edge")
        self.assertNotIn(0.0, padded[2:])
        np.testing.assert_array_equal(padded[2:], [6.0, 6.0])

    def test_truncates_only_at_the_end(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        truncated, padded_count, truncated_count = pad_signal(values, 3, "edge")
        np.testing.assert_array_equal(truncated, [1.0, 2.0, 3.0])
        self.assertEqual(padded_count, 0)
        self.assertEqual(truncated_count, 2)

    def test_exact_length_is_unchanged(self) -> None:
        values = np.array([1.0, 2.0, 3.0])
        resized, padded_count, truncated_count = pad_signal(values, 3, "edge")
        np.testing.assert_array_equal(resized, values)
        self.assertEqual(padded_count, 0)
        self.assertEqual(truncated_count, 0)

    def test_rejects_unsupported_padding_method(self) -> None:
        with self.assertRaises(ValueError):
            pad_signal(np.array([1.0, 2.0]), 5, "zero")

    def test_rejects_empty_signal(self) -> None:
        with self.assertRaises(ValueError):
            pad_signal(np.array([]), 5, "edge")


class BuildPaddingMaskTests(unittest.TestCase):
    def test_mask_marks_real_samples_and_padding(self) -> None:
        mask = build_padding_mask(original_length=4, target_length=6)
        np.testing.assert_array_equal(mask, [1, 1, 1, 1, 0, 0])

    def test_mask_is_all_real_when_exact_length(self) -> None:
        mask = build_padding_mask(original_length=5, target_length=5)
        np.testing.assert_array_equal(mask, [1, 1, 1, 1, 1])

    def test_mask_is_all_real_when_truncated(self) -> None:
        mask = build_padding_mask(original_length=10, target_length=4)
        np.testing.assert_array_equal(mask, [1, 1, 1, 1])


class ComputeCycleLengthsTests(unittest.TestCase):
    def test_counts_reference_signal_samples_per_cycle(self) -> None:
        measurements_df = pd.concat(
            [
                _build_cycle_measurements(1, position_sample_count=5),
                _build_cycle_measurements(2, position_sample_count=9),
            ],
            ignore_index=True,
        )
        lengths = compute_cycle_lengths(measurements_df, "position")
        self.assertEqual(lengths.loc[1], 5)
        self.assertEqual(lengths.loc[2], 9)

    def test_empty_measurements_returns_empty_series(self) -> None:
        lengths = compute_cycle_lengths(pd.DataFrame(columns=["cycle_id", "signal_name"]), "position")
        self.assertTrue(lengths.empty)


class DetermineTargetLengthTests(unittest.TestCase):
    def test_max_strategy_uses_the_longest_cycle(self) -> None:
        lengths = pd.Series([5, 10, 20, 100])
        self.assertEqual(determine_target_length(lengths, "max", 99), 100)

    def test_percentile_strategy_uses_requested_percentile(self) -> None:
        lengths = pd.Series(range(1, 101))  # 1..100
        target_length = determine_target_length(lengths, "percentile", 99)
        self.assertEqual(target_length, int(np.ceil(np.percentile(lengths.to_numpy(), 99))))

    def test_percentile_never_exceeds_max_for_this_distribution(self) -> None:
        lengths = pd.Series(range(1, 101))
        target_length = determine_target_length(lengths, "percentile", 99)
        self.assertLessEqual(target_length, int(lengths.max()))

    def test_rejects_unsupported_strategy(self) -> None:
        with self.assertRaises(ValueError):
            determine_target_length(pd.Series([1, 2, 3]), "cubic", 99)

    def test_rejects_empty_lengths(self) -> None:
        with self.assertRaises(ValueError):
            determine_target_length(pd.Series(dtype=int), "max", 99)


class BuildCycleLengthStatisticsTests(unittest.TestCase):
    def test_reports_distribution_and_selected_target(self) -> None:
        lengths = pd.Series([10, 20, 30, 40, 50])
        stats = build_cycle_length_statistics(lengths, target_length=30, strategy="percentile", percentile=99)
        self.assertEqual(stats["minimum_cycle_length"], 10)
        self.assertEqual(stats["maximum_cycle_length"], 50)
        self.assertEqual(stats["mean_cycle_length"], 30.0)
        self.assertEqual(stats["median_cycle_length"], 30.0)
        self.assertEqual(stats["selected_target_length"], 30)
        self.assertEqual(stats["number_of_padded_cycles"], 2)
        self.assertEqual(stats["number_of_truncated_cycles"], 2)


class BuildCycleTensorTests(unittest.TestCase):
    def test_output_shape_matches_target_length_and_signal_count(self) -> None:
        measurements_df = _build_cycle_measurements(1, position_sample_count=6)
        matrix, mask, original_length, padded_samples, truncated_samples, signal_lengths = (
            build_cycle_tensor(
                measurements_df, DEFAULT_REQUIRED_SIGNALS, "position", target_length=10
            )
        )
        self.assertEqual(matrix.shape, (10, len(DEFAULT_REQUIRED_SIGNALS)))
        self.assertEqual(mask.shape, (10, len(DEFAULT_REQUIRED_SIGNALS)))
        self.assertEqual(original_length, 6)
        self.assertEqual(padded_samples, 4)
        self.assertEqual(truncated_samples, 0)
        self.assertEqual(signal_lengths["position"], 6)

    def test_signal_order_matches_required_signals_order(self) -> None:
        measurements_df = _build_cycle_measurements(1, position_sample_count=4)
        reordered_signals = ("temperature", "position", "pressure", "current", "velocity")
        matrix, mask, _, _, _, signal_lengths = build_cycle_tensor(
            measurements_df, reordered_signals, "position", target_length=6
        )
        # position's raw values are 0, 10, 20, 30, then edge-padded with 30.
        np.testing.assert_allclose(matrix[:, 1], [0.0, 10.0, 20.0, 30.0, 30.0, 30.0])
        # temperature is a single constant raw sample, edge-padded throughout.
        np.testing.assert_allclose(matrix[:, 0], np.full(6, 21.5))
        # Mask columns follow the same reordered_signals order as the matrix.
        np.testing.assert_array_equal(mask[:, 0], [1, 0, 0, 0, 0, 0])  # temperature: 1 real sample
        np.testing.assert_array_equal(mask[:, 1], [1, 1, 1, 1, 0, 0])  # position: 4 real samples
        self.assertEqual(
            signal_lengths,
            {"temperature": 1, "position": 4, "pressure": 1, "current": 3, "velocity": 2},
        )

    def test_preserves_original_measured_values_without_interpolation(self) -> None:
        measurements_df = _build_cycle_measurements(1, position_sample_count=5)
        matrix, _, _, _, _, _ = build_cycle_tensor(
            measurements_df, DEFAULT_REQUIRED_SIGNALS, "position", target_length=5
        )
        current_column = DEFAULT_REQUIRED_SIGNALS.index("current")
        # current has exactly 3 raw samples (0.1, 0.2, 0.3); edge-padded to length 5.
        np.testing.assert_allclose(matrix[:, current_column], [0.1, 0.2, 0.3, 0.3, 0.3])

    def test_truncates_longer_cycle_at_the_end(self) -> None:
        measurements_df = _build_cycle_measurements(1, position_sample_count=10)
        matrix, mask, original_length, padded_samples, truncated_samples, signal_lengths = (
            build_cycle_tensor(
                measurements_df, DEFAULT_REQUIRED_SIGNALS, "position", target_length=4
            )
        )
        position_column = DEFAULT_REQUIRED_SIGNALS.index("position")
        np.testing.assert_allclose(matrix[:, position_column], [0.0, 10.0, 20.0, 30.0])
        self.assertEqual(original_length, 10)
        self.assertEqual(truncated_samples, 6)
        self.assertEqual(padded_samples, 0)
        # position (10 raw samples, truncated) is all real within target_length.
        np.testing.assert_array_equal(mask[:, position_column], [1, 1, 1, 1])
        # velocity has 2 raw samples -> padded to 4.
        velocity_column = DEFAULT_REQUIRED_SIGNALS.index("velocity")
        np.testing.assert_array_equal(mask[:, velocity_column], [1, 1, 0, 0])
        # current has 3 raw samples -> padded to 4.
        current_column = DEFAULT_REQUIRED_SIGNALS.index("current")
        np.testing.assert_array_equal(mask[:, current_column], [1, 1, 1, 0])
        # pressure/temperature have 1 raw sample each -> padded to 4.
        pressure_column = DEFAULT_REQUIRED_SIGNALS.index("pressure")
        np.testing.assert_array_equal(mask[:, pressure_column], [1, 0, 0, 0])
        self.assertEqual(signal_lengths["position"], 10)
        self.assertEqual(signal_lengths["velocity"], 2)
        self.assertEqual(signal_lengths["current"], 3)
        self.assertEqual(signal_lengths["pressure"], 1)
        self.assertEqual(signal_lengths["temperature"], 1)

    def test_signal_specific_masks_with_different_original_lengths(self) -> None:
        # position=63, velocity=62, current=61, pressure=51, temperature=3
        # (mirrors the thesis example of independently padded signals).
        rows: list[dict] = []
        lengths = {"position": 63, "velocity": 62, "current": 61, "pressure": 51, "temperature": 3}
        for signal_name, sample_count in lengths.items():
            for index in range(sample_count):
                rows.append(
                    _measurement_row(
                        1,
                        signal_name,
                        CYCLE_START + pd.Timedelta(seconds=index),
                        float(index),
                    )
                )
        measurements_df = pd.DataFrame(rows)

        matrix, mask, original_length, padded_samples, truncated_samples, signal_lengths = (
            build_cycle_tensor(
                measurements_df, DEFAULT_REQUIRED_SIGNALS, "position", target_length=63
            )
        )

        self.assertEqual(matrix.shape, (63, 5))
        self.assertEqual(mask.shape, (63, 5))
        self.assertEqual(original_length, 63)
        self.assertEqual(padded_samples, 0)
        self.assertEqual(truncated_samples, 0)
        self.assertEqual(signal_lengths, lengths)

        expected_ones = {
            "position": 63,
            "velocity": 62,
            "current": 61,
            "pressure": 51,
            "temperature": 3,
        }
        for signal_name, expected in expected_ones.items():
            column_index = DEFAULT_REQUIRED_SIGNALS.index(signal_name)
            column_mask = mask[:, column_index]
            self.assertEqual(int(column_mask.sum()), expected)
            np.testing.assert_array_equal(column_mask[:expected], np.ones(expected, dtype=np.int8))
            np.testing.assert_array_equal(
                column_mask[expected:], np.zeros(63 - expected, dtype=np.int8)
            )

    def test_no_padding_case_mask_is_all_ones(self) -> None:
        # Every required signal has exactly target_length raw samples.
        rows: list[dict] = []
        for signal_name in DEFAULT_REQUIRED_SIGNALS:
            for index in range(5):
                rows.append(
                    _measurement_row(
                        1,
                        signal_name,
                        CYCLE_START + pd.Timedelta(seconds=index),
                        float(index),
                    )
                )
        measurements_df = pd.DataFrame(rows)

        matrix, mask, _, padded_samples, truncated_samples, signal_lengths = build_cycle_tensor(
            measurements_df, DEFAULT_REQUIRED_SIGNALS, "position", target_length=5
        )

        self.assertEqual(padded_samples, 0)
        self.assertEqual(truncated_samples, 0)
        np.testing.assert_array_equal(mask, np.ones((5, len(DEFAULT_REQUIRED_SIGNALS)), dtype=np.int8))
        for signal_name in DEFAULT_REQUIRED_SIGNALS:
            self.assertEqual(signal_lengths[signal_name], 5)

    def test_missing_required_signal_raises(self) -> None:
        measurements_df = _build_cycle_measurements(1, position_sample_count=5)
        measurements_df = measurements_df[measurements_df["signal_name"] != "pressure"]
        with self.assertRaises(MissingRequiredSignalError):
            build_cycle_tensor(measurements_df, DEFAULT_REQUIRED_SIGNALS, "position", target_length=5)

    def test_deterministic_across_repeated_calls(self) -> None:
        measurements_df = _build_cycle_measurements(1, position_sample_count=6)
        first = build_cycle_tensor(measurements_df, DEFAULT_REQUIRED_SIGNALS, "position", target_length=8)
        second = build_cycle_tensor(measurements_df, DEFAULT_REQUIRED_SIGNALS, "position", target_length=8)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[2:], second[2:])


class CycleTensorGenerationConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        config = CycleTensorGenerationConfig.from_mapping(None)
        self.assertEqual(config.required_signals, DEFAULT_REQUIRED_SIGNALS)
        self.assertEqual(config.reference_signal, "position")
        self.assertEqual(config.target_length_strategy, "percentile")
        self.assertEqual(config.target_length_percentile, 99)
        self.assertEqual(config.padding_method, "edge")
        self.assertTrue(config.truncate_long_cycles)
        self.assertTrue(config.save_padding_mask)
        self.assertEqual(config.output_format, "npy")
        self.assertEqual(config.cycles_per_file, 64)

    def test_rejects_unsupported_target_length_strategy(self) -> None:
        with self.assertRaises(ValueError):
            CycleTensorGenerationConfig.from_mapping({"target_length_strategy": "min"})

    def test_rejects_unsupported_padding_method(self) -> None:
        with self.assertRaises(ValueError):
            CycleTensorGenerationConfig.from_mapping({"padding_method": "zero"})

    def test_rejects_non_positive_cycles_per_file(self) -> None:
        with self.assertRaises(ValueError):
            CycleTensorGenerationConfig.from_mapping({"cycles_per_file": 0})

    def test_rejects_reference_signal_not_in_required_signals(self) -> None:
        with self.assertRaises(ValueError):
            CycleTensorGenerationConfig.from_mapping(
                {"required_signals": ["velocity"], "reference_signal": "position"}
            )


class GenerateCycleTensorDatasetTests(unittest.TestCase):
    def _build_fixture(
        self, temp_dir: Path, position_sample_counts: list[int]
    ) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
        cycle_rows = []
        measurement_frames = []
        for offset, sample_count in enumerate(position_sample_counts, start=1):
            cycle_id = offset
            start = CYCLE_START + pd.Timedelta(minutes=cycle_id)
            end = start + pd.Timedelta(seconds=max(sample_count, 1))
            cycle_rows.append(
                {
                    "cycle_id": cycle_id,
                    "session_id": 1,
                    "start_time": start,
                    "end_time": end,
                }
            )
            measurement_frames.append(
                _build_cycle_measurements(cycle_id, sample_count, start=start)
            )

        cycle_index_df = pd.DataFrame(cycle_rows)
        valid_core_cycles_df = pd.DataFrame({"cycle_id": [row["cycle_id"] for row in cycle_rows]})
        measurement_dataset_path = _write_measurement_dataset(
            temp_dir, pd.concat(measurement_frames, ignore_index=True)
        )
        return cycle_index_df, valid_core_cycles_df, measurement_dataset_path

    def test_batch_tensor_and_mask_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            # 5 cycles with lengths 4, 6, 8, 10, 20 -> max=20.
            cycle_index_df, valid_core_cycles_df, measurement_dataset_path = self._build_fixture(
                temp_dir, [4, 6, 8, 10, 20]
            )
            config = CycleTensorGenerationConfig.from_mapping(
                {"target_length_strategy": "max", "cycles_per_file": 3}
            )
            output_directory = temp_dir / "feature_engineering"

            result = generate_cycle_tensor_dataset(
                valid_core_cycles_df,
                cycle_index_df,
                measurement_dataset_path,
                config,
                output_directory,
                dataset_name="D63_Nr7",
                experiment="Versuch1",
            )

            target_length = result.summary["target_length"]
            self.assertEqual(target_length, 20)
            self.assertEqual(len(result.written_files), 2)
            self.assertEqual(len(result.mask_files), 2)

            number_of_signals = len(DEFAULT_REQUIRED_SIGNALS)
            first_batch = np.load(result.written_files[0])
            self.assertEqual(first_batch.shape, (3, target_length, number_of_signals))
            first_mask_batch = np.load(result.mask_files[0])
            self.assertEqual(first_mask_batch.shape, (3, target_length, number_of_signals))
            # Cycle 1 has 4 real position samples -> position column has exactly 4 ones.
            position_column = DEFAULT_REQUIRED_SIGNALS.index("position")
            self.assertEqual(int(first_mask_batch[0, :, position_column].sum()), 4)
            # velocity/current/pressure/temperature keep their own fixed sample counts
            # (2, 3, 1, 1) regardless of the reference (position) signal's length.
            velocity_column = DEFAULT_REQUIRED_SIGNALS.index("velocity")
            current_column = DEFAULT_REQUIRED_SIGNALS.index("current")
            pressure_column = DEFAULT_REQUIRED_SIGNALS.index("pressure")
            temperature_column = DEFAULT_REQUIRED_SIGNALS.index("temperature")
            self.assertEqual(int(first_mask_batch[0, :, velocity_column].sum()), 2)
            self.assertEqual(int(first_mask_batch[0, :, current_column].sum()), 3)
            self.assertEqual(int(first_mask_batch[0, :, pressure_column].sum()), 1)
            self.assertEqual(int(first_mask_batch[0, :, temperature_column].sum()), 1)

            self.assertEqual(len(result.metadata), 5)
            for column in (
                "cycle_id",
                "session_id",
                "cycle_start",
                "cycle_end",
                "cycle_duration_seconds",
                "original_cycle_length",
                "target_length",
                "padded_samples",
                "truncated_samples",
                "signal_original_lengths",
                "batch_file",
            ):
                self.assertIn(column, result.metadata.columns)

    def test_batch_file_and_mask_naming_convention(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            cycle_index_df, valid_core_cycles_df, measurement_dataset_path = self._build_fixture(
                temp_dir, [4, 5, 6, 7, 8]
            )
            config = CycleTensorGenerationConfig.from_mapping({"cycles_per_file": 64})
            output_directory = temp_dir / "feature_engineering"

            result = generate_cycle_tensor_dataset(
                valid_core_cycles_df,
                cycle_index_df,
                measurement_dataset_path,
                config,
                output_directory,
                dataset_name="D63_Nr7",
                experiment="Versuch1",
            )

            self.assertEqual(len(result.written_files), 1)
            self.assertEqual(
                result.written_files[0].name,
                "D63_Nr7_Versuch1_cycles_000001_000005.npy",
            )
            self.assertEqual(
                result.mask_files[0].name,
                "D63_Nr7_Versuch1_cycles_000001_000005_mask.npy",
            )

    def test_truncation_is_recorded_in_metadata_and_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            # Cycles with lengths 4, 4, 4, 4, 100 -> 99th percentile stays small,
            # forcing the longest cycle to be truncated.
            cycle_index_df, valid_core_cycles_df, measurement_dataset_path = self._build_fixture(
                temp_dir, [4, 4, 4, 4, 100]
            )
            config = CycleTensorGenerationConfig.from_mapping(
                {"target_length_strategy": "percentile", "target_length_percentile": 50}
            )
            output_directory = temp_dir / "feature_engineering"

            result = generate_cycle_tensor_dataset(
                valid_core_cycles_df,
                cycle_index_df,
                measurement_dataset_path,
                config,
                output_directory,
            )

            target_length = result.summary["target_length"]
            self.assertLess(target_length, 100)
            truncated_row = result.metadata[result.metadata["cycle_id"] == 5].iloc[0]
            self.assertEqual(truncated_row["original_cycle_length"], 100)
            self.assertGreater(truncated_row["truncated_samples"], 0)
            self.assertEqual(
                truncated_row["truncated_samples"], 100 - target_length
            )
            self.assertGreaterEqual(result.length_statistics["number_of_truncated_cycles"], 1)

    def test_reads_only_cycle_id_column_from_valid_core_cycles_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            cycle_index_df, valid_core_cycles_df, measurement_dataset_path = self._build_fixture(
                temp_dir, [5, 6]
            )
            valid_core_cycles_df = valid_core_cycles_df.assign(
                final_class="valid_core_cycle", reasons=""
            )
            valid_core_cycles_path = temp_dir / "valid_core_cycles.parquet"
            valid_core_cycles_df.to_parquet(valid_core_cycles_path, index=False)

            config = CycleTensorGenerationConfig.from_mapping({})
            output_directory = temp_dir / "feature_engineering"

            result = generate_cycle_tensor_dataset(
                valid_core_cycles_path,
                cycle_index_df,
                measurement_dataset_path,
                config,
                output_directory,
            )

            self.assertEqual(len(result.metadata), 2)

    def test_missing_required_signal_is_skipped_not_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            cycle_index_df, valid_core_cycles_df, measurement_dataset_path = self._build_fixture(
                temp_dir, [5, 6]
            )
            measurements_df = pd.read_parquet(measurement_dataset_path)
            measurements_df = measurements_df[
                ~((measurements_df["cycle_id"] == 1) & (measurements_df["signal_name"] == "pressure"))
            ]
            for existing_file in measurement_dataset_path.glob("*.parquet"):
                existing_file.unlink()
            measurements_df.to_parquet(measurement_dataset_path / "part-0.parquet", index=False)

            config = CycleTensorGenerationConfig.from_mapping({})
            output_directory = temp_dir / "feature_engineering"

            result = generate_cycle_tensor_dataset(
                valid_core_cycles_df,
                cycle_index_df,
                measurement_dataset_path,
                config,
                output_directory,
            )

            self.assertEqual(len(result.metadata), 1)
            self.assertEqual(result.metadata.iloc[0]["cycle_id"], 2)
            self.assertEqual(len(result.skipped_cycles), 1)
            self.assertEqual(result.skipped_cycles.iloc[0]["cycle_id"], 1)

    def test_no_padding_mask_written_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            cycle_index_df, valid_core_cycles_df, measurement_dataset_path = self._build_fixture(
                temp_dir, [4, 5]
            )
            config = CycleTensorGenerationConfig.from_mapping({"save_padding_mask": False})
            output_directory = temp_dir / "feature_engineering"

            result = generate_cycle_tensor_dataset(
                valid_core_cycles_df,
                cycle_index_df,
                measurement_dataset_path,
                config,
                output_directory,
            )

            self.assertEqual(len(result.written_files), 1)
            self.assertEqual(len(result.mask_files), 0)

    def test_deterministic_results_across_repeated_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            cycle_index_df, valid_core_cycles_df, measurement_dataset_path = self._build_fixture(
                temp_dir, [4, 6, 8, 10]
            )
            config = CycleTensorGenerationConfig.from_mapping({"cycles_per_file": 2})

            first_output_directory = temp_dir / "run_one"
            first_result = generate_cycle_tensor_dataset(
                valid_core_cycles_df,
                cycle_index_df,
                measurement_dataset_path,
                config,
                first_output_directory,
                dataset_name="D63_Nr7",
                experiment="Versuch1",
            )
            second_output_directory = temp_dir / "run_two"
            second_result = generate_cycle_tensor_dataset(
                valid_core_cycles_df,
                cycle_index_df,
                measurement_dataset_path,
                config,
                second_output_directory,
                dataset_name="D63_Nr7",
                experiment="Versuch1",
            )

            self.assertEqual(
                [path.name for path in first_result.written_files],
                [path.name for path in second_result.written_files],
            )
            for first_path, second_path in zip(
                first_result.written_files, second_result.written_files
            ):
                first_batch = np.load(first_output_directory / first_path.name)
                second_batch = np.load(second_output_directory / second_path.name)
                np.testing.assert_array_equal(first_batch, second_batch)
            for first_path, second_path in zip(
                first_result.mask_files, second_result.mask_files
            ):
                first_mask = np.load(first_output_directory / first_path.name)
                second_mask = np.load(second_output_directory / second_path.name)
                np.testing.assert_array_equal(first_mask, second_mask)
            pd.testing.assert_frame_equal(
                first_result.metadata.drop(columns=["batch_file"]),
                second_result.metadata.drop(columns=["batch_file"]),
            )


if __name__ == "__main__":
    unittest.main()
