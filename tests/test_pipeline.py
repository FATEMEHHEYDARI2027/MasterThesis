"""Behavior tests for the reusable preprocessing pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.pipeline import (
    IMPLEMENTED_STAGES,
    STAGE_ORDER,
    PipelineConfig,
    PipelineStage,
    _build_validation_subset,
    _plot_cycle_validation,
    _run_cycle_detection_stage,
    _run_cycle_quality_profiling_stage,
    _run_multi_sensor_extraction_stage,
    build_int_signal_info_from_metadata,
    build_uuid_signal_info_from_metadata,
    load_metadata,
    run_pipeline,
)
from src.storage.cycle_index_writer import write_cycle_index
from src.visualization.multi_sensor_cycle_plot import plot_multi_sensor_cycle


def _write_parquet_table(path: Path, frame: pd.DataFrame) -> None:
    """Write a small pandas frame to one parquet file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)


def _write_uuid_measurements(dataset_dir: Path, frame: pd.DataFrame) -> None:
    """Write UUID measurements using the project's hive-partitioned layout."""

    pq.write_to_dataset(
        pa.Table.from_pandas(frame, preserve_index=False),
        root_path=dataset_dir / "signal_data_point.parquet",
        partition_cols=["signal_id"],
    )


def _write_int_measurements(dataset_dir: Path, frame: pd.DataFrame) -> None:
    """Write INT measurements using the project's per-partition directory layout."""

    for signal_id, signal_frame in frame.groupby("signal_id", sort=False):
        output_path = (
            dataset_dir
            / "vibration.parquet"
            / f"signal_id={int(signal_id)}"
            / "part-0.parquet"
        )
        _write_parquet_table(output_path, signal_frame.reset_index(drop=True))


def create_dataset_fixture(
    root_dir: Path,
    dataset_name: str = "FixtureDataset",
    experiment_name: str = "ExperimentAlpha",
) -> Path:
    """Create a tiny ERA-shaped dataset for pipeline tests."""

    dataset_dir = root_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    experiment_node_id = str(uuid4())
    drive_node_id = str(uuid4())
    temperature_node_id = str(uuid4())
    vibration_node_id = str(uuid4())
    position_signal_uuid = str(uuid4())
    temperature_signal_uuid = str(uuid4())
    vibration_signal_id = 101

    _write_parquet_table(
        dataset_dir / "nodes.parquet",
        pd.DataFrame(
            [
                {"node_id": experiment_node_id, "name": experiment_name, "parent_node": pd.NA},
                {"node_id": drive_node_id, "name": "Drive", "parent_node": experiment_node_id},
                {
                    "node_id": temperature_node_id,
                    "name": "Temperature",
                    "parent_node": experiment_node_id,
                },
                {"node_id": vibration_node_id, "name": "Sensor A", "parent_node": experiment_node_id},
            ]
        ),
    )
    _write_parquet_table(
        dataset_dir / "units.parquet",
        pd.DataFrame(
            [
                {"common_code": "position", "symbol": "mm", "name": "Position"},
                {"common_code": "temperature", "symbol": "C", "name": "Temperature"},
                {"common_code": "vibration", "symbol": "m/s2", "name": "Vibration"},
            ]
        ),
    )
    _write_parquet_table(
        dataset_dir / "signal_data_point_rel.parquet",
        pd.DataFrame(
            [
                {
                    "signal_id": position_signal_uuid,
                    "node_id": drive_node_id,
                    "unit": "position",
                },
                {
                    "signal_id": temperature_signal_uuid,
                    "node_id": temperature_node_id,
                    "unit": "temperature",
                },
            ]
        ),
    )
    _write_parquet_table(
        dataset_dir / "signal_data_point_rel_int.parquet",
        pd.DataFrame(
            [
                {
                    "signal_id": vibration_signal_id,
                    "node_id": vibration_node_id,
                    "unit": "vibration",
                }
            ]
        ),
    )

    uuid_measurements = pd.DataFrame(
        [
            {"signal_id": position_signal_uuid, "time": "2026-01-01 00:00:00", "value": 0.0},
            {"signal_id": position_signal_uuid, "time": "2026-01-01 00:00:00.500000", "value": 0.0},
            {"signal_id": position_signal_uuid, "time": "2026-01-01 00:00:01", "value": 2.0},
            {"signal_id": position_signal_uuid, "time": "2026-01-01 00:00:01.500000", "value": 2.0},
            {"signal_id": position_signal_uuid, "time": "2026-01-01 00:00:02", "value": 0.0},
            {"signal_id": position_signal_uuid, "time": "2026-01-01 00:00:02.500000", "value": 0.0},
            {"signal_id": position_signal_uuid, "time": "2026-01-01 02:00:00", "value": 0.0},
            {"signal_id": position_signal_uuid, "time": "2026-01-01 02:00:00.500000", "value": 3.0},
            {"signal_id": position_signal_uuid, "time": "2026-01-01 02:00:01", "value": 0.0},
            {"signal_id": temperature_signal_uuid, "time": "2026-01-01 00:00:00", "value": 20.0},
            {"signal_id": temperature_signal_uuid, "time": "2026-01-01 00:00:01", "value": 20.5},
            {"signal_id": temperature_signal_uuid, "time": "2026-01-01 02:00:00", "value": 21.0},
        ]
    )
    uuid_measurements["time"] = pd.to_datetime(uuid_measurements["time"], format="mixed")
    _write_uuid_measurements(dataset_dir, uuid_measurements)

    int_measurements = pd.DataFrame(
        [
            {"signal_id": vibration_signal_id, "time": "2026-01-01 00:00:01", "value": 0.1},
            {"signal_id": vibration_signal_id, "time": "2026-01-01 00:00:01.500000", "value": 0.2},
            {"signal_id": vibration_signal_id, "time": "2026-01-01 02:00:00.500000", "value": 0.3},
        ]
    )
    int_measurements["time"] = pd.to_datetime(int_measurements["time"], format="mixed")
    _write_int_measurements(dataset_dir, int_measurements)

    return dataset_dir


class PipelineTests(unittest.TestCase):
    """End-to-end tests for the pipeline public interface."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)
        self.dataset_dir = create_dataset_fixture(self.root_path)
        self.output_root = self.root_path / "outputs"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _config(self, **overrides: object) -> PipelineConfig:
        values = {
            "dataset_path": self.dataset_dir,
            "experiment": "ExperimentAlpha",
            "stop_after": "cycle_detection",
            "output_root": self.output_root,
            "session_gap_seconds": 3600.0,
            "movement_threshold": 1.0,
        }
        values.update(overrides)
        return PipelineConfig(**values)

    def _stage_keys(self, result: dict[str, object]) -> list[str]:
        return [key for key in result if key != "run"]

    def test_invalid_dataset_path_raises_file_not_found(self) -> None:
        config = self._config(dataset_path=self.root_path / "missing-dataset")

        with self.assertRaisesRegex(FileNotFoundError, "Dataset path does not exist"):
            run_pipeline(config)

    def test_invalid_experiment_raises_clear_error(self) -> None:
        config = self._config(experiment="MissingExperiment")

        with self.assertRaisesRegex(ValueError, "Experiment not found in metadata"):
            run_pipeline(config)

    def test_invalid_stop_point_raises_clear_error(self) -> None:
        config = self._config(stop_after="not_a_stage")

        with self.assertRaisesRegex(ValueError, "Invalid stop_after stage"):
            run_pipeline(config)

    def test_stops_after_metadata(self) -> None:
        result = run_pipeline(self._config(stop_after="metadata"))

        self.assertEqual(self._stage_keys(result), ["metadata"])
        metadata = result["metadata"]
        self.assertIn("uuid_signal_info", metadata)
        self.assertIn("int_signal_info", metadata)

    def test_stops_after_signal_discovery(self) -> None:
        result = run_pipeline(self._config(stop_after="signal_discovery"))

        self.assertEqual(self._stage_keys(result), ["metadata", "signal_discovery"])
        discovery = result["signal_discovery"]
        self.assertEqual(discovery["reference_signal_uuid"], discovery["reference_signals"].iloc[0]["signal_id_uuid"])

    def test_stages_execute_in_fixed_order(self) -> None:
        result = run_pipeline(self._config(stop_after="cycle_detection"))

        self.assertEqual(
            self._stage_keys(result),
            [
                "metadata",
                "signal_discovery",
                "timestamp_analysis",
                "session_detection",
                "cycle_detection",
            ],
        )

    def test_later_stages_do_not_run_after_stop_point(self) -> None:
        result = run_pipeline(self._config(stop_after="signal_discovery"))

        self.assertNotIn("timestamp_analysis", result)
        run_dir = Path(result["run"]["run_directory"])
        self.assertFalse((run_dir / "timestamp_analysis" / "statistics.csv").exists())
        self.assertFalse((run_dir / "sessions" / "sessions.csv").exists())

    def test_returns_outputs_for_all_completed_stages(self) -> None:
        result = run_pipeline(self._config(stop_after="cycle_detection"))

        for stage_name in self._stage_keys(result):
            stage_result = result[stage_name]
            self.assertIn("output_paths", stage_result)
            self.assertTrue(stage_result["output_paths"])

    def test_metadata_are_loaded_only_once(self) -> None:
        import src.pipeline as pipeline_module

        load_calls = 0
        original = pipeline_module.load_metadata

        def counting_loader(*args: object, **kwargs: object) -> object:
            nonlocal load_calls
            load_calls += 1
            return original(*args, **kwargs)

        with mock.patch.object(pipeline_module, "load_metadata", side_effect=counting_loader):
            run_pipeline(self._config(stop_after="signal_discovery"))

        self.assertEqual(load_calls, 1)

    def test_run_manifest_is_created(self) -> None:
        result = run_pipeline(self._config(stop_after="metadata"))

        manifest_path = Path(result["run"]["manifest_path"])
        self.assertTrue(manifest_path.exists())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["dataset_name"], "FixtureDataset")
        self.assertEqual(manifest["experiment"], "ExperimentAlpha")
        self.assertEqual(manifest["completed_stages"], ["metadata"])

    def test_dataset_specific_values_are_not_hard_coded(self) -> None:
        custom_dataset = create_dataset_fixture(
            self.root_path,
            dataset_name="AnotherDataset",
            experiment_name="Run42",
        )
        result = run_pipeline(
            PipelineConfig(
                dataset_path=custom_dataset,
                experiment="Run42",
                stop_after="signal_discovery",
                output_root=self.output_root,
            )
        )

        self.assertEqual(result["run"]["dataset_name"], "AnotherDataset")
        self.assertEqual(result["run"]["experiment"], "Run42")
        self.assertEqual(len(result["signal_discovery"]["selected_signals"]), 3)


class PlotCycleValidationTests(unittest.TestCase):
    """Behavior tests for the interactive Plotly cycle validation output."""

    def _position_df(self, num_rows: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01", periods=num_rows, freq="s"),
                "value": [float(index % 5) for index in range(num_rows)],
            }
        )

    def _cycles_df(self, position_df: pd.DataFrame, num_cycles: int) -> pd.DataFrame:
        step = max(1, len(position_df) // max(num_cycles, 1))
        rows = []
        for cycle_id in range(1, num_cycles + 1):
            start_index = min((cycle_id - 1) * step, len(position_df) - 2)
            end_index = min(start_index + 1, len(position_df) - 1)
            rows.append(
                {
                    "cycle_id": cycle_id,
                    "session_id": 1,
                    "start_time": position_df["time"].iloc[start_index],
                    "end_time": position_df["time"].iloc[end_index],
                }
            )
        return pd.DataFrame(rows)

    def test_empty_position_df_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "cycle_validation.html"
            result = _plot_cycle_validation(
                pd.DataFrame(columns=["time", "value"]),
                pd.DataFrame(columns=["cycle_id", "start_time", "end_time"]),
                output_path,
                experiment="ExperimentAlpha",
                movement_threshold=1.0,
            )

        self.assertIsNone(result)

    def test_html_file_is_created_with_plotly_content(self) -> None:
        position_df = self._position_df(100)
        cycles_df = self._cycles_df(position_df, 5)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "cycle_validation.html"
            result = _plot_cycle_validation(
                position_df,
                cycles_df,
                output_path,
                experiment="ExperimentAlpha",
                movement_threshold=1.0,
            )

            self.assertIsNotNone(result)
            self.assertTrue(str(result).endswith(".html"))
            self.assertTrue(output_path.exists())
            html_content = output_path.read_text(encoding="utf-8")
            self.assertIn("plotly", html_content.lower())

    def test_large_input_is_downsampled_for_plotting(self) -> None:
        num_rows = 250_000
        num_cycles = 25_000
        position_df = self._position_df(num_rows)
        cycles_df = self._cycles_df(position_df, num_cycles)
        original_position_rows = len(position_df)
        original_cycle_rows = len(cycles_df)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "cycle_validation.html"
            result = _plot_cycle_validation(
                position_df,
                cycles_df,
                output_path,
                experiment="ExperimentAlpha",
                movement_threshold=1.0,
            )

            self.assertIsNotNone(result)
            self.assertTrue(output_path.exists())

        # The original DataFrames must remain untouched.
        self.assertEqual(len(position_df), original_position_rows)
        self.assertEqual(len(cycles_df), original_cycle_rows)

    def test_output_paths_validation_plot_points_to_html(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)
            dataset_dir = create_dataset_fixture(root_path)
            output_root = root_path / "outputs"
            result = run_pipeline(
                PipelineConfig(
                    dataset_path=dataset_dir,
                    experiment="ExperimentAlpha",
                    stop_after="cycle_detection",
                    output_root=output_root,
                    session_gap_seconds=3600.0,
                    movement_threshold=1.0,
                )
            )

            output_paths = result["cycle_detection"]["output_paths"]
            if "validation_plot" in output_paths:
                self.assertTrue(output_paths["validation_plot"].endswith(".html"))
                self.assertTrue(Path(output_paths["validation_plot"]).exists())


class BuildValidationSubsetTests(unittest.TestCase):
    """Behavior tests for restricting validation data to a small subset."""

    def _position_df(self, num_rows: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01", periods=num_rows, freq="s"),
                "value": [float(index % 5) for index in range(num_rows)],
            }
        )

    def _cycles_df(self, position_df: pd.DataFrame, num_cycles: int) -> pd.DataFrame:
        step = max(1, len(position_df) // max(num_cycles, 1))
        rows = []
        for cycle_id in range(1, num_cycles + 1):
            start_index = min((cycle_id - 1) * step, len(position_df) - 2)
            end_index = min(start_index + 1, len(position_df) - 1)
            rows.append(
                {
                    "cycle_id": cycle_id,
                    "session_id": 1,
                    "start_time": position_df["time"].iloc[start_index],
                    "end_time": position_df["time"].iloc[end_index],
                }
            )
        return pd.DataFrame(rows)

    def test_at_most_100_cycles_are_selected(self) -> None:
        position_df = self._position_df(10_000)
        cycles_df = self._cycles_df(position_df, 250)

        validation_position_df, validation_cycles_df = _build_validation_subset(
            position_df, cycles_df
        )

        self.assertLessEqual(len(validation_cycles_df), 100)
        self.assertTrue(validation_cycles_df["cycle_id"].equals(cycles_df["cycle_id"].iloc[:100]))
        self.assertFalse(validation_position_df.empty)

    def test_position_window_is_restricted_to_selected_cycle_range(self) -> None:
        position_df = self._position_df(10_000)
        cycles_df = self._cycles_df(position_df, 250)

        validation_position_df, validation_cycles_df = _build_validation_subset(
            position_df, cycles_df
        )

        expected_start = pd.Timestamp(
            validation_cycles_df["start_time"].iloc[0]
        ) - pd.Timedelta(seconds=1)
        expected_end = pd.Timestamp(
            validation_cycles_df["end_time"].iloc[-1]
        ) + pd.Timedelta(seconds=1)

        self.assertGreaterEqual(validation_position_df["time"].min(), expected_start)
        self.assertLessEqual(validation_position_df["time"].max(), expected_end)
        # The window must be far smaller than the full session.
        self.assertLess(len(validation_position_df), len(position_df))

    def test_empty_session_cycles_returns_empty_subset(self) -> None:
        position_df = self._position_df(10)
        empty_cycles_df = pd.DataFrame(columns=["cycle_id", "start_time", "end_time"])

        validation_position_df, validation_cycles_df = _build_validation_subset(
            position_df, empty_cycles_df
        )

        self.assertTrue(validation_position_df.empty)
        self.assertTrue(validation_cycles_df.empty)


class RunCycleDetectionValidationSubsetTests(unittest.TestCase):
    """Tests for the validation-subset wiring inside the cycle-detection stage."""

    def _position_df(self, num_rows: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01", periods=num_rows, freq="s"),
                "value": [2.0 if index % 2 == 0 else 0.0 for index in range(num_rows)],
                "signal_id": ["signal-uuid"] * num_rows,
            }
        )

    def test_validation_plot_receives_at_most_100_cycles_full_csv_unchanged(self) -> None:
        position_df = self._position_df(2_000)
        sessions_df = pd.DataFrame(
            [
                {
                    "session_id": 1,
                    "start_time": position_df["time"].iloc[0],
                    "end_time": position_df["time"].iloc[-1],
                }
            ]
        )

        captured: dict[str, object] = {}

        def fake_plot_cycle_validation(
            validation_position_df, validation_cycles_df, output_path, **kwargs
        ):
            captured["position_len"] = len(validation_position_df)
            captured["cycles_len"] = len(validation_cycles_df)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("<html>stub</html>", encoding="utf-8")
            return output_path

        with tempfile.TemporaryDirectory() as temp_dir:
            stage_directory = Path(temp_dir) / "cycle_detection"
            with mock.patch(
                "src.pipeline._load_position_window", return_value=position_df
            ), mock.patch(
                "src.pipeline._plot_cycle_validation",
                side_effect=fake_plot_cycle_validation,
            ):
                result = _run_cycle_detection_stage(
                    dataset_path=Path(temp_dir),
                    stage_directory=stage_directory,
                    experiment="ExperimentAlpha",
                    reference_signal_uuid="signal-uuid",
                    sessions_df=sessions_df,
                    movement_threshold=1.0,
                )

            cycles_df = result["cycles"]
            self.assertGreater(len(cycles_df), 100)
            self.assertLessEqual(captured["cycles_len"], 100)
            self.assertLess(captured["position_len"], len(position_df))

            cycles_csv_path = Path(result["output_paths"]["cycles_csv"])
            saved_cycles_df = pd.read_csv(cycles_csv_path)
            self.assertEqual(len(saved_cycles_df), len(cycles_df))


class PlotMultiSensorCycleTests(unittest.TestCase):
    """Behavior tests for the interactive multi-sensor cycle visualization."""

    def _signal_df(self, num_rows: int, start_value: float = 0.0) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01", periods=num_rows, freq="ms"),
                "value": [start_value + float(index) for index in range(num_rows)],
            }
        )

    def _signal_descriptors(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"signal_name": "position", "unit_symbol": "mm"},
                {"signal_name": "temperature", "unit_symbol": "C"},
                {"signal_name": "vibration_x", "unit_symbol": "m/s2"},
                {"signal_name": "vibration_y", "unit_symbol": "m/s2"},
            ]
        )

    def test_empty_extracted_signals_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "multi_sensor_cycle.html"
            result = plot_multi_sensor_cycle(
                {},
                self._signal_descriptors(),
                output_path,
                experiment="ExperimentAlpha",
                cycle_id=1,
                session_id=1,
            )

        self.assertIsNone(result)
        self.assertFalse(output_path.exists())

    def test_html_file_is_created_with_plotly_content_and_subplot_titles(self) -> None:
        extracted_signals = {
            "position": self._signal_df(50),
            "temperature": self._signal_df(20, start_value=20.0),
            "vibration_x": self._signal_df(30, start_value=-1.0),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "multi_sensor_cycle.html"
            result = plot_multi_sensor_cycle(
                extracted_signals,
                self._signal_descriptors(),
                output_path,
                experiment="ExperimentAlpha",
                cycle_id=7,
                session_id=2,
            )

            self.assertEqual(result, output_path)
            self.assertTrue(output_path.exists())
            html_content = output_path.read_text(encoding="utf-8")
            self.assertIn("plotly", html_content.lower())
            for signal_name in extracted_signals:
                self.assertIn(signal_name, html_content)

    def test_empty_signal_is_represented_without_failure(self) -> None:
        extracted_signals = {
            "position": self._signal_df(10),
            "temperature": pd.DataFrame(columns=["time", "value"]),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "multi_sensor_cycle.html"
            result = plot_multi_sensor_cycle(
                extracted_signals,
                self._signal_descriptors(),
                output_path,
                experiment="ExperimentAlpha",
                cycle_id=3,
                session_id=None,
            )

            self.assertEqual(result, output_path)
            html_content = output_path.read_text(encoding="utf-8")
            self.assertIn("No samples available", html_content)

    def test_large_signal_is_downsampled_without_mutating_original(self) -> None:
        large_signal_df = self._signal_df(50_000)
        original_rows = len(large_signal_df)
        extracted_signals = {"position": large_signal_df}

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "multi_sensor_cycle.html"
            result = plot_multi_sensor_cycle(
                extracted_signals,
                self._signal_descriptors(),
                output_path,
                experiment="ExperimentAlpha",
                cycle_id=4,
                session_id=1,
            )

            self.assertEqual(result, output_path)
            self.assertTrue(output_path.exists())

        self.assertEqual(len(large_signal_df), original_rows)
        self.assertEqual(len(extracted_signals["position"]), original_rows)


class RunMultiSensorExtractionStageTests(unittest.TestCase):
    """Tests for the scalable, batch-oriented, Parquet-based extraction stage."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)
        self.dataset_dir = create_dataset_fixture(self.root_path)

        metadata_frames = load_metadata(self.dataset_dir)
        self.uuid_signal_info = build_uuid_signal_info_from_metadata(metadata_frames)
        self.int_signal_info = build_int_signal_info_from_metadata(metadata_frames)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _synthetic_cycles_df(self) -> pd.DataFrame:
        # Three cycles inside session 1 (00:00:00 - 00:00:02.5) and two
        # cycles inside session 2 (02:00:00 - 02:00:01) of the fixture
        # dataset, so extraction spans two sessions and multiple cycles.
        rows = [
            {
                "cycle_id": 1,
                "session_id": 1,
                "start_time": pd.Timestamp("2026-01-01 00:00:00"),
                "end_time": pd.Timestamp("2026-01-01 00:00:01"),
            },
            {
                "cycle_id": 2,
                "session_id": 1,
                "start_time": pd.Timestamp("2026-01-01 00:00:01"),
                "end_time": pd.Timestamp("2026-01-01 00:00:02"),
            },
            {
                "cycle_id": 3,
                "session_id": 1,
                "start_time": pd.Timestamp("2026-01-01 00:00:02"),
                "end_time": pd.Timestamp("2026-01-01 00:00:02.5"),
            },
            {
                "cycle_id": 4,
                "session_id": 2,
                "start_time": pd.Timestamp("2026-01-01 02:00:00"),
                "end_time": pd.Timestamp("2026-01-01 02:00:00.5"),
            },
            {
                "cycle_id": 5,
                "session_id": 2,
                "start_time": pd.Timestamp("2026-01-01 02:00:00.5"),
                "end_time": pd.Timestamp("2026-01-01 02:00:01"),
            },
        ]
        cycles_df = pd.DataFrame(rows)
        cycles_df["experiment"] = "ExperimentAlpha"
        return cycles_df

    def _run_stage(self, cycles_df: pd.DataFrame, stage_directory: Path, **overrides: object) -> dict[str, object]:
        cycle_index_path = write_cycle_index(cycles_df, stage_directory.parent / "cycles" / "cycles.parquet")
        values: dict[str, object] = {
            "dataset_path": self.dataset_dir,
            "stage_directory": stage_directory,
            "uuid_signal_info": self.uuid_signal_info,
            "int_signal_info": self.int_signal_info,
            "experiment": "ExperimentAlpha",
            "cycles_df": cycles_df,
            "cycle_index_path": cycle_index_path,
            "max_cycles_to_extract": 100,
            "extract_all_cycles": False,
            "cycle_batch_size": 4,
            "resume_extraction": True,
            "overwrite_existing": False,
            "selected_extraction_signals": (),
            "validation_cycle_count": 2,
            "required_validation_signals": ("position", "temperature"),
            "minimum_samples_per_validation_cycle": {"position": 1, "temperature": 1},
            "require_consecutive_validation_cycles": True,
            "max_cycles_to_scan_for_validation": 10_000,
            "generate_validation_html": True,
            "generate_cycle_features": False,
            "parquet_compression": "snappy",
        }
        values.update(overrides)
        return _run_multi_sensor_extraction_stage(**values)

    def test_only_validation_cycles_receive_html(self) -> None:
        cycles_df = self._synthetic_cycles_df()
        with tempfile.TemporaryDirectory() as temp_dir:
            stage_directory = Path(temp_dir) / "multi_sensor"
            result = self._run_stage(cycles_df, stage_directory)

            output_paths = result["output_paths"]
            self.assertIn("validation_html_files", output_paths)
            html_paths = output_paths["validation_html_files"]
            # validation_cycle_count=2, so at most 2 HTML files, never one per
            # detected cycle.
            self.assertLessEqual(len(html_paths), 2)
            self.assertGreater(len(html_paths), 0)
            for html_path in html_paths:
                self.assertTrue(Path(html_path).exists())
                self.assertTrue(html_path.endswith(".html"))

            self.assertEqual(result["row_counts"]["selected_validation_cycles"], len(html_paths))

    def test_batches_span_multiple_cycles_partitioned_by_experiment_and_session(self) -> None:
        cycles_df = self._synthetic_cycles_df()
        with tempfile.TemporaryDirectory() as temp_dir:
            stage_directory = Path(temp_dir) / "multi_sensor"
            result = self._run_stage(cycles_df, stage_directory, extract_all_cycles=True)

            batch_files = result["output_paths"]["measurement_batch_files"]
            self.assertGreater(len(batch_files), 0)
            for batch_file in batch_files:
                batch_path = Path(batch_file)
                self.assertTrue(batch_path.exists())
                self.assertIn("experiment=ExperimentAlpha", str(batch_path))
                self.assertIn("session_id=", str(batch_path))
                # Never partitioned by cycle_id.
                self.assertNotIn("cycle_id=", str(batch_path))

                batch_frame = pd.read_parquet(batch_path)
                # At least one batch contains more than one cycle.
                if batch_frame["cycle_id"].nunique() > 1:
                    break
            else:
                self.fail("Expected at least one batch file containing multiple cycles.")

        self.assertEqual(result["row_counts"]["detected_cycles"], 5)
        self.assertEqual(result["row_counts"]["processed_cycles"], 5)

    def test_full_mode_does_not_truncate_cycles_with_head(self) -> None:
        cycles_df = self._synthetic_cycles_df()
        with tempfile.TemporaryDirectory() as temp_dir:
            stage_directory = Path(temp_dir) / "multi_sensor"
            result = self._run_stage(
                cycles_df,
                stage_directory,
                extract_all_cycles=True,
                max_cycles_to_extract=1,
            )

        # extract_all_cycles=True must ignore max_cycles_to_extract entirely.
        self.assertEqual(result["row_counts"]["processed_cycles"], 5)

    def test_limited_mode_respects_max_cycles_to_extract(self) -> None:
        cycles_df = self._synthetic_cycles_df()
        with tempfile.TemporaryDirectory() as temp_dir:
            stage_directory = Path(temp_dir) / "multi_sensor"
            result = self._run_stage(
                cycles_df,
                stage_directory,
                extract_all_cycles=False,
                max_cycles_to_extract=2,
            )

        self.assertEqual(result["row_counts"]["processed_cycles"], 2)

    def test_original_timestamps_are_preserved(self) -> None:
        cycles_df = self._synthetic_cycles_df()
        with tempfile.TemporaryDirectory() as temp_dir:
            stage_directory = Path(temp_dir) / "multi_sensor"
            result = self._run_stage(cycles_df, stage_directory, extract_all_cycles=True)

            all_rows = pd.concat(
                [pd.read_parquet(path) for path in result["output_paths"]["measurement_batch_files"]],
                ignore_index=True,
            )

        position_rows = all_rows[all_rows["signal_name"] == "position"]
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(position_rows["time"]))
        self.assertIn(pd.Timestamp("2026-01-01 00:00:00"), position_rows["time"].tolist())

    def test_missing_signal_is_recorded_not_filled(self) -> None:
        cycles_df = self._synthetic_cycles_df()
        with tempfile.TemporaryDirectory() as temp_dir:
            stage_directory = Path(temp_dir) / "multi_sensor"
            result = self._run_stage(cycles_df, stage_directory, extract_all_cycles=True)

        summary_df = result["signal_window_summary"]
        self.assertIn("is_missing", summary_df.columns)
        # Vibration signal has no samples in the session-2 cycles, so it must
        # be recorded as missing rather than filled with fake values.
        self.assertTrue(summary_df["is_missing"].any())

    def test_original_dataframes_are_not_mutated(self) -> None:
        cycles_df = self._synthetic_cycles_df()
        original_copy = cycles_df.copy(deep=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            stage_directory = Path(temp_dir) / "multi_sensor"
            self._run_stage(cycles_df, stage_directory, extract_all_cycles=True)

        pd.testing.assert_frame_equal(cycles_df, original_copy)

    def test_output_paths_and_checkpoint_are_present(self) -> None:
        cycles_df = self._synthetic_cycles_df()
        with tempfile.TemporaryDirectory() as temp_dir:
            stage_directory = Path(temp_dir) / "multi_sensor"
            result = self._run_stage(cycles_df, stage_directory)

            output_paths = result["output_paths"]
            for key in (
                "cycle_index_parquet",
                "measurement_batch_files",
                "signal_window_summary_parquet",
                "extraction_checkpoint",
                "selected_validation_cycles_parquet",
                "validation_cycle_quality_parquet",
            ):
                self.assertIn(key, output_paths)

            self.assertTrue(Path(output_paths["extraction_checkpoint"]).exists())
            self.assertTrue(Path(output_paths["signal_window_summary_parquet"]).exists())
            self.assertTrue(Path(output_paths["selected_validation_cycles_parquet"]).exists())
            self.assertTrue(Path(output_paths["validation_cycle_quality_parquet"]).exists())

    def test_resume_skips_already_completed_batches(self) -> None:
        cycles_df = self._synthetic_cycles_df()
        with tempfile.TemporaryDirectory() as temp_dir:
            stage_directory = Path(temp_dir) / "multi_sensor"
            first_result = self._run_stage(cycles_df, stage_directory, extract_all_cycles=True)
            first_batch_files = sorted(first_result["output_paths"]["measurement_batch_files"])
            first_mtimes = [Path(path).stat().st_mtime_ns for path in first_batch_files]

            second_result = self._run_stage(cycles_df, stage_directory, extract_all_cycles=True)
            second_batch_files = sorted(second_result["output_paths"]["measurement_batch_files"])
            second_mtimes = [Path(path).stat().st_mtime_ns for path in second_batch_files]

        self.assertEqual(first_batch_files, second_batch_files)
        self.assertEqual(first_mtimes, second_mtimes)


class RunFullPipelineMultiSensorIntegrationTests(unittest.TestCase):
    """End-to-end manifest coverage for the scalable multi-sensor stage."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)
        self.dataset_dir = create_dataset_fixture(self.root_path)
        self.output_root = self.root_path / "outputs"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_manifest_contains_scalable_output_paths(self) -> None:
        result = run_pipeline(
            PipelineConfig(
                dataset_path=self.dataset_dir,
                experiment="ExperimentAlpha",
                stop_after="multi_sensor_extraction",
                output_root=self.output_root,
                session_gap_seconds=3600.0,
                movement_threshold=1.0,
                validation_cycle_count=1,
                required_validation_signals=(),
                generate_validation_html=True,
            )
        )

        extraction_result = result["multi_sensor_extraction"]
        output_paths = extraction_result["output_paths"]
        self.assertIn("cycle_index_parquet", output_paths)
        self.assertIn("measurement_batch_files", output_paths)
        self.assertIn("signal_window_summary_parquet", output_paths)
        self.assertIn("extraction_checkpoint", output_paths)

        manifest_path = self.output_root.rglob("run_manifest.json")
        manifest_files = list(manifest_path)
        self.assertGreater(len(manifest_files), 0)
        manifest = json.loads(manifest_files[0].read_text(encoding="utf-8"))
        self.assertIn("multi_sensor_extraction", manifest["generated_output_paths"])
        self.assertIn(
            "cycle_index_parquet",
            manifest["generated_output_paths"]["multi_sensor_extraction"],
        )

        cycle_detection_paths = result["cycle_detection"]["output_paths"]
        self.assertIn("cycles_parquet", cycle_detection_paths)
        self.assertTrue(Path(cycle_detection_paths["cycles_parquet"]).exists())


class RunCycleQualityProfilingStageTests(unittest.TestCase):
    """Tests for the exploratory, non-rejecting cycle quality profiling stage."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)
        self.dataset_dir = create_dataset_fixture(self.root_path)

        metadata_frames = load_metadata(self.dataset_dir)
        self.uuid_signal_info = build_uuid_signal_info_from_metadata(metadata_frames)
        self.int_signal_info = build_int_signal_info_from_metadata(metadata_frames)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _synthetic_cycles_df(self) -> pd.DataFrame:
        rows = [
            {
                "cycle_id": 1,
                "session_id": 1,
                "start_time": pd.Timestamp("2026-01-01 00:00:00"),
                "end_time": pd.Timestamp("2026-01-01 00:00:01"),
                "duration_seconds": 1.0,
            },
            {
                "cycle_id": 2,
                "session_id": 1,
                "start_time": pd.Timestamp("2026-01-01 00:00:01"),
                "end_time": pd.Timestamp("2026-01-01 00:00:02"),
                "duration_seconds": 1.0,
            },
            {
                "cycle_id": 3,
                "session_id": 2,
                "start_time": pd.Timestamp("2026-01-01 02:00:00"),
                "end_time": pd.Timestamp("2026-01-01 02:00:00.5"),
                "duration_seconds": 0.5,
            },
            {
                "cycle_id": 4,
                "session_id": 2,
                "start_time": pd.Timestamp("2026-01-01 02:00:00.5"),
                "end_time": pd.Timestamp("2026-01-01 02:00:01"),
                "duration_seconds": 0.5,
            },
        ]
        cycles_df = pd.DataFrame(rows)
        cycles_df["experiment"] = "ExperimentAlpha"
        return cycles_df

    def _extract_measurements(self, cycles_df: pd.DataFrame, multi_sensor_directory: Path) -> Path:
        _run_multi_sensor_extraction_stage(
            dataset_path=self.dataset_dir,
            stage_directory=multi_sensor_directory,
            uuid_signal_info=self.uuid_signal_info,
            int_signal_info=self.int_signal_info,
            experiment="ExperimentAlpha",
            cycles_df=cycles_df,
            cycle_index_path=write_cycle_index(
                cycles_df, multi_sensor_directory.parent / "cycles" / "cycles.parquet"
            ),
            max_cycles_to_extract=100,
            extract_all_cycles=True,
            cycle_batch_size=4,
            resume_extraction=True,
            overwrite_existing=False,
            selected_extraction_signals=(),
            validation_cycle_count=1,
            required_validation_signals=("position",),
            minimum_samples_per_validation_cycle={"position": 1},
            require_consecutive_validation_cycles=True,
            max_cycles_to_scan_for_validation=10_000,
            generate_validation_html=False,
            generate_cycle_features=False,
            parquet_compression="snappy",
        )
        return multi_sensor_directory / "measurements"

    def test_does_not_reject_cycles_and_reports_missing_signals(self) -> None:
        cycles_df = self._synthetic_cycles_df()
        with tempfile.TemporaryDirectory() as temp_dir:
            multi_sensor_directory = Path(temp_dir) / "multi_sensor"
            measurements_root = self._extract_measurements(cycles_df, multi_sensor_directory)

            quality_directory = Path(temp_dir) / "quality_profiling"
            result = _run_cycle_quality_profiling_stage(
                stage_directory=quality_directory,
                measurement_dataset_path=measurements_root,
                extracted_cycles_df=cycles_df,
                quality_profiling_batch_size=2,
            )

            cycle_profile_df = result["cycle_quality_profile"]
            signal_quality_df = result["signal_quality_metrics"]
            distribution_df = result["distribution_summary"]

            # No rejection: every extracted cycle is profiled, regardless of
            # how many required signals are missing.
            self.assertEqual(len(cycle_profile_df), len(cycles_df))
            self.assertNotIn("is_validation_ready", cycle_profile_df.columns)
            self.assertNotIn("rejection_reason", cycle_profile_df.columns)

            # Vibration has no samples for the session-2 cycle, so it must be
            # reported as missing rather than silently dropped or filled.
            self.assertTrue(signal_quality_df["is_missing"].any())
            self.assertIn("missing_signal_names", cycle_profile_df.columns)

            self.assertFalse(distribution_df.empty)
            self.assertIn("p50", distribution_df.columns)

            for output_path in result["output_paths"].values():
                self.assertTrue(Path(output_path).exists())

    def test_full_pipeline_run_reaches_cycle_quality_profiling(self) -> None:
        output_root = self.root_path / "outputs"
        result = run_pipeline(
            PipelineConfig(
                dataset_path=self.dataset_dir,
                experiment="ExperimentAlpha",
                stop_after="cycle_quality_profiling",
                output_root=output_root,
                session_gap_seconds=3600.0,
                movement_threshold=1.0,
                validation_cycle_count=1,
                required_validation_signals=(),
                generate_validation_html=False,
            )
        )

        self.assertIn("cycle_quality_profiling", result["run"]["completed_stages"])
        profiling_result = result["cycle_quality_profiling"]
        for key in (
            "signal_quality_metrics_parquet",
            "cycle_quality_profile_parquet",
            "quality_metric_distribution_summary_csv",
        ):
            self.assertIn(key, profiling_result["output_paths"])
            self.assertTrue(Path(profiling_result["output_paths"][key]).exists())

        manifest_files = list(output_root.rglob("run_manifest.json"))
        self.assertGreater(len(manifest_files), 0)
        manifest = json.loads(manifest_files[0].read_text(encoding="utf-8"))
        self.assertIn("cycle_quality_profiling", manifest["generated_output_paths"])


class NewMethodologicalStageOrderTests(unittest.TestCase):
    """Integration tests for the profile -> generate rules -> validate sequence."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)
        self.dataset_dir = create_dataset_fixture(self.root_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_default_stage_order_is_exact(self) -> None:
        self.assertEqual(
            STAGE_ORDER,
            (
                PipelineStage.METADATA,
                PipelineStage.SIGNAL_DISCOVERY,
                PipelineStage.TIMESTAMP_ANALYSIS,
                PipelineStage.SESSION_DETECTION,
                PipelineStage.CYCLE_DETECTION,
                PipelineStage.MULTI_SENSOR_EXTRACTION,
                PipelineStage.CYCLE_QUALITY_PROFILING,
                PipelineStage.VALIDATION_RULE_GENERATION,
                PipelineStage.DATASET_VALIDATION,
                PipelineStage.FEATURE_ENGINEERING,
                PipelineStage.DATASET_GENERATION,
            ),
        )

    def test_cycle_selection_is_not_a_pipeline_stage(self) -> None:
        stage_values = {stage.value for stage in PipelineStage}
        self.assertNotIn("cycle_selection", stage_values)
        self.assertNotIn(PipelineStage.CYCLE_QUALITY_PROFILING.value, ("cycle_selection",))

    def test_cycle_selection_not_required_by_default_execution(self) -> None:
        implemented_values = {stage.value for stage in IMPLEMENTED_STAGES}
        self.assertNotIn("cycle_selection", implemented_values)

    def _run(self, stop_after: str) -> dict[str, object]:
        output_root = self.root_path / "outputs"
        return run_pipeline(
            PipelineConfig(
                dataset_path=self.dataset_dir,
                experiment="ExperimentAlpha",
                stop_after=stop_after,
                output_root=output_root,
                session_gap_seconds=3600.0,
                movement_threshold=1.0,
                extract_all_cycles=True,
                validation_cycle_count=1,
                required_validation_signals=(),
                generate_validation_html=False,
                minimum_samples_per_validation_cycle={"position": 1},
            )
        )

    def test_multi_sensor_extraction_reads_cycles_parquet_directly(self) -> None:
        result = self._run("multi_sensor_extraction")
        extraction_result = result["multi_sensor_extraction"]
        cycle_detection_result = result["cycle_detection"]
        self.assertEqual(
            sorted(extraction_result["cycles_extracted"]["cycle_id"].tolist()),
            sorted(cycle_detection_result["cycles"]["cycle_id"].tolist()),
        )
        self.assertNotIn("cycle_selection", result)

    def test_stop_after_validation_rule_generation(self) -> None:
        result = self._run("validation_rule_generation")
        self.assertEqual(result["run"]["completed_stages"][-1], "validation_rule_generation")
        self.assertNotIn("dataset_validation", result["run"]["completed_stages"])
        output_paths = result["validation_rule_generation"]["output_paths"]
        for key in (
            "validation_thresholds_json",
            "threshold_derivation_summary_csv",
            "rule_generation_summary_json",
            "skipped_metrics_csv",
        ):
            self.assertIn(key, output_paths)
            self.assertTrue(Path(output_paths[key]).exists())

    def test_stop_after_dataset_validation(self) -> None:
        result = self._run("dataset_validation")
        self.assertEqual(result["run"]["completed_stages"][-1], "dataset_validation")
        output_paths = result["dataset_validation"]["output_paths"]
        for key in (
            "cycle_validation_results_parquet",
            "signal_validation_results_parquet",
            "validation_reason_summary_csv",
            "validation_summary_json",
            "valid_core_cycles_parquet",
            "valid_complete_multisensor_cycles_parquet",
            "invalid_cycles_parquet",
        ):
            self.assertIn(key, output_paths)
            self.assertTrue(Path(output_paths[key]).exists())

    def test_manifest_row_counts_for_new_stages(self) -> None:
        output_root = self.root_path / "outputs"
        run_pipeline(
            PipelineConfig(
                dataset_path=self.dataset_dir,
                experiment="ExperimentAlpha",
                stop_after="dataset_validation",
                output_root=output_root,
                session_gap_seconds=3600.0,
                movement_threshold=1.0,
                extract_all_cycles=True,
                validation_cycle_count=1,
                required_validation_signals=(),
                generate_validation_html=False,
                minimum_samples_per_validation_cycle={"position": 1},
            )
        )
        manifest_files = list(output_root.rglob("run_manifest.json"))
        self.assertGreater(len(manifest_files), 0)
        manifest = json.loads(manifest_files[0].read_text(encoding="utf-8"))

        rule_generation_counts = manifest["row_counts"]["validation_rule_generation"]
        for key in (
            "rules_generated",
            "metrics_skipped",
            "provisional_rules",
            "reference_cycles",
        ):
            self.assertIn(key, rule_generation_counts)

        dataset_validation_counts = manifest["row_counts"]["dataset_validation"]
        for key in (
            "cycles_evaluated",
            "valid_core_cycles",
            "valid_complete_multisensor_cycles",
            "invalid_cycles",
            "vibration_unavailable_cycles",
            "vibration_partial_cycles",
            "vibration_complete_cycles",
        ):
            self.assertIn(key, dataset_validation_counts)


if __name__ == "__main__":
    unittest.main()
