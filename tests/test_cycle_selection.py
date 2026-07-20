"""Behavior tests for the vibration-aware ``cycle_selection`` pipeline stage."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.pipeline import (
    build_int_signal_info_from_metadata,
    build_uuid_signal_info_from_metadata,
    load_metadata,
)
from src.preprocessing.cycle_selection import (
    CycleSelectionConfig,
    MODE_COMPLETE_MULTISENSOR_STRATIFIED,
    MODE_FIRST_N,
    _BurstTracker,
    insufficient_signal_code,
    missing_signal_code,
    select_cycles,
)
from src.storage.cycle_index_writer import write_cycle_index

EXPERIMENT = "ExperimentAlpha"
REQUIRED_SIGNALS: tuple[str, ...] = ("position", "vibration_x", "vibration_y", "vibration_z")
MINIMUM_SAMPLES: dict[str, int] = {
    "position": 3,
    "vibration_x": 3,
    "vibration_y": 3,
    "vibration_z": 3,
}

BURST_PERIOD_SECONDS = 100.0
BURST_DURATION_SECONDS = 20.0
SESSION_LENGTH_SECONDS = 300.0
CYCLE_SPACING_SECONDS = 5.0
CYCLE_DURATION_SECONDS = 4.0


def _write_parquet_table(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)


def _write_uuid_measurements(dataset_dir: Path, frame: pd.DataFrame) -> None:
    pq.write_to_dataset(
        pa.Table.from_pandas(frame, preserve_index=False),
        root_path=dataset_dir / "signal_data_point.parquet",
        partition_cols=["signal_id"],
    )


def _session_position_series(session_offset_seconds: float) -> pd.DataFrame:
    """Continuous 1s-spaced position samples covering one whole session.

    A few extra half-second samples are added so the dataset has some
    sub-1-second timestamp gaps too (matching real recordings and avoiding
    an empty "gaps under 1 second" histogram in timestamp analysis).
    """

    seconds = np.arange(0.0, SESSION_LENGTH_SECONDS + 1.0, 1.0)
    half_second_seconds = seconds[:-1] + 0.5
    all_seconds = np.sort(np.concatenate([seconds, half_second_seconds]))
    times = pd.to_datetime(session_offset_seconds + all_seconds, unit="s", origin="2026-01-01")
    return pd.DataFrame({"time": times, "value": np.sin(all_seconds / 10.0)})


def _session_vibration_series(session_offset_seconds: float) -> pd.DataFrame:
    """0.5s-spaced vibration samples only inside each duty-cycle burst."""

    rows: list[pd.DataFrame] = []
    burst_start = 0.0
    while burst_start < SESSION_LENGTH_SECONDS:
        burst_seconds = np.arange(burst_start, burst_start + BURST_DURATION_SECONDS, 0.5)
        times = pd.to_datetime(
            session_offset_seconds + burst_seconds, unit="s", origin="2026-01-01"
        )
        rows.append(pd.DataFrame({"time": times, "value": np.full(burst_seconds.shape, 0.1)}))
        burst_start += BURST_PERIOD_SECONDS
    return pd.concat(rows, ignore_index=True)


def build_cycle_selection_fixture(
    root_dir: Path,
    session_offsets_seconds: tuple[float, ...] = (0.0, 10_000.0),
) -> tuple[Path, list[dict[str, object]]]:
    """Create a small ERA-shaped dataset with duty-cycled vibration bursts.

    Returns the dataset directory and a list of session descriptors
    (``session_id``, ``offset_seconds``) so callers can build matching
    candidate cycles and a ``sessions_df``.
    """

    dataset_dir = root_dir / "FixtureDataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    experiment_node_id = str(uuid4())
    drive_node_id = str(uuid4())
    vibration_node_ids = {axis: str(uuid4()) for axis in ("x", "y", "z")}

    position_signal_uuid = str(uuid4())
    vibration_signal_uuids = {axis: str(uuid4()) for axis in ("x", "y", "z")}

    nodes_rows = [
        {"node_id": experiment_node_id, "name": EXPERIMENT, "parent_node": pd.NA},
        {"node_id": drive_node_id, "name": "Drive", "parent_node": experiment_node_id},
    ]
    for axis, node_id in vibration_node_ids.items():
        nodes_rows.append({"node_id": node_id, "name": axis, "parent_node": experiment_node_id})

    _write_parquet_table(dataset_dir / "nodes.parquet", pd.DataFrame(nodes_rows))
    _write_parquet_table(
        dataset_dir / "units.parquet",
        pd.DataFrame(
            [
                {"common_code": "position", "symbol": "mm", "name": "Position"},
                {"common_code": "vibration", "symbol": "m/s2", "name": "Vibration"},
            ]
        ),
    )

    rel_rows = [{"signal_id": position_signal_uuid, "node_id": drive_node_id, "unit": "position"}]
    for axis, node_id in vibration_node_ids.items():
        rel_rows.append(
            {"signal_id": vibration_signal_uuids[axis], "node_id": node_id, "unit": "vibration"}
        )
    _write_parquet_table(dataset_dir / "signal_data_point_rel.parquet", pd.DataFrame(rel_rows))
    _write_parquet_table(dataset_dir / "signal_data_point_rel_int.parquet", pd.DataFrame(columns=["signal_id", "node_id", "unit"]))

    position_frames = []
    vibration_frames: dict[str, list[pd.DataFrame]] = {"x": [], "y": [], "z": []}
    for offset in session_offsets_seconds:
        position_df = _session_position_series(offset)
        position_df["signal_id"] = position_signal_uuid
        position_frames.append(position_df)
        for axis in ("x", "y", "z"):
            vibration_df = _session_vibration_series(offset)
            vibration_df["signal_id"] = vibration_signal_uuids[axis]
            vibration_frames[axis].append(vibration_df)

    _write_uuid_measurements(dataset_dir, pd.concat(position_frames, ignore_index=True))
    for axis in ("x", "y", "z"):
        _write_uuid_measurements(dataset_dir, pd.concat(vibration_frames[axis], ignore_index=True))

    session_descriptors = [
        {"session_id": index + 1, "offset_seconds": offset}
        for index, offset in enumerate(session_offsets_seconds)
    ]
    return dataset_dir, session_descriptors


def _session_bounds_df(session_descriptors: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for descriptor in session_descriptors:
        offset = float(descriptor["offset_seconds"])
        rows.append(
            {
                "session_id": int(descriptor["session_id"]),
                "start_time": pd.Timestamp("2026-01-01") + pd.Timedelta(seconds=offset),
                "end_time": pd.Timestamp("2026-01-01")
                + pd.Timedelta(seconds=offset + SESSION_LENGTH_SECONDS),
            }
        )
    return pd.DataFrame(rows)


def _candidate_cycles_for_session(session_id: int, offset_seconds: float) -> pd.DataFrame:
    """Build 4s-duration candidate cycles every 5s across one whole session."""

    rows = []
    cycle_start = 0.0
    cycle_id_within_session = 0
    while cycle_start + CYCLE_DURATION_SECONDS <= SESSION_LENGTH_SECONDS:
        cycle_id_within_session += 1
        start_time = pd.Timestamp("2026-01-01") + pd.Timedelta(
            seconds=offset_seconds + cycle_start
        )
        end_time = start_time + pd.Timedelta(seconds=CYCLE_DURATION_SECONDS)
        rows.append(
            {
                "experiment": EXPERIMENT,
                "session_id": session_id,
                "cycle_id": None,
                "start_time": start_time,
                "end_time": end_time,
                "duration_seconds": CYCLE_DURATION_SECONDS,
            }
        )
        cycle_start += CYCLE_SPACING_SECONDS
    return pd.DataFrame(rows)


def build_candidate_cycles(session_descriptors: list[dict[str, object]]) -> pd.DataFrame:
    """Build the full multi-session candidate cycle index used by tests."""

    frames = [
        _candidate_cycles_for_session(int(descriptor["session_id"]), float(descriptor["offset_seconds"]))
        for descriptor in session_descriptors
    ]
    cycles_df = pd.concat(frames, ignore_index=True)
    cycles_df["cycle_id"] = range(1, len(cycles_df) + 1)
    return cycles_df


class CycleSelectionFixtureTestCase(unittest.TestCase):
    """Base class providing the shared duty-cycled vibration fixture."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)
        self.dataset_dir, self.session_descriptors = build_cycle_selection_fixture(self.root_path)

        metadata_frames = load_metadata(self.dataset_dir)
        self.uuid_signal_info = build_uuid_signal_info_from_metadata(metadata_frames)
        self.int_signal_info = build_int_signal_info_from_metadata(metadata_frames)

        self.sessions_df = _session_bounds_df(self.session_descriptors)
        self.candidate_cycles_df = build_candidate_cycles(self.session_descriptors)
        self.cycles_parquet_path = write_cycle_index(
            self.candidate_cycles_df, self.root_path / "cycles" / "cycles.parquet"
        )

        # 20 candidate cycles per 100s burst period; the first 4 (0-19s) fall
        # fully inside the 20s vibration burst -> 4 eligible per burst.
        self.eligible_cycles_per_burst = 4
        self.bursts_per_session = math.ceil(SESSION_LENGTH_SECONDS / BURST_PERIOD_SECONDS)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _select(self, **config_overrides: object):
        config = CycleSelectionConfig.from_mapping(
            {
                "mode": MODE_COMPLETE_MULTISENSOR_STRATIFIED,
                "target_cycle_count": 10,
                "candidate_batch_size": 50,
                "max_cycles_to_scan": 100_000,
                "time_strata_per_session": 3,
                "max_cycles_per_vibration_burst": 4,
                "vibration_burst_gap_seconds": 60.0,
                "random_seed": 42,
                **config_overrides,
            }
        )
        return select_cycles(
            base_dir=self.dataset_dir,
            cycles_parquet_path=self.cycles_parquet_path,
            sessions_df=self.sessions_df,
            experiment=EXPERIMENT,
            required_signals=REQUIRED_SIGNALS,
            minimum_samples=MINIMUM_SAMPLES,
            uuid_signal_info=self.uuid_signal_info,
            int_signal_info=self.int_signal_info,
            config=config,
        )


class BurstTrackerTests(unittest.TestCase):
    """Direct unit tests for the sequential vibration burst-id assignment."""

    def test_gap_above_threshold_starts_new_burst(self) -> None:
        tracker = _BurstTracker(gap_seconds=60.0)
        times = pd.to_datetime(
            ["2026-01-01 00:00:00", "2026-01-01 00:00:05", "2026-01-01 00:05:00"]
        ).to_numpy()
        burst_ids = tracker.assign(times)
        self.assertEqual(list(burst_ids), [0, 0, 1])

    def test_state_persists_across_batches(self) -> None:
        tracker = _BurstTracker(gap_seconds=60.0)
        first = tracker.assign(pd.to_datetime(["2026-01-01 00:00:00"]).to_numpy())
        second = tracker.assign(pd.to_datetime(["2026-01-01 00:00:05"]).to_numpy())
        third = tracker.assign(pd.to_datetime(["2026-01-01 00:05:00"]).to_numpy())
        self.assertEqual(list(first) + list(second) + list(third), [0, 0, 1])


class RejectionCodeHelperTests(unittest.TestCase):
    """Tests for the generic per-signal rejection-code naming helpers."""

    def test_missing_and_insufficient_code_names(self) -> None:
        self.assertEqual(missing_signal_code("vibration_x"), "missing_vibration_x")
        self.assertEqual(insufficient_signal_code("vibration_x"), "insufficient_vibration_x_samples")
        self.assertEqual(missing_signal_code("position"), "missing_position")


class FirstNModeTests(CycleSelectionFixtureTestCase):
    """first_n mode must preserve the previous simple selection behavior."""

    def test_first_n_mode_returns_leading_cycles_unchanged(self) -> None:
        result = self._select(mode=MODE_FIRST_N, target_cycle_count=7)

        expected_ids = self.candidate_cycles_df.head(7)["cycle_id"].tolist()
        self.assertEqual(result.selected_cycles["cycle_id"].tolist(), expected_ids)
        self.assertEqual(result.summary["selection_mode"], MODE_FIRST_N)
        self.assertEqual(result.summary["selection_shortfall"], 0)

    def test_first_n_mode_does_not_reject_any_cycle(self) -> None:
        result = self._select(mode=MODE_FIRST_N, target_cycle_count=5)

        self.assertTrue(result.candidate_evaluation["eligible"].all())
        self.assertEqual(result.rejection_reason_counts, {})


class EligibilityTests(CycleSelectionFixtureTestCase):
    """Tests covering per-cycle eligibility and rejection-code assignment."""

    def test_valid_complete_multisensor_cycle(self) -> None:
        result = self._select(target_cycle_count=1)

        self.assertGreater(len(result.selected_cycles), 0)
        first_row = result.selected_cycles.iloc[0]
        self.assertGreaterEqual(first_row["vibration_x_sample_count"], MINIMUM_SAMPLES["vibration_x"])
        self.assertGreaterEqual(first_row["position_sample_count"], MINIMUM_SAMPLES["position"])

    def test_missing_all_vibration_axes(self) -> None:
        result = self._select(target_cycle_count=100)

        evaluation_df = result.candidate_evaluation
        # Cycles well outside any burst (e.g. the 3rd candidate of each burst
        # period, at t=10s within a burst window [20,100) gap) must be
        # rejected for missing every vibration axis.
        outside_burst = evaluation_df[
            (evaluation_df["session_id"] == 1)
            & (evaluation_df["start_time"] == pd.Timestamp("2026-01-01 00:00:30"))
        ]
        self.assertEqual(len(outside_burst), 1)
        reasons = outside_burst.iloc[0]["rejection_reasons"]
        self.assertIn("missing_vibration_x", reasons)
        self.assertIn("missing_vibration_y", reasons)
        self.assertIn("missing_vibration_z", reasons)
        self.assertFalse(outside_burst.iloc[0]["eligible"])

    def test_invalid_cycle_interval_is_rejected(self) -> None:
        cycles_df = self.candidate_cycles_df.copy()
        cycles_df.loc[0, "end_time"] = cycles_df.loc[0, "start_time"] - pd.Timedelta(seconds=1)
        cycles_parquet_path = write_cycle_index(cycles_df, self.root_path / "cycles_invalid" / "cycles.parquet")

        result = select_cycles(
            base_dir=self.dataset_dir,
            cycles_parquet_path=cycles_parquet_path,
            sessions_df=self.sessions_df,
            experiment=EXPERIMENT,
            required_signals=REQUIRED_SIGNALS,
            minimum_samples=MINIMUM_SAMPLES,
            uuid_signal_info=self.uuid_signal_info,
            int_signal_info=self.int_signal_info,
            config=CycleSelectionConfig.from_mapping(
                {"mode": MODE_COMPLETE_MULTISENSOR_STRATIFIED, "target_cycle_count": 5, "candidate_batch_size": 50}
            ),
        )

        invalid_row = result.candidate_evaluation[
            result.candidate_evaluation["cycle_id"] == cycles_df.loc[0, "cycle_id"]
        ].iloc[0]
        self.assertIn("invalid_cycle_interval", invalid_row["rejection_reasons"])
        self.assertFalse(invalid_row["eligible"])

    def test_insufficient_vibration_samples_rejected_when_threshold_raised(self) -> None:
        # Each 4s vibration burst window has ~9 samples at 0.5s spacing;
        # raising the minimum well above that must yield an "insufficient"
        # code instead of "missing" for burst-covered cycles.
        strict_config = CycleSelectionConfig.from_mapping(
            {
                "mode": MODE_COMPLETE_MULTISENSOR_STRATIFIED,
                "target_cycle_count": 100,
                "candidate_batch_size": 50,
                "time_strata_per_session": 3,
            }
        )
        strict_result = select_cycles(
            base_dir=self.dataset_dir,
            cycles_parquet_path=self.cycles_parquet_path,
            sessions_df=self.sessions_df,
            experiment=EXPERIMENT,
            required_signals=REQUIRED_SIGNALS,
            minimum_samples={**MINIMUM_SAMPLES, "vibration_x": 50, "vibration_y": 50, "vibration_z": 50},
            uuid_signal_info=self.uuid_signal_info,
            int_signal_info=self.int_signal_info,
            config=strict_config,
        )
        self.assertEqual(len(strict_result.selected_cycles), 0)
        self.assertGreater(
            strict_result.rejection_reason_counts.get("insufficient_vibration_x_samples", 0), 0
        )

    def test_insufficient_non_vibration_samples_rejected(self) -> None:
        # Burst-covered cycles normally have ~9 position samples in their
        # 4s window; raising the position minimum well above that must
        # yield "insufficient_position_samples" for otherwise-eligible
        # cycles, without affecting the vibration rejection codes.
        strict_result = select_cycles(
            base_dir=self.dataset_dir,
            cycles_parquet_path=self.cycles_parquet_path,
            sessions_df=self.sessions_df,
            experiment=EXPERIMENT,
            required_signals=REQUIRED_SIGNALS,
            minimum_samples={**MINIMUM_SAMPLES, "position": 500},
            uuid_signal_info=self.uuid_signal_info,
            int_signal_info=self.int_signal_info,
            config=CycleSelectionConfig.from_mapping(
                {
                    "mode": MODE_COMPLETE_MULTISENSOR_STRATIFIED,
                    "target_cycle_count": 100,
                    "candidate_batch_size": 50,
                }
            ),
        )
        self.assertEqual(len(strict_result.selected_cycles), 0)
        self.assertGreater(
            strict_result.rejection_reason_counts.get("insufficient_position_samples", 0), 0
        )

    def test_one_missing_vibration_axis_rejected(self) -> None:
        # Build a variant dataset where vibration_y has no samples in the
        # very first burst window (session 1, [0s, 20s)) while x and z keep
        # their full burst coverage there, so the first candidate cycle is
        # rejected for exactly one missing axis.
        metadata_frames = load_metadata(self.dataset_dir)
        uuid_signal_info = build_uuid_signal_info_from_metadata(metadata_frames)
        vibration_rows = uuid_signal_info[uuid_signal_info["unit_code"] == "vibration"]

        partial_dataset_dir = self.root_path / "PartialAxisFixture"
        partial_dataset_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "nodes.parquet",
            "units.parquet",
            "signal_data_point_rel.parquet",
            "signal_data_point_rel_int.parquet",
        ):
            (partial_dataset_dir / name).write_bytes((self.dataset_dir / name).read_bytes())

        position_row = uuid_signal_info[uuid_signal_info["unit_code"] == "position"].iloc[0]
        position_df = _session_position_series(0.0)
        position_df["signal_id"] = position_row["signal_id_uuid"]
        _write_uuid_measurements(partial_dataset_dir, position_df)

        for _, row in vibration_rows.iterrows():
            vibration_df = _session_vibration_series(0.0)
            path_leaf = str(row["path"]).rsplit("/", 1)[-1].lower()
            if path_leaf == "y":
                # Drop every sample inside the first burst window only.
                vibration_df = vibration_df[
                    ~(
                        (vibration_df["time"] >= pd.Timestamp("2026-01-01 00:00:00"))
                        & (vibration_df["time"] < pd.Timestamp("2026-01-01 00:00:20"))
                    )
                ]
            vibration_df["signal_id"] = row["signal_id_uuid"]
            _write_uuid_measurements(partial_dataset_dir, vibration_df)

        result = select_cycles(
            base_dir=partial_dataset_dir,
            cycles_parquet_path=self.cycles_parquet_path,
            sessions_df=self.sessions_df,
            experiment=EXPERIMENT,
            required_signals=REQUIRED_SIGNALS,
            minimum_samples=MINIMUM_SAMPLES,
            uuid_signal_info=uuid_signal_info,
            int_signal_info=self.int_signal_info,
            config=CycleSelectionConfig.from_mapping(
                {
                    "mode": MODE_COMPLETE_MULTISENSOR_STRATIFIED,
                    "target_cycle_count": 5,
                    "candidate_batch_size": 50,
                }
            ),
        )

        first_cycle_id = int(self.candidate_cycles_df.iloc[0]["cycle_id"])
        first_row = result.candidate_evaluation[
            result.candidate_evaluation["cycle_id"] == first_cycle_id
        ].iloc[0]
        self.assertFalse(first_row["eligible"])
        self.assertIn("missing_vibration_y", first_row["rejection_reasons"])
        self.assertNotIn("missing_vibration_x", first_row["rejection_reasons"])
        self.assertNotIn("missing_vibration_z", first_row["rejection_reasons"])

    def test_non_finite_required_signal_rejected(self) -> None:
        # Overwrite the position signal with an all-NaN series so every
        # cycle has samples but none are finite.
        nan_position = _session_position_series(0.0)
        nan_position["value"] = np.nan
        metadata_frames = load_metadata(self.dataset_dir)
        uuid_signal_info = build_uuid_signal_info_from_metadata(metadata_frames)
        position_row = uuid_signal_info[uuid_signal_info["unit_code"] == "position"].iloc[0]

        nan_dataset_dir = self.root_path / "NanFixture"
        nan_dataset_dir.mkdir(parents=True, exist_ok=True)
        for name in ("nodes.parquet", "units.parquet", "signal_data_point_rel.parquet", "signal_data_point_rel_int.parquet"):
            (nan_dataset_dir / name).write_bytes((self.dataset_dir / name).read_bytes())

        nan_position["signal_id"] = position_row["signal_id_uuid"]
        _write_uuid_measurements(nan_dataset_dir, nan_position)
        vibration_signal_info = uuid_signal_info[uuid_signal_info["unit_code"] == "vibration"]
        for _, row in vibration_signal_info.iterrows():
            vibration_df = _session_vibration_series(0.0)
            vibration_df["signal_id"] = row["signal_id_uuid"]
            _write_uuid_measurements(nan_dataset_dir, vibration_df)

        result = select_cycles(
            base_dir=nan_dataset_dir,
            cycles_parquet_path=self.cycles_parquet_path,
            sessions_df=self.sessions_df,
            experiment=EXPERIMENT,
            required_signals=REQUIRED_SIGNALS,
            minimum_samples=MINIMUM_SAMPLES,
            uuid_signal_info=uuid_signal_info,
            int_signal_info=self.int_signal_info,
            config=CycleSelectionConfig.from_mapping(
                {"mode": MODE_COMPLETE_MULTISENSOR_STRATIFIED, "target_cycle_count": 5, "candidate_batch_size": 50}
            ),
        )
        self.assertGreater(result.rejection_reason_counts.get("non_finite_required_signal", 0), 0)


class BurstDetectionAndLimitTests(CycleSelectionFixtureTestCase):
    """Tests for vibration burst-id assignment and the per-burst selection cap."""

    def test_vibration_burst_ids_increase_monotonically(self) -> None:
        result = self._select(target_cycle_count=100)
        selected = result.selected_cycles.sort_values("start_time")
        burst_ids = selected["vibration_burst_id"].tolist()
        self.assertEqual(burst_ids, sorted(burst_ids))

    def test_maximum_cycles_per_vibration_burst_is_enforced(self) -> None:
        result = self._select(target_cycle_count=100, max_cycles_per_vibration_burst=2)

        counts_per_burst = result.selected_cycles.groupby("vibration_burst_id").size()
        self.assertTrue((counts_per_burst <= 2).all())

        limited_rows = result.candidate_evaluation[
            result.candidate_evaluation["rejection_reasons"].str.contains(
                "burst_selection_limit", na=False
            )
        ]
        self.assertGreater(len(limited_rows), 0)

    def test_default_config_selects_at_most_four_per_burst(self) -> None:
        result = self._select(target_cycle_count=1000)
        counts_per_burst = result.selected_cycles.groupby("vibration_burst_id").size()
        self.assertTrue((counts_per_burst <= 4).all())


class RepresentativeSelectionTests(CycleSelectionFixtureTestCase):
    """Tests for session/time-stratified, deterministic representative selection."""

    def test_balanced_selection_across_sessions(self) -> None:
        result = self._select(target_cycle_count=16)

        counts_per_session = result.selected_cycles.groupby("session_id").size()
        self.assertEqual(set(counts_per_session.index), {1, 2})
        self.assertGreater(counts_per_session.min(), 0)

    def test_temporal_stratification_inside_sessions(self) -> None:
        result = self._select(target_cycle_count=16, time_strata_per_session=3)

        session_1_strata = result.selected_cycles[result.selected_cycles["session_id"] == 1][
            "time_stratum"
        ]
        self.assertGreater(session_1_strata.nunique(), 1)

    def test_deterministic_with_fixed_random_seed(self) -> None:
        first_result = self._select(target_cycle_count=10, random_seed=42)
        second_result = self._select(target_cycle_count=10, random_seed=42)

        self.assertEqual(
            first_result.selected_cycles["cycle_id"].tolist(),
            second_result.selected_cycles["cycle_id"].tolist(),
        )
        self.assertEqual(
            first_result.selected_cycles["selection_rank"].tolist(),
            second_result.selected_cycles["selection_rank"].tolist(),
        )

    def test_fewer_eligible_cycles_than_requested_records_shortfall(self) -> None:
        with self.assertLogs("src.preprocessing.cycle_selection", level="WARNING"):
            result = self._select(target_cycle_count=10_000)

        eligible_count = result.summary["eligible_cycles_found"]
        self.assertEqual(len(result.selected_cycles), eligible_count)
        self.assertEqual(result.summary["selection_shortfall"], 10_000 - eligible_count)

    def test_no_eligible_cycles_does_not_crash(self) -> None:
        strict_config_overrides = {
            "vibration_x": 10_000,
            "vibration_y": 10_000,
            "vibration_z": 10_000,
        }
        result = select_cycles(
            base_dir=self.dataset_dir,
            cycles_parquet_path=self.cycles_parquet_path,
            sessions_df=self.sessions_df,
            experiment=EXPERIMENT,
            required_signals=REQUIRED_SIGNALS,
            minimum_samples={**MINIMUM_SAMPLES, **strict_config_overrides},
            uuid_signal_info=self.uuid_signal_info,
            int_signal_info=self.int_signal_info,
            config=CycleSelectionConfig.from_mapping(
                {"mode": MODE_COMPLETE_MULTISENSOR_STRATIFIED, "target_cycle_count": 10, "candidate_batch_size": 50}
            ),
        )
        self.assertEqual(len(result.selected_cycles), 0)
        self.assertEqual(result.summary["eligible_cycles_found"], 0)
        self.assertEqual(result.summary["selection_shortfall"], 10)


class SummaryJsonSafetyTests(CycleSelectionFixtureTestCase):
    """Tests that the selection_summary payload is always valid, finite JSON."""

    def test_json_contains_no_nan_or_infinity(self) -> None:
        result = self._select(target_cycle_count=10)
        payload = json.dumps(result.summary, allow_nan=False)
        self.assertNotIn("NaN", payload)
        self.assertNotIn("Infinity", payload)

    def test_json_safety_holds_for_zero_eligible_case(self) -> None:
        result = select_cycles(
            base_dir=self.dataset_dir,
            cycles_parquet_path=self.cycles_parquet_path,
            sessions_df=self.sessions_df,
            experiment=EXPERIMENT,
            required_signals=REQUIRED_SIGNALS,
            minimum_samples={**MINIMUM_SAMPLES, "vibration_x": 10_000},
            uuid_signal_info=self.uuid_signal_info,
            int_signal_info=self.int_signal_info,
            config=CycleSelectionConfig.from_mapping(
                {"mode": MODE_COMPLETE_MULTISENSOR_STRATIFIED, "target_cycle_count": 5, "candidate_batch_size": 50}
            ),
        )
        payload = json.dumps(result.summary, allow_nan=False)
        self.assertNotIn("NaN", payload)
        self.assertNotIn("Infinity", payload)


if __name__ == "__main__":
    unittest.main()
