"""Command-line entry point for the reusable preprocessing pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import PipelineConfig, run_pipeline  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run the MasterThesis preprocessing pipeline.")
    parser.add_argument("--dataset", help="Dataset directory path or relative dataset name.")
    parser.add_argument("--experiment", help="Experiment name to process.")
    parser.add_argument("--stop-after", dest="stop_after", help="Pipeline stage to stop after.")
    parser.add_argument("--reference-signal", dest="reference_signal", help="Reference signal unit code.")
    parser.add_argument(
        "--session-gap-seconds",
        dest="session_gap_seconds",
        type=float,
        help="Gap threshold in seconds for recording-session detection.",
    )
    parser.add_argument(
        "--movement-threshold",
        dest="movement_threshold",
        type=float,
        help="Movement threshold for cycle detection.",
    )
    parser.add_argument("--output-root", dest="output_root", help="Root output directory.")
    parser.add_argument("--config", help="Optional YAML configuration file.")
    return parser.parse_args()


def _load_yaml_config(config_path: str | None) -> dict[str, Any]:
    """Load one YAML configuration file if provided."""

    if config_path is None:
        return {}

    path = Path(config_path).expanduser()
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Pipeline config must be a mapping: {path}")
    return loaded


def _pick_value(cli_value: Any, yaml_value: Any, default: Any = None) -> Any:
    """Prefer a CLI value when present, otherwise fall back to YAML or a default."""

    if cli_value is not None:
        return cli_value
    if yaml_value is not None:
        return yaml_value
    return default


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    """Merge YAML and command-line options into a pipeline config."""

    yaml_config = _load_yaml_config(args.config)

    dataset_value = _pick_value(
        args.dataset,
        yaml_config.get("dataset_path", yaml_config.get("dataset")),
    )
    experiment_value = _pick_value(args.experiment, yaml_config.get("experiment"))
    stop_after_value = _pick_value(args.stop_after, yaml_config.get("stop_after"))
    if dataset_value is None or experiment_value is None or stop_after_value is None:
        raise ValueError(
            "dataset, experiment, and stop_after must be provided through CLI arguments or YAML."
        )

    return PipelineConfig(
        dataset_path=Path(dataset_value),
        experiment=str(experiment_value),
        stop_after=str(stop_after_value),
        reference_signal=str(
            _pick_value(args.reference_signal, yaml_config.get("reference_signal"), "position")
        ),
        session_gap_seconds=_pick_value(
            args.session_gap_seconds,
            yaml_config.get("session_gap_seconds"),
        ),
        movement_threshold=float(
            _pick_value(args.movement_threshold, yaml_config.get("movement_threshold"), 1.0)
        ),
        output_root=Path(_pick_value(args.output_root, yaml_config.get("output_root"), "outputs")),
        max_cycles_to_extract=int(
            _pick_value(
                None,
                yaml_config.get("max_cycles_to_extract"),
                3,
            )
        ),
        extract_all_cycles=bool(yaml_config.get("extract_all_cycles", False)),
        cycle_batch_size=int(yaml_config.get("cycle_batch_size", 500)),
        resume_extraction=bool(yaml_config.get("resume_extraction", True)),
        overwrite_existing=bool(yaml_config.get("overwrite_existing", False)),
        selected_extraction_signals=tuple(yaml_config.get("selected_extraction_signals", ()) or ()),
        validation_cycle_count=int(yaml_config.get("validation_cycle_count", 3)),
        required_validation_signals=tuple(
            yaml_config.get("required_validation_signals", ()) or ()
        ),
        minimum_samples_per_validation_cycle=yaml_config.get(
            "minimum_samples_per_validation_cycle"
        ),
        require_consecutive_validation_cycles=bool(
            yaml_config.get("require_consecutive_validation_cycles", True)
        ),
        max_cycles_to_scan_for_validation=yaml_config.get(
            "max_cycles_to_scan_for_validation", 10_000
        ),
        generate_validation_html=bool(yaml_config.get("generate_validation_html", True)),
        generate_cycle_features=bool(yaml_config.get("generate_cycle_features", False)),
        parquet_compression=str(yaml_config.get("parquet_compression", "zstd")),
        quality_profiling_batch_size=int(yaml_config.get("quality_profiling_batch_size", 1000)),
        signal_roles=yaml_config.get("signal_roles"),
        validation_rule_generation=yaml_config.get("validation_rule_generation"),
        dataset_validation=yaml_config.get("dataset_validation"),
        cycle_tensor_generation=yaml_config.get("cycle_tensor_generation"),
    )


def _important_counts(result: dict[str, object]) -> dict[str, int]:
    """Extract a small set of useful row counts for the final summary."""

    counts: dict[str, int] = {}
    for stage_name, stage_result in result.items():
        if stage_name == "run":
            continue
        row_counts = stage_result.get("row_counts", {})
        for key, value in row_counts.items():
            if isinstance(value, int):
                counts[f"{stage_name}.{key}"] = value
    return counts


def main() -> None:
    """Run the pipeline and print a concise summary."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    config = _build_config(_parse_args())
    result = run_pipeline(config)
    run_summary = result["run"]
    counts = _important_counts(result)

    print("completed stages:", ", ".join(run_summary["completed_stages"]))
    print("output directory:", run_summary["run_directory"])
    if counts:
        count_summary = ", ".join(f"{key}={value}" for key, value in counts.items())
        print("important row counts:", count_summary)
    print("stopped after stage:", run_summary["stop_after"])
    print("runtime (seconds):", f"{run_summary['runtime_seconds']:.3f}")


if __name__ == "__main__":
    main()
