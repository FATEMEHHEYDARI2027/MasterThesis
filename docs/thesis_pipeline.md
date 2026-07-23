# Preprocessing Pipeline Documentation

Project:
Predictive Maintenance for Industrial Electric Actuators

Author:
Fatemeh Heydari

Version:
2.2

Last Update:
2026-07-23

---

# Purpose

The objective of the preprocessing pipeline is to transform large-scale raw
industrial measurement data into structured multi-sensor actuator cycles that
can later be used for feature engineering and anomaly detection.

The pipeline was designed according to the following requirements:

- automatic processing of new experiments,
- scalability to industrial datasets,
- memory-efficient processing,
- preservation of original measurements,
- reproducible preprocessing.

---

# Overall Pipeline

```

Raw Measurement Data
↓

Metadata Integration
↓

Signal Discovery
↓

Timestamp Analysis
↓

Recording Session Detection

↓

Position-Based Cycle Detection

↓

Multi-Sensor Cycle Extraction

↓

Cycle Quality Profiling (descriptive, non-rejecting)

↓

Validation Rule Generation (robust, data-driven thresholds)

↓

Dataset Validation (frozen-rule cycle classification)

↓

Feature Engineering (planned)
↓

Dataset Generation (planned)
↓

Machine Learning and Anomaly Detection (planned)

```

---

# Stage 1

## Metadata Integration

### Objective

The raw measurement files contain only signal identifiers.

The metadata are therefore used to identify

- available experiments,
- available sensors,
- signal UUIDs,
- node hierarchy,
- physical units.

### Input

- signal_data_point_rel.parquet
- signal_data_point_rel_int.parquet
- nodes.parquet
- units.parquet

### Output

Two metadata tables

- UUID signals
- Integer vibration signals

Every signal is represented by

- experiment
- signal UUID
- signal type
- measurement unit
- hierarchical path

---

# Stage 2

## Automatic Signal Discovery

### Objective

Avoid manually specifying signal UUIDs.

Instead, the required sensor is identified automatically using

- experiment name

and

- semantic signal type

Example

Experiment

Versuch1

Signal

Position

↓

Automatically returns

2a87bb14-59e3-40b3-911f-e54db4e3b791

### Advantages

- scalable
- reusable
- independent of experiment-specific UUIDs

---

# Stage 3

## Efficient Measurement Loading

### Objective

The complete industrial dataset contains billions of measurements.

Loading the complete dataset into memory is infeasible.

### Approach

Only the required

- signal
- time interval

are loaded using

PyArrow Dataset

with

- predicate pushdown
- partition pruning

### Output

One pandas DataFrame containing

- timestamp
- value
- signal identifier

---

# Stage 4

## Timestamp Analysis

### Objective

Investigate the temporal properties of the measurements before performing any
further preprocessing.

### Analysis

The following statistics are computed.

- sampling interval
- median sampling interval
- maximum gap
- gap histogram
- gap distribution

### Purpose

Determine

- normal sampling behaviour,
- recording interruptions,
- measurement discontinuities.

### Supervisor Feedback

The industrial supervisors confirmed that the measurements are discrete signals.

Small variations in consecutive timestamps are therefore expected and do not
represent measurement errors.

Large timestamp gaps require further investigation.

### Output

- timestamp statistics
- timestamp histogram
- gap distribution

---

# Stage 5

## Recording Session Detection

### Objective

Separate the complete measurement history into continuous recording sessions.

### Method

Timestamp gaps are analysed.

Large interruptions indicate the beginning of a new recording session.

Current threshold

1 hour

### Output

Recording sessions

Session 1

↓

Session 2

↓

Session 3

...

### Purpose

Prevent later preprocessing steps from operating across recording interruptions.

---

# Stage 6

## Position-Based Cycle Detection

### Objective

Identify individual actuator operating cycles.

### Motivation

During discussions with the industrial supervisors it was confirmed that

the electrical actuator is position-controlled and follows a predefined
position trajectory.

Consequently, the Position signal provides the most reliable reference for
detecting actuator cycles.

### Computed Information

Each detected cycle contains

- start timestamp
- end timestamp
- duration
- sample count
- peak position

### Output

Cycle table

Cycle 1

↓

Cycle 2

↓

Cycle 3

...

---

# Stage 7

## Multi-Sensor Cycle Extraction

### Objective

Extract all available sensor signals belonging to one detected Position cycle.

### Extracted Signals

- Position
- Velocity
- Current
- Pressure
- Temperature
- Counter
- ESP32 vibration (`vibration_x`, `vibration_y`, `vibration_z`)
- Integer vibration

Method

The cycle start and end timestamps generated during Position-based cycle detection define the extraction interval for each cycle.

To avoid repeatedly reading the same Parquet data, the pipeline uses a session-level signal cache.

For each recording session, the pipeline:

loads every configured signal once for the complete session interval,
stores the measurements temporarily in memory,
extracts each cycle by slicing the cached signal data between the cycle start and end timestamps,
releases the cached data after all cycles belonging to the session have been processed.

The extracted cycle boundaries remain unchanged and are still taken directly from the Position-based cycle-detection output.

The default extraction population is read from:

cycles.parquet

In pilot mode:

extract_all_cycles: false

the first max_cycles_to_extract detected cycles are processed.

No cycle is skipped or rejected during extraction because an optional signal is unavailable.

Batch Processing

Cycles are processed in configurable batches.

Batching is used to:

write extracted measurements incrementally,
limit the amount of temporary result data held in memory,
record progress,
support checkpointing and resumed runs.

Batching and session caching have different roles:

the session cache avoids repeated Parquet reads,
batching controls output writing, progress tracking and restart behavior.

### Important Design Decision: Extraction Never Rejects on Missing Signals

Because the ESP32 vibration axes (`vibration_x`, `vibration_y`, `vibration_z`)
are duty-cycled — they record a short (~20 s) burst roughly every 10 minutes
and are silent otherwise — most individual actuator cycles legitimately have
**no** vibration samples. This is expected acquisition behavior, not a data
quality problem.

The pipeline therefore distinguishes explicitly between:

- extraction failure — a technical error prevented the cycle from being processed, such as an unreadable or corrupted data partition,
- signal unavailable — the signal contains zero samples inside the cycle interval,
- signal available but containing non-finite values,
- successfully extracted signal.

These states are recorded using information such as:

extraction_error,
missing_signal,
signal_present,
sample_count,
finite_sample_count,
first_timestamp,
last_timestamp.

An absent optional signal (most commonly vibration) never causes the whole
cycle to be marked as a failed extraction.

### Important Design Decision: No Resampling

The original timestamps and sampling frequencies are preserved.

No interpolation or resampling is performed during preprocessing.


## Backward Compatibility

The extraction function still supports the previous direct-loading method.

When a session cache is provided, measurements are sliced from memory.

When no cache is provided, the function falls back to loading the requested cycle interval directly from the Parquet dataset.

## Standalone Utility: Vibration-Aware Cycle Selection

src/preprocessing/cycle_selection.py provides the optional complete_multisensor_stratified selection mode.

The utility scans candidate cycles in bounded batches, checks the raw measurements within their cycle intervals, detects vibration-recording bursts from timestamp gaps, and selects a deterministic subset with complete multi-sensor coverage.

The selected cycles can be distributed across:

recording sessions,
time strata,
vibration bursts.

This utility is not part of the default pipeline sequence and is not required for:

multi-sensor extraction,
cycle-quality profiling,
validation-rule generation,
dataset validation.

It remains useful for targeted analyses that require cycles with complete vibration coverage.

The default scientific workflow continues to extract, profile and validate the naturally detected cycle population, including cycles without vibration.

## Validation

The updated extraction implementation was checked with the complete automated test suite.

90 / 90 tests passed

The optimization did not change:

cycle detection,
cycle boundaries,
external APIs,
output formats,
checkpoint behavior,
validation logic.
---

# Stage 8

## Cycle Quality Profiling

### Objective

Compute exploratory, cycle-level and signal-level quality metrics for every
extracted cycle and signal.

This stage does **not** reject cycles and does **not** apply any final hard
validation threshold. Its purpose is purely descriptive: it produces the
distributions of quality metrics that will later be used to derive robust,
data-driven validation thresholds. Warnings recorded here are descriptive
only.

The following aspects are analysed per cycle and per signal.

- `sample_count`, `finite_sample_count`, `non_finite_count`
- `missing_signal`, `constant_signal`
- `signal_min`, `signal_max`, `signal_range`, `mean`, `median`,
  `standard_deviation`
- `first_timestamp`, `last_timestamp`, `observed_duration_seconds`
- `median_sampling_interval`, `estimated_sampling_rate`,
  `maximum_timestamp_gap`, `duplicate_timestamp_count`
- `coverage_ratio` (signal coverage relative to the cycle duration)
- `cycle_duration_seconds`, position stroke range (per cycle)

### Output

- `signal_quality_metrics.parquet`: one row per (cycle, signal) with the
  metrics above.
- `cycle_quality_profile.parquet`: one row per cycle aggregating the
  signal-level metrics without any pass/fail decision. Preserves the
  original `cycle_id` and `session_id`.
- `quality_metric_distribution_summary.csv`: per-signal distribution
  summary (mean, std, min, max, and percentiles) of every metric, intended
  as the basis for deriving validation thresholds in the next stage.

---

# Stage 9

## Validation Rule Generation

### Objective

Automatically derive robust, data-driven validation thresholds from the
profiling population produced in Stage 8 — **before** any cycle has been
rejected. This avoids circular validation: rules must come from unfiltered,
profiled data, not from already-validated data.

### Why Profiling Must Precede Validation

If rules were derived after cycles had already been filtered by fixed
thresholds, the reference population used to compute the thresholds would
itself be biased by an earlier, arbitrary cutoff. Generating rules from the
full profiled population first, then freezing them, then validating,
removes this circularity.

### Hard Rules vs. Learned Rules

Two categories of rules are kept strictly separate:

- **Hard rules** are logical/domain constraints that must never be learned
  from the data distribution, e.g. `start_time < end_time`, positive cycle
  duration, mandatory presence of the Position signal, parseable and
  monotonic timestamps, and "a signal with zero finite values cannot be
  valid". These are fixed regardless of what the data looks like.
- **Learned rules** are robust statistical thresholds derived from the
  observed metric distributions using:
  - **median/MAD** (default): `robust_z = 0.6745 * (x - median) / MAD`;
    bounds are `median ∓ (mad_z_limit / 0.6745) * MAD`.
  - **quantile fallback**: used when `MAD == 0` (or too little variation),
    bounds taken from configurable `lower_quantile`/`upper_quantile`.
  - metrics are **skipped** (not assigned a fabricated rule) when the
    reference population is too small (`minimum_reference_count`), the
    metric is constant, or all values are null/non-finite.

### Special Handling of Duty-Cycled Vibration

Because vibration is intentionally duty-cycled, a single global
sample-count threshold across *all* cycles would mix cycles with zero
vibration samples, partial burst overlap, and complete bursts — producing
a meaningless statistic. Every cycle is first classified into:

- `vibration_unavailable` — all three axes have zero samples,
- `vibration_partial` — some but not all axes/coverage present,
- `vibration_complete` — all three axes present, finite, and above a
  data-driven per-axis sample-count threshold.

Vibration-specific learned thresholds are derived **only** from the
`vibration_complete` reference subset, never from the full population.

### Provisional Rules

The reference population's representativeness is recorded explicitly (e.g.
number of sessions covered, cycle count). When the population is limited —
for example only the first pilot cycles from a single session — all
generated rules are marked `provisional=True` rather than being presented
as final, general-purpose thresholds.

### Output

Written to `validation_rule_generation/`:

- `validation_thresholds.json` — every generated rule (hard and learned)
  with signal, metric, method, reference population, reference count,
  median/MAD or quantile bounds, `hard_rule`, `provisional`, and
  `fallback_used` flags. NaN/Infinity never appear; unavailable values are
  serialized as `null`.
- `threshold_derivation_summary.csv` — per-metric derivation details.
- `rule_generation_summary.json` — run-level counts (rules generated,
  metrics skipped, provisional rules, reference cycle count) and the
  representative-population warning.
- `skipped_metrics.csv` — metrics that were not learned, with the reason
  (insufficient data, constant, all-null, excluded by configuration).

---

# Stage 10

## Dataset Validation

### Objective

Apply the frozen thresholds from Stage 9 consistently to every profiled
cycle, producing the final cycle classification. This is the **only** stage
that rejects cycles.

### Final Classes

- `valid_core_cycle` — every configured `core_required` signal satisfies
  both hard and learned rules; vibration is unavailable or incomplete.
  Absence of vibration alone never invalidates a cycle.
- `valid_complete_multisensor_cycle` — all core-required signals are valid
  **and** vibration is `vibration_complete` and passes vibration-specific
  learned rules.
- `invalid_cycle` — one or more core-required signals fail a hard or
  learned rule, a cycle-level hard rule fails, or a technical error
  prevented reliable extraction.

The independent vibration-availability class
(`vibration_unavailable`/`partial`/`complete`) is retained alongside the
final class for every cycle.

### Output

Written to `dataset_validation/`:

- `cycle_validation_results.parquet`, `signal_validation_results.parquet`
- `validation_reason_summary.csv` — reason-code counts (e.g.
  `duration_below_lower_bound`, `position_stroke_out_of_range`,
  `vibration_unavailable`); `vibration_unavailable` is recorded as an
  informational reason but never by itself flips a cycle to
  `invalid_cycle`.
- `validation_summary.json`
- `valid_core_cycles.parquet`, `valid_complete_multisensor_cycles.parquet`,
  `invalid_cycles.parquet` — mutually exclusive by final class.

---

# Stage 11

## Feature Engineering

Status

Partially Implemented

### Objective

Transform the raw sensor measurements into numerical descriptors suitable for
machine learning.

### Stage 10.1 — Cycle Tensor Dataset Generation

Implemented in `src/feature_engineering/cycle_tensor_generation.py` and
integrated into the pipeline as the `feature_engineering` stage
(runs immediately after `dataset_validation`).

Follows the thesis' **padding** methodology, not interpolation: it turns
every `valid_core_cycle` into a fixed-length, ML-ready sample of shape
`target_length x number_of_signals`, then stacks samples into batches of
shape `batch_size x target_length x number_of_signals`, while keeping every
original measured value exactly as extracted:

- reads only `cycle_id` from `valid_core_cycles.parquet` (never re-derives or
  re-checks validity),
- resolves each cycle's `start_time`/`end_time` from the cycle index,
- defines one cycle's length as its number of raw `reference_signal`
  samples (`position` by default),
- derives a single `target_length` up front from the observed distribution
  of cycle lengths across every valid core cycle (`target_length_strategy`:
  `max`, or a configurable `target_length_percentile`, default `99`),
- for every required signal (`position`, `velocity`, `current`, `pressure`,
  `temperature` by default), keeps its own raw, native-rate values and
  pads shorter sequences at the **end** with edge padding (repeats the last
  real value; zero padding is never used) or truncates longer sequences at
  the **end** — no signal is ever resampled onto another signal's timeline,
  preserving signal order,
- builds a padding mask per cycle (`1` = real sample, `0` = padding) derived
  from the reference signal's original length vs. `target_length`,
- logs every truncation and records the truncated sample count,
- skips (and records) any cycle missing a required signal instead of
  aborting the run.

Configurable via the `cycle_tensor_generation` config block:
`required_signals`, `reference_signal` (default `position`),
`target_length_strategy` (`max`/`percentile`, default `percentile`),
`target_length_percentile` (default `99`), `padding_method` (default
`edge`), `truncate_long_cycles` (default `true`), `save_padding_mask`
(default `true`), `output_format` (default `npy`), `cycles_per_file`
(default `64`).

Written to `feature_engineering/`:

- one NumPy tensor file per batch, named e.g.
  `D63_Nr7_Versuch1_cycles_000001_000064.npy`,
- one matching NumPy padding-mask file per batch (when
  `save_padding_mask` is enabled), e.g.
  `D63_Nr7_Versuch1_cycles_000001_000064_mask.npy`,
- `cycle_tensor_metadata.parquet` — one row per written cycle with
  `cycle_id`, `session_id`, `cycle_start`, `cycle_end`,
  `cycle_duration_seconds`, `original_cycle_length`, `target_length`,
  `padded_samples`, `truncated_samples`, and its batch file,
- `skipped_cycles.parquet` — cycles skipped due to a missing required
  signal,
- `cycle_tensor_generation_summary.json` — run-level counts,
- `cycle_length_statistics.json` — minimum/maximum/mean/median/p95/p99
  cycle length, the selected target length, and the number of padded vs.
  truncated cycles.

### Current Progress

A utility is currently available for computing basic descriptive statistics
(e.g., minimum, maximum, mean, standard deviation and RMS) for extracted
signals.

These statistics support exploratory analysis and debugging but are not part
of the formal Feature Engineering stage.

Frequency-domain, cycle-shape and cross-sensor features remain future work.



---

# Stage 12

## Dataset Generation

Status

Planned

### Objective

Create structured machine-learning-ready datasets from the validated actuator
cycles and the engineered features.

The generated datasets will contain:

- feature vectors,
- cycle metadata,
- recording-session information,
- experiment identifiers,
- validation results,
- sensor availability indicators.

The final datasets will subsequently be used for training and evaluating
machine-learning models.

---

# Stage 13

## Machine Learning and Anomaly Detection

Status

Planned

The final dataset will be used to train anomaly detection models capable of
learning the normal behaviour of the actuator.

During inference, every new actuator cycle will pass through the same
preprocessing pipeline before being evaluated by the trained model.

---

# Current Pipeline Status

| Stage | Status |
|--------|--------|
| Metadata Integration | ✅ Completed |
| Automatic Signal Discovery | ✅ Completed |
| Efficient Measurement Loading | ✅ Completed |
| Timestamp Analysis | ✅ Completed |
| Recording Session Detection | ✅ Completed |
| Position Cycle Detection | ✅ Completed |
| Vibration-Aware Cycle Selection | ✅ Completed |
| Multi-Sensor Cycle Extraction | ✅ Completed |
| Cycle Quality Profiling | ✅ Completed |
| Validation Rule Generation | ✅ Completed |
| Dataset Validation | ✅ Completed |
| Feature Engineering | ⏳ Partially Implemented (cycle tensor dataset generation available) |
| Machine Learning Dataset | ⏳ Planned |
| Anomaly Detection | ⏳ Planned |

---

# Future Work

The next implementation milestone is

Gap Context Analysis.

For every significant timestamp gap

- locate the interruption,
- extract all surrounding sensor signals,
- visualize their behaviour,
- determine whether the interruption represents

    - a recording pause,
    - machine shutdown,
    - experiment transition,
    - maintenance,
    - or measurement interruption.

The validated recording sessions will subsequently be used for batch cycle
detection across the complete industrial dataset.
