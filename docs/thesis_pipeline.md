# Preprocessing Pipeline Documentation

Project:
Predictive Maintenance for Industrial Electric Actuators

Author:
Fatemeh Heydari

Version:
1.0

Last Update:
2026-07-10

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

### Method

The Position cycle timestamps define the extraction interval.

Every remaining sensor is loaded only inside this interval.

The **default extraction population is read directly from the Position-based
cycle-detection output (`cycles.parquet`)**. In pilot mode
(`extract_all_cycles: false`) the first `max_cycles_to_extract` candidate
cycles are extracted, exactly as detected — no cycle is skipped or rejected
because a signal is missing.

### Important Design Decision: Extraction Never Rejects on Missing Signals

Because the ESP32 vibration axes (`vibration_x`, `vibration_y`, `vibration_z`)
are duty-cycled — they record a short (~20 s) burst roughly every 10 minutes
and are silent otherwise — most individual actuator cycles legitimately have
**no** vibration samples. This is expected acquisition behavior, not a data
quality problem.

The pipeline therefore distinguishes explicitly between:

- **extraction failure** — a real technical error prevented the cycle from
  being processed (e.g. a corrupt partition, an unreadable file). Recorded
  via an `extraction_error` flag and counted in
  `multi_sensor_extraction.failed_cycles`.
- **signal unavailable** — the signal has zero samples inside the cycle
  window. Recorded via `missing_signal` / `signal_present=False`. This is
  **not** an extraction failure.
- **signal available but empty after filtering**, **signal available with
  non-finite values**, and **successfully extracted signal** — all recorded
  explicitly via `sample_count`, `finite_sample_count`,
  `first_timestamp`/`last_timestamp` per cycle/signal.

An absent optional signal (most commonly vibration) never causes the whole
cycle to be marked as a failed extraction.

### Important Design Decision: No Resampling

The original timestamps and sampling frequencies are preserved.

No interpolation or resampling is performed during preprocessing.

### Standalone Utility: Vibration-Aware Cycle Selection

`src/preprocessing/cycle_selection.py` implements an optional, standalone
sampling utility (`complete_multisensor_stratified` mode) that scans
`cycles.parquet` in bounded batches, checks the actual raw measurements
inside each candidate cycle's own time window, detects vibration-recording
bursts from real inter-sample gaps, and picks a deterministic subset of
cycles that are guaranteed to contain complete multisensor (including
vibration) coverage, spread across sessions, time strata, and vibration
bursts.

This utility is **not** part of the default `PipelineStage` sequence and is
not required by `multi_sensor_extraction`, `cycle_quality_profiling`,
`validation_rule_generation`, or `dataset_validation`. It remains useful for
targeted studies that specifically need a pilot subset with complete
vibration coverage (e.g. vibration-focused analyses), but the default
scientific workflow profiles and validates the naturally detected cycle
population — including cycles without vibration — rather than pre-filtering
for vibration completeness.

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

Planned

### Objective

Transform the raw sensor measurements into numerical descriptors suitable for
machine learning.

### Currently Implemented

### Current Progress

A utility is currently available for computing basic descriptive statistics
(e.g., minimum, maximum, mean, standard deviation and RMS) for extracted
signals.

These statistics support exploratory analysis and debugging but are not part
of the formal Feature Engineering stage.

The complete Feature Engineering stage, including frequency-domain,
cycle-shape and cross-sensor features, remains future work.



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
| Feature Engineering | ⏳ Planned (partial time-domain utility available) |
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
