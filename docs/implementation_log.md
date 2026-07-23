# Implementation Log

**Project:**
Predictive Maintenance for Industrial Electric Actuators

**Author:**
Fatemeh Heydari

**Version:**
Preprocessing Pipeline V2.1

**Last Updated:**
2026-07-22

---

# 1. Project Objective

The objective of this work is to develop a scalable and reproducible preprocessing pipeline for large industrial actuator measurement datasets.

The pipeline prepares heterogeneous multi-sensor measurements for machine-learning-based anomaly detection and predictive maintenance.

The preprocessing system is designed to:

* process very large Parquet datasets efficiently,
* limit memory consumption through filtered and batch-based loading,
* identify relevant sensor signals automatically,
* detect independent recording sessions,
* detect actuator operating cycles from the Position signal,
* extract measurements from multiple sensors for each cycle,
* preserve native timestamps and sampling frequencies,
* compute non-rejecting cycle-quality metrics,
* derive robust validation rules from the observed data,
* classify cycles as technically valid or invalid,
* retain physically meaningful unusual behaviour for later anomaly detection.

---

# 2. Current Pipeline Structure

The current preprocessing pipeline contains eleven defined stages. Stages 1–9 are implemented. Feature Engineering and final Dataset Generation are planned.

```text
Raw ERA dataset
    ↓
1. Metadata Integration
    ↓
2. Signal Discovery
    ↓
3. Timestamp Analysis
    ↓
4. Recording Session Detection
    ↓
5. Position-Based Cycle Detection
    ↓
6. Multi-Sensor Cycle Extraction
    ↓
7. Cycle Quality Profiling
    ↓
8. Validation Rule Generation
    ↓
9. Dataset Validation
    ↓
10. Feature Engineering — planned
    ↓
11. Dataset Generation — planned
    ↓
Machine Learning and Anomaly Detection — planned
```

The default pipeline does not reject cycles before quality profiles and validation rules have been generated. This prevents arbitrary early filtering and keeps the rule-generation process traceable. 

---

# 3. Stage 1 — Metadata Integration

## Status

Completed

## Objective

Load and integrate the metadata required to identify and interpret the available measurement signals.

## Input Files

The metadata loader reads:

* `signal_data_point_rel.parquet`
* `signal_data_point_rel_int.parquet`
* `nodes.parquet`
* `units.parquet`

## Implemented Functionality

The metadata module:

* converts UUID values into a consistent representation,
* reconstructs hierarchical node paths,
* connects signals to their units,
* distinguishes UUID-based and integer-based signals,
* generates reusable signal catalogues,
* supports experiment-specific signal searches.

## Main Functions

* `load_metadata()`
* `build_node_paths()`
* `build_uuid_signal_info_from_metadata()`
* `build_int_signal_info_from_metadata()`
* `find_signals()`

## Generated Outputs

* `uuid_signal_catalogue.csv`
* `int_signal_catalogue.csv`

## Validation

The metadata integration successfully identified the relevant Position, Velocity, Current, Pressure, Temperature, Counter and Vibration signals for the investigated experiments.

---

# 4. Stage 2 — Signal Discovery

## Status

Completed

## Objective

Identify all measurement channels associated with the selected experiment and determine the Position signal used as the cycle reference.

## Implemented Functionality

The stage:

* searches the metadata for signals belonging to the configured experiment,
* creates an experiment-specific signal catalogue,
* identifies the configured reference-signal type,
* requires exactly one matching Position reference signal,
* marks the reference channel in the output catalogue.

If no Position signal or more than one matching Position signal is found, the pipeline stops with an explicit error.

## Generated Output

* `selected_signals.csv`

The output contains an `is_reference_signal` marker for the selected Position channel.

---

# 5. Efficient Measurement Loading

## Status

Completed and used across multiple pipeline stages

## Objective

Load only the required parts of the large measurement datasets without reading the complete Parquet files into memory.

## Implemented Functionality

Reusable measurement loaders were implemented using PyArrow Dataset.

The loaders support:

* partition-aware access,
* predicate pushdown,
* signal-ID filtering,
* start- and end-time filtering,
* UUID-based signal loading,
* integer-based signal loading,
* bounded time-window extraction,
* protection against accidental full-table scans.

The loader reads only:

```text
requested signal
+
requested time interval
```

instead of scanning the complete measurement table.

## Validation Example

Complete Position signal:

```text
approximately 358 million rows
```

One-hour Position interval:

```text
71,855 rows
```

The filtered loader substantially reduced memory usage and processing time compared with loading the complete measurement table.

This loader is a shared infrastructure component rather than a separate formal stage in the current pipeline order.

---

# 6. Stage 3 — Timestamp Analysis

## Status

Completed

## Objective

Characterize the timestamp behaviour of the Position reference signal and identify both normal sampling variation and large recording interruptions.

## Computed Metrics

The module calculates:

* total number of samples,
* minimum sampling interval,
* median sampling interval,
* mean sampling interval,
* percentile statistics,
* maximum timestamp gap,
* timestamp-gap frequency distribution.

## Observations

Typical Position sampling interval:

```text
approximately 50 ms
```

Small observed variations:

```text
approximately 43–58 ms
```

Large interruptions ranged from minutes to several days.

The results indicate that the measurements are discrete and approximately periodic rather than perfectly equidistant.

## Generated Outputs

* `statistics.csv`
* `time_gap_histogram.png`
* `time_gap_histogram_under_1_second.png`

## Assessment

Small timestamp variations are expected and should not automatically be treated as data errors.

Large timestamp gaps are used as evidence for separating independent recording periods.

The current orchestrator performs this analysis on the selected Position reference signal. Future analyses may extend the same procedure to additional signals.

---

# 7. Stage 4 — Recording Session Detection

## Status

Completed

## Objective

Divide the complete Position recording into independent, internally continuous recording sessions.

## Current Criterion

A new recording session begins when the time difference between two consecutive Position measurements exceeds:

```text
3600 seconds
```

or:

```text
1 hour
```

The threshold is configurable through:

```yaml
session_gap_seconds: 3600.0
```

## Rationale

Session separation prevents cycle detection from incorrectly connecting measurements across:

* recording interruptions,
* system shutdowns,
* experiment transitions,
* maintenance periods,
* long data-acquisition gaps.

## Current Result

For `Versuch1`, the Position signal was divided into six recording sessions.

## Generated Output

* `sessions.csv`

## Current Assessment

The one-hour threshold is implemented and operationally stable.

However, its scientific justification should be documented using the observed gap distribution or through a sensitivity analysis with alternative thresholds.

---

# 8. Stage 5 — Position-Based Cycle Detection

## Status

Completed

## Objective

Detect individual actuator operating cycles from the Position reference signal.

## Reference-Signal Justification

The actuator is position-controlled and follows a predefined Position trajectory.

Therefore, Position is used as the primary reference for determining cycle boundaries.

## Current Detection Logic

The configured movement threshold is:

```text
Position > 1.0
```

A cycle starts when Position changes from:

```text
Position ≤ 1.0
```

to:

```text
Position > 1.0
```

A cycle ends when Position changes from:

```text
Position > 1.0
```

to:

```text
Position ≤ 1.0
```

Rows with unusable timestamps or Position values are removed during data preparation.

Incomplete cycles at the boundaries of recording sessions are discarded.

Cycle detection is performed separately for each recording session. This prevents cycles from spanning long interruptions.

## Computed Cycle Information

For every detected cycle, the pipeline records:

* experiment,
* recording-session ID,
* global cycle ID,
* Position signal UUID,
* start timestamp,
* end timestamp,
* duration,
* cycle-specific Position characteristics.

## Observations

Typical cycle duration:

```text
approximately 3.1 seconds
```

Typical Position sample count:

```text
approximately 63 samples per cycle
```

Typical maximum Position:

```text
approximately 85
```

The cycles show a highly consistent operating pattern across multiple recording periods.

## Generated Outputs

* `cycles.csv`
* `cycles.parquet`
* `cycle_validation.html`

## Current Limitation

The value:

```text
movement_threshold = 1.0
```

is still provisional.

Its final justification should be based on:

* the Position noise distribution near the retracted state,
* visual inspection,
* comparison of alternative thresholds,
* stability of the detected cycle count,
* stability of the resulting cycle-duration distribution.

---

# 9. Stage 6 — Multi-Sensor Cycle Extraction

## Status

Completed

## Objective

Extract the measurements of all configured sensor signals within the time boundaries of each detected cycle.

## Current Extracted Signals

* Position
* Velocity
* Current
* Pressure
* Temperature
* Counter, where available
* ESP32 Vibration
* Integer-based Vibration
* Vibration X
* Vibration Y
* Vibration Z

The exact extraction population is controlled by:

```yaml
extract_all_cycles: true
```

For the final full-dataset run, all detected cycles are extracted.

When `extract_all_cycles` is false, only the configured number of initial cycles is processed.

## Implemented Functionality

The extraction stage:

* uses the Position cycle start and end timestamps generated by cycle detection,
* groups cycles by recording session,
* loads each configured signal once for the complete session interval,
* stores the loaded signal data temporarily in a session-level cache,
* extracts individual cycle windows by slicing the cached data in memory,
* releases the cache after all cycles of the session have been processed,
* processes cycles in bounded batches,
* writes measurements to Parquet,
* supports resumable processing,
* supports controlled overwrite behaviour,
* preserves missing channels instead of fabricating values,
* records cycle and signal metadata,
* optionally generates visual validation plots.

## Session-Level Signal Caching

The original implementation loaded every signal separately for every detected cycle. Each call reopened the large Parquet dataset and applied a new signal-ID and time-window filter. For approximately 107,555 cycles and about 17 signals, this produced roughly 1.8 million individual Parquet queries.

Profiling of the previous full run showed that Multi-Sensor Cycle Extraction consumed approximately 109,814 seconds of a total runtime of 120,767 seconds, corresponding to about 91% of the complete pipeline runtime.

To remove this bottleneck, a session-level cache was introduced:

```text
Recording session
    ↓
Load each signal once for the full session interval
    ↓
Store signal DataFrames in memory
    ↓
Slice Cycle 1 using its detected start and end timestamps
Slice Cycle 2 using its detected start and end timestamps
...
Slice Cycle N using its detected start and end timestamps
    ↓
Release the session cache
```

The cycle-detection logic remains unchanged. Cycle boundaries continue to come from the Position-based cycle-detection stage. The optimization changes only the data-access strategy used during multi-sensor extraction.

## Added Components

* `src/utils/session_signal_cache.py`
  * `SessionSignalCache` Stores all measurements of one recording session in memory after loading them once. Instead of reading the same data from the Parquet files for every cycle, the pipeline reuses the cached data for all cycles within that session.
  * `SessionSignalCacheManager` Manages the caches automatically. It creates a cache when a new recording session starts, provides it to the extraction process, and removes it from memory after all cycles of that session have been processed.
* `extract_cycle_measurements()` Was extended with an optional cache parameter. If a cache is available, measurements are taken directly from memory. Otherwise, the function uses the previous method of loading data from the Parquet files. This keeps the function compatible with existing code.
* `batch_extraction.py` Was updated to use the session cache during cycle extraction. Instead of loading the same signal repeatedly for every cycle, all cycles belonging to the same recording session share the same cached data.
* `batched_measurement_writer.py` Was updated to create and pass the cache manager during extraction. It also supports resumed runs by creating caches only for the remaining recording sessions that still need to be processed.

## Memory Behaviour

Each configured signal is loaded fully for one recording session. No streaming or chunking is currently used inside a session. Peak memory therefore depends on the largest individual session rather than on the complete dataset. The cache is released before processing later sessions. Available server memory is currently sufficient for this approach.

## Verification

The refactoring preserved the existing public interfaces, cycle-detection results, output structure, checkpointing behaviour and downstream pipeline stages.

The complete automated test suite passed after the change:

```text
90 / 90 tests passed
```

The tests include integration tests that generate Parquet fixtures and exercise the multi-sensor extraction stage end to end. 

## Sampling Strategy

Native timestamps and native sampling frequencies are preserved.

No interpolation or resampling is performed during extraction.

This is necessary because the sensors have substantially different acquisition frequencies.

## Example Sample Counts per Cycle

Position:

```text
approximately 62–63 samples
```

Pressure:

```text
approximately 51 samples
```

Temperature:

```text
approximately 3 samples
```

ESP32 Vibration:

```text
approximately 1,269 samples
```

## Visual Validation Subset

A small block of structurally suitable cycles may be selected for HTML inspection.

The visual subset checks:

* availability of required signals,
* minimum sample count per signal,
* consecutive cycles from the same session.

This selection is used only for human-readable validation plots.

It does not filter the full extraction population and does not determine the final validity of the dataset.

## Generated Outputs

* cycle-index Parquet file,
* partitioned measurement batches,
* signal-window summary,
* extraction checkpoint,
* optional multi-sensor validation HTML files.

## Current Limitation

Some channels are unavailable during specific recording periods.

These missing channels are retained explicitly and evaluated in the later quality-profiling and validation stages.

---

# 10. Stage 7 — Cycle Quality Profiling

## Status

Completed

## Objective

Compute descriptive quality metrics for every extracted cycle without making an early keep/reject decision.

## Methodological Role

This stage is intentionally non-rejecting.

It describes the observed quality and structure of each cycle and provides the statistical basis for generating validation rules.

## Signal-Level Metrics

Depending on signal availability and signal role, the profiling stage computes metrics including:

* sample count,
* finite sample count,
* estimated sampling rate,
* maximum timestamp gap,
* signal range,
* standard deviation,
* cycle coverage ratio,
* signal presence.

## Cycle-Level Metrics

The cycle-quality profile includes metrics such as:

* cycle duration,
* Position stroke range,
* interval ordering,
* signal availability,
* multi-sensor completeness.

## Processing Strategy

Quality profiling is performed in batches to support large datasets.

The profiling population consists of all cycles that were actually extracted.

Therefore:

```text
extract_all_cycles = true
```

causes the complete extracted cycle population to be profiled.

## Generated Outputs

* `signal_quality_metrics.parquet`
* `cycle_quality_profile.parquet`
* `quality_metric_distribution_summary.csv`

## Important Distinction

Cycle Quality Profiling does not classify cycles as valid or invalid.

Its output is descriptive and is passed to Validation Rule Generation.

---

# 11. Stage 8 — Validation Rule Generation

## Status

Completed

## Objective

Derive robust and reproducible validation thresholds from the cycle-quality distributions.

## Input Data

* signal-level quality metrics,
* cycle-level quality profiles,
* configured signal roles,
* rule-generation parameters.

## Statistical Method

The primary method is:

```text
Median + Median Absolute Deviation
```

or:

```text
Median + MAD
```

The current configurable MAD limit is:

```yaml
mad_z_limit: 3.5
```

When MAD is zero, unstable or unsuitable, the rule generator falls back to quantile-based thresholds.

Current fallback quantiles:

```yaml
lower_quantile: 0.01
upper_quantile: 0.99
```

## Rule Types

The implementation separates:

### Learned statistical rules

Examples:

* acceptable sample-count interval,
* acceptable estimated sampling rate,
* acceptable timestamp gap,
* acceptable coverage range,
* acceptable cycle-duration range.

### Hard logical rules

Examples:

* required signal must be present,
* timestamps must be correctly ordered,
* duration must be positive,
* required values must be finite.

## Reference Population

The current final-run configuration uses:

```yaml
reference_population: all_profiled_cycles
```

This means the rules characterize the full profiled population.

They should not automatically be described as confirmed healthy-behaviour limits unless the reference population has independently been verified as healthy.

## Rule Freezing

The generated rules are frozen before Dataset Validation.

Dataset Validation reads the generated thresholds but does not modify or regenerate them.

## Generated Outputs

* `validation_thresholds.json`
* `threshold_derivation_summary.csv`
* `rule_generation_summary.json`
* `skipped_metrics.csv`

## Current Methodological Limitation

Rules learned from all profiled cycles may include both normal and degraded behaviour.

The current rules therefore represent technical data-quality boundaries for the observed population rather than definitive health-state boundaries.

---

# 12. Stage 9 — Dataset Validation

## Status

Completed

## Objective

Apply the frozen validation rules to every profiled cycle and classify its technical usability.

## Inputs

* `signal_quality_metrics.parquet`
* `cycle_quality_profile.parquet`
* frozen validation thresholds,
* configured signal roles,
* validation configuration.

## Validation Logic

The stage evaluates:

* required signal presence,
* finite-value requirements,
* timestamp consistency,
* sample-count rules,
* sampling behaviour,
* coverage ratio,
* signal-quality thresholds,
* cycle-level structural rules.

## Signal Roles

### Cycle Reference

* Position

### Core Required Signals

* Position
* Velocity
* Current
* Pressure
* Temperature

### Optional Duty-Cycled Signals

* Vibration X
* Vibration Y
* Vibration Z

Missing optional vibration measurements are recorded separately and do not necessarily invalidate the core cycle.

## Validation Outputs

The stage separates cycles into:

* valid core cycles,
* valid complete multi-sensor cycles,
* invalid cycles,
* cycles with unavailable vibration,
* cycles with partial vibration,
* cycles with complete vibration.

## Generated Outputs

* `cycle_validation_results.parquet`
* `signal_validation_results.parquet`
* `validation_reason_summary.csv`
* `validation_summary.json`
* `valid_core_cycles.parquet`
* `valid_complete_multisensor_cycles.parquet`
* `invalid_cycles.parquet`

## Important Methodological Distinction

Dataset Validation determines technical data usability.

It should reject cycles with structural or measurement-quality problems, such as:

* missing core signals,
* invalid values,
* invalid timestamps,
* insufficient data coverage,
* impossible cycle intervals.

However, physically unusual but technically valid signal behaviour should normally remain available for anomaly detection.

An unusual current profile, Position range or vibration pattern may represent degradation rather than bad data.

---

# 13. Implemented Validation Summary

The following components have been implemented and tested:

* Metadata loading
* Metadata integration
* UUID and integer signal catalogue generation
* Automatic experiment-specific signal discovery
* Efficient time-window measurement loading
* Timestamp-gap analysis
* Recording-session detection
* Position-based cycle detection
* Batch-based multi-sensor extraction
* Native timestamp preservation
* Native sampling-rate preservation
* Cycle Quality Profiling
* Validation Rule Generation
* Dataset Validation
* Valid/invalid cycle separation
* Validation reason reporting

---

# 14. Key Dataset Observations

The preprocessing pipeline revealed several important properties of the industrial actuator dataset.

## 14.1 Discrete Measurements

Measurements are discrete and approximately periodic.

Small timestamp variations are expected and do not automatically indicate errors.

## 14.2 Different Sampling Frequencies

Different sensors operate at different acquisition frequencies.

This prevents direct sample-by-sample comparison before an explicit alignment strategy is defined.

## 14.3 Independent Recording Periods

Large timestamp gaps separate independent recording periods.

Cycle detection must therefore be performed within individual recording sessions.

## 14.4 Stable Position Cycles

The Position signal shows highly consistent cycle durations, sample counts and stroke ranges across recording periods.

## 14.5 Variable Sensor Availability

Sensor availability differs across recording sessions and experiments.

In particular, vibration channels may be:

* unavailable,
* partially available,
* fully available.

## 14.6 Quality and Health Are Different Concepts

A technically valid cycle may still contain physically unusual behaviour.

Technical validation and anomaly detection must therefore remain separate stages.

---

# 15. Current Limitations and Open Questions

## Unidentified Signals

The physical meaning of the following channels remains unclear:

* `Sensor_1`
* `Sensor_2`
* `Sensor_3`
* `Sensor_4`

These signals should not be used in final modelling until their meaning and units are clarified.

## Velocity Scaling

The physical scaling and unit interpretation of the Velocity signal still require verification.

## Movement Threshold

The current Position movement threshold of `1.0` is provisional and requires empirical justification.

## Session-Gap Threshold

The one-hour session-gap threshold is implemented but should be supported by documented analysis.

## Optional Vibration Availability

Vibration measurements are not continuously available across all recording periods.

The downstream dataset design must represent this availability explicitly.

## Reference Population

The current validation rules are generated from all profiled cycles.

A future experiment should examine whether rules generated from an explicitly healthy reference period produce more appropriate boundaries.

## Validation Severity

Some learned statistical rules may be better treated as warnings rather than strict rejection rules.

This is particularly relevant for:

* signal range,
* standard deviation,
* unusual current behaviour,
* unusual vibration behaviour.

---

# 16. Planned Stage 10 — Feature Engineering

## Status

Planned

## Objective

Transform technically valid cycles into machine-learning-ready representations.

## Planned Feature Groups

### Time-Domain Features

* mean,
* median,
* minimum,
* maximum,
* standard deviation,
* variance,
* root mean square,
* peak-to-peak range,
* skewness,
* kurtosis.

### Cycle-Shape Features

* cycle duration,
* opening duration,
* closing duration,
* maximum Position,
* Position stroke,
* transition slopes,
* dwell times,
* area under the signal curve.

### Current and Pressure Features

* peak Current,
* average Current,
* Current integral,
* Pressure minimum and maximum,
* Pressure drop,
* Pressure recovery behaviour.

### Vibration Features

* RMS vibration,
* peak vibration,
* crest factor,
* frequency-domain energy,
* dominant frequencies,
* spectral entropy,
* band-specific FFT features.

### Temperature Features

* cycle temperature,
* temperature trend,
* temperature increase over operating time.

## Required Design Decision

Signals have different native sampling frequencies.

Feature Engineering must therefore define whether each feature requires:

* native-rate calculation,
* resampling,
* interpolation,
* fixed-length cycle normalization,
* frequency-domain processing.

## Stage 10.1 — Cycle Tensor Dataset Generation (Implemented)

The fixed-length-cycle-normalization option above has been implemented as
`src/feature_engineering/cycle_tensor_generation.py`, wired into the
pipeline as the `feature_engineering` stage running directly after
`dataset_validation`. It follows the thesis' **padding** methodology, not
interpolation/resampling, so every original measured value is preserved
exactly.

For every `valid_core_cycle` (read by `cycle_id` only from
`valid_core_cycles.parquet`):

* One cycle's length is defined as its number of raw `reference_signal`
  samples (`position` by default -- the signal used to detect the cycle in
  the first place).
* A single `target_length` is derived once, up front, from the observed
  distribution of cycle lengths across every valid core cycle:
  `target_length_strategy="max"` uses the longest observed cycle, or
  `target_length_strategy="percentile"` (default) uses
  `target_length_percentile` (default `99`). The full distribution
  (minimum, maximum, mean, median, p95, p99, selected target length, number
  of padded/truncated cycles) is written to `cycle_length_statistics.json`.
* For every required signal (`position`, `velocity`, `current`, `pressure`,
  `temperature` by default), the raw, native-rate value sequence (sorted by
  timestamp, otherwise untouched) is:
  - **edge-padded at the end** (the last real value is repeated -- zero
    padding is never used) when shorter than `target_length`, or
  - **truncated at the end** when longer than `target_length` (logged, and
    the truncated sample count is recorded in the metadata).
* A padding mask (`1` = real sample, `0` = padding) is derived from the
  reference signal's original length vs. `target_length`.

This produces one `target_length x number_of_signals` matrix per cycle,
stacked into `batch_size x target_length x number_of_signals` NumPy tensor
files (one per batch of `cycles_per_file` cycles), each with a matching
`batch_size x target_length` NumPy padding-mask file. Metadata
(`cycle_tensor_metadata.parquet`) records `cycle_id`, `session_id`,
`cycle_start`, `cycle_end`, `cycle_duration_seconds`,
`original_cycle_length`, `target_length`, `padded_samples`,
`truncated_samples`, and the batch file; `skipped_cycles.parquet` records
cycles missing a required signal. All of `required_signals`,
`reference_signal`, `target_length_strategy`, `target_length_percentile`,
`padding_method` (only `edge` is currently supported),
`truncate_long_cycles`, `save_padding_mask`, `output_format`, and
`cycles_per_file` are configurable via the `cycle_tensor_generation` config
block.

The remaining feature groups above (time-domain summary statistics,
cycle-shape, current/pressure, vibration, temperature features) remain
future work.

---

# 17. Planned Stage 11 — Dataset Generation

## Status

Planned

## Objective

Create the final structured dataset used for model development and evaluation.

## Planned Outputs

Potential datasets include:

* one row per cycle with engineered features,
* fixed-length multi-sensor cycle tensors,
* separate core-signal and vibration-enhanced datasets,
* training, validation and test partitions,
* cycle-to-session and cycle-to-experiment metadata,
* data-quality and vibration-availability indicators.

## Dataset-Splitting Requirement

The final split must avoid information leakage.

Cycles from the same recording period should not be randomly distributed across training and test datasets without considering temporal and session dependencies.

---

# 18. Planned Machine-Learning Work

After preprocessing and dataset generation, the planned modelling workflow is:

```text
Technically valid cycles
    ↓
Feature Engineering or Cycle Tensor Generation
    ↓
Feature Scaling and Normalization
    ↓
Healthy Reference Selection
    ↓
Model Training
    ↓
Model Validation
    ↓
Anomaly-Score Generation
    ↓
Temporal Degradation Analysis
    ↓
Model Evaluation
```

Potential approaches include:

* statistical baseline methods,
* Isolation Forest,
* One-Class SVM,
* Local Outlier Factor,
* Autoencoders,
* sequence autoencoders,
* convolutional autoencoders,
* time-series representation learning.

---

# 19. Next Development Steps

1. Complete and verify the full-dataset preprocessing run.

2. Compare the following counts:

```text
detected cycles
extracted cycles
profiled cycles
validated cycles
valid core cycles
invalid cycles
```

3. Review `validation_reason_summary.csv` and identify the most common rejection reasons.

4. Verify that final rules are not unexpectedly marked as provisional.

5. Perform sensitivity analysis for the Position movement threshold.

6. Document the empirical justification for the one-hour session-gap threshold.

7. Verify the physical scaling and unit of the Velocity signal.

8. Distinguish strict rejection rules from warning-only behavioural rules.

9. Define the signal-alignment and fixed-length cycle representation strategy.

10. Implement Feature Engineering.

11. Implement final Dataset Generation.

12. Select the healthy training reference population.

13. Train baseline anomaly-detection models.

14. Compare model performance across signal combinations.

15. Evaluate whether anomaly scores increase over actuator lifetime.

---

# 20. Current Project Status

Stages 1–9 of the preprocessing pipeline are implemented.

The current final workflow is:

```text
Metadata Integration
→ Signal Discovery
→ Timestamp Analysis
→ Session Detection
→ Position Cycle Detection
→ Full Multi-Sensor Extraction
→ Cycle Quality Profiling
→ Validation Rule Generation
→ Dataset Validation
→ Cycle Tensor Dataset Generation (Stage 10.1)
```

The pipeline can now process the complete detected cycle population, derive frozen technical validation rules, create explicit valid and invalid cycle datasets, and generate a fixed-length `cycles x time_steps x signals` tensor dataset from every valid core cycle.

The next major development phase is the remaining feature groups (cycle-shape, frequency-domain, cross-sensor) and the downstream machine-learning dataset generation stage.
