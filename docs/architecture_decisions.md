# Architecture Decision Records (ADR)

Project:
Predictive Maintenance for Industrial Electric Actuators

Author:
Fatemeh Heydari

Last Update:
2026-07-10

---

# Purpose

This document records all important architectural and preprocessing decisions
taken during the development of the predictive maintenance pipeline.

Instead of documenting only the final implementation, every important design
decision is justified together with the considered alternatives and the reasons
for selecting the final solution.

This document serves as a reference for future development and provides the
technical rationale behind the preprocessing pipeline described in the master
thesis.

---

# ADR-001

## Title

Modular Project Architecture

## Date

2026-07-09

## Problem

The project contains multiple processing stages including metadata handling,
measurement loading, preprocessing, visualization, feature engineering,
machine learning, and evaluation.

Implementing the complete workflow inside one Python script would quickly become
difficult to maintain.

## Options

### Option 1

Single Python script.

Advantages

- Simple initial implementation.

Disadvantages

- Difficult to maintain.
- Difficult to test.
- Poor scalability.
- High coupling.

---

### Option 2

Modular project architecture.

Advantages

- Separation of responsibilities.
- Reusable components.
- Easier testing.
- Better maintainability.
- Easier extension.

Disadvantages

- Slightly higher implementation effort.

---

## Decision

Option 2 was selected.

The project follows a modular architecture in which every preprocessing step is
implemented as an independent reusable module.

## Status

Accepted.

---

# ADR-002

## Title

Metadata-Driven Signal Discovery

## Date

2026-07-09

## Problem

Every experiment contains different UUIDs for Position, Velocity, Current,
Temperature, Pressure and other signals.

Hard-coding these UUIDs would require code modifications whenever additional
experiments are added.

## Options

### Option 1

Store UUIDs manually inside the source code.

Advantages

- Very simple implementation.

Disadvantages

- Poor scalability.
- High maintenance effort.
- Error-prone.

---

### Option 2

Determine signal UUIDs dynamically from the metadata.

Advantages

- Fully automatic.
- Supports additional experiments.
- No code modifications required.
- Scalable.

Disadvantages

- Requires metadata processing.

---

## Decision

Signal UUIDs are determined dynamically from the metadata.

The preprocessing pipeline identifies the required signal using

- experiment name
- semantic signal type

instead of manually specifying UUIDs.

## Status

Accepted.

---

# ADR-003

## Title

Separate Processing of UUID and Integer Signals

## Date

2026-07-09

## Problem

The dataset stores

- Position
- Velocity
- Current
- Pressure
- Temperature

inside

signal_data_point.parquet

while vibration measurements are stored separately inside

vibration.parquet.

The two datasets use different identifier systems.

## Decision

Maintain two independent loading pipelines.

UUID signals are processed separately from integer vibration signals.

## Reason

The datasets differ regarding

- identifiers
- storage layout
- sampling frequency

Separate processing simplifies the implementation and avoids incorrect joins.

## Status

Accepted.

---

# ADR-004

## Title

Memory-Efficient Measurement Loading

## Date

2026-07-09

## Problem

The measurement dataset contains several hundred gigabytes and billions of
measurements.

Loading the complete dataset into memory is infeasible.

## Options

### Option 1

Load the complete parquet dataset.

Advantages

Simple implementation.

Disadvantages

- Very high memory usage.
- Long execution time.
- Poor scalability.

---

### Option 2

Selective loading using PyArrow Dataset.

Advantages

- Predicate pushdown.
- Partition pruning.
- Time filtering.
- Low memory usage.

Disadvantages

Slightly more complex implementation.

## Decision

Use PyArrow Dataset together with

- signal UUID
- start timestamp
- end timestamp

to load only the required measurements.

## Validation

Loading one complete Position signal produced approximately

358 million rows.

Loading one hour returned only

71,855 rows.

The approach significantly reduced memory consumption.

## Status

Accepted.

---

# ADR-005

## Title

Timestamp Analysis Before Session Detection

## Date

2026-07-10

## Problem

Large timestamp gaps exist inside the measurement history.

Choosing a session threshold without analysing the timestamps could incorrectly
split continuous recordings.

## Decision

Timestamp behaviour is analysed before recording sessions are detected.

The analysis includes

- sampling interval distribution
- gap distribution
- histogram
- statistics
- investigation of large gaps

## Supervisor Feedback

The supervisors confirmed that the measurements are discrete signals.

Small variations in consecutive timestamps are therefore expected.

Large gaps should be investigated individually.

## Status

Accepted.

---

# ADR-006

## Title

Recording Session Detection

## Date

2026-07-10

## Problem

The complete measurement history contains multiple recording periods separated
by long interruptions.

Cycle detection should not operate across recording interruptions.

## Decision

Recording sessions are identified from large timestamp gaps.

The current implementation uses

1 hour

as the session threshold based on the empirical timestamp analysis.

## Status

Accepted.

---

# ADR-007

## Title

Position-Based Cycle Detection

## Date

2026-07-10

## Problem

The actuator operates continuously.

Machine learning requires individual operating cycles rather than one continuous
measurement.

## Supervisor Feedback

The industrial supervisors confirmed that

the electrical actuator is position-controlled and follows a predefined
position trajectory.

## Decision

Use the Position signal as the reference signal for cycle detection.

The Position signal defines

- cycle start
- cycle end

The remaining sensor signals are extracted afterwards using the same time
interval.

## Status

Accepted.

---

# ADR-008

## Title

Native Sampling Preservation

## Date

2026-07-10

## Problem

Different sensors use different sampling frequencies.

Examples

Position

≈62 samples

Temperature

≈3 samples

ESP32 vibration

≈1269 samples

per actuator cycle.

## Options

### Option 1

Resample every signal immediately.

### Option 2

Preserve original timestamps.

## Decision

Keep every signal on its native sampling frequency during preprocessing.

Synchronization will be evaluated during feature engineering.

## Status

Accepted.

---

# ADR-009

## Title

Gap Context Analysis

## Date

2026-07-10

## Problem

Large timestamp gaps indicate interruptions.

The underlying cause is unknown.

## Decision

Every significant timestamp gap will be analysed together with the surrounding
sensor behaviour.

The analysis will include

- Position
- Velocity
- Current
- Pressure
- Temperature
- Vibration

before and after the interruption.

## Goal

Determine whether the interruption represents

- recording pause
- machine shutdown
- experiment transition
- maintenance
- acquisition problem

## Status

Planned.

---

# ADR-010

## Title

Cycle-Based Multi-Sensor Representation

## Date

2026-07-10

## Problem

Machine learning requires a consistent representation of actuator behaviour.

## Decision

Each training sample represents exactly one actuator cycle.

Every sample contains all available sensor signals extracted within the detected
cycle boundaries.

## Benefits

- Comparable observations.
- Physical consistency.
- Supports anomaly detection.
- Independent of experiment length.

## Status

Accepted.

---
# ADR-011

## Title

Non-Rejecting Multi-Sensor Extraction

# Date

2026-07-18

# Problem

The detected actuator cycles do not always contain measurements for every sensor.
Some sensors, particularly the ESP32 vibration signals, are recorded intermittently
as part of the experimental acquisition strategy.

Rejecting cycles during extraction because one sensor is unavailable would
discard valid actuator behaviour before its quality can be analysed.

# Options
## Option 1

Reject cycles during extraction whenever one or more required sensors are missing.

# Advantages

Produces a homogeneous dataset immediately.

# Disadvantages

Valid actuator cycles may be discarded.
Data quality cannot be analysed before rejection.
Assumes validation criteria before analysing the data.

## Option 2

Extract every detected cycle and preserve the availability of every signal.

# Advantages

Preserves all available information.
Enables objective quality analysis.
Separates extraction from validation.
Supports automatic threshold generation.

# Disadvantages

Some extracted cycles contain incomplete sensor information.
Decision

Option 2 was selected.

Multi-sensor extraction stores every detected cycle together with all available
sensor measurements.

Missing signals are explicitly represented in the extraction metadata rather than
being treated as extraction failures.

Only technical errors are considered extraction failures.

# Status

Accepted.

--- 
# ADR-012

## Title

Data-Driven Validation Rule Generation

# Date

2026-07-18

# Problem

Fixed validation thresholds such as minimum sample counts or acceptable cycle
durations are difficult to generalize across experiments and may not reflect the
actual measurement characteristics.

The preprocessing pipeline should automatically adapt its validation criteria to
the analysed dataset.

# Options
## Option 1

Define validation thresholds manually.

# Advantages

Simple implementation.
Easy to understand.

# Disadvantages

Dataset dependent.
Difficult to maintain.
Requires manual tuning.
Poor scalability.

## Option 2

Generate validation rules automatically from the profiled data.

# Advantages

Dataset specific.
Reproducible.
Scalable.
Robust against changing experiments.
Supports automated preprocessing.

# Disadvantages

Requires an additional preprocessing stage.

# Decision

Option 2 was selected.

Cycle quality profiling is used to derive validation rules automatically.

Depending on the metric, robust statistical methods such as

quantiles
median
median absolute deviation (MAD)

are used to estimate acceptable value ranges.

The generated rules are stored and subsequently applied during dataset
validation.

# Status

Accepted.

---

# ADR-013

## Title

Separation of Profiling and Dataset Validation

# Date

2026-07-18

# Problem

If cycles are validated before analysing their statistical properties, the
validation criteria become arbitrary and difficult to justify scientifically.

The preprocessing workflow should distinguish between descriptive analysis and
quality assessment.

# Decision

Cycle quality profiling and dataset validation are implemented as two independent
processing stages.

The pipeline follows the sequence

Cycle Detection

↓

Multi-Sensor Extraction

↓

Cycle Quality Profiling

↓

Validation Rule Generation

↓

Dataset Validation

Profiling performs descriptive statistical analysis only.

It does not reject cycles.

Validation rules are generated afterwards and applied consistently to all
profiled cycles.

# Benefits
Scientifically justified preprocessing.
Fully reproducible validation.
Clear separation between observation and decision making.
Eases future adaptation to new datasets.

# Status

Accepted.

---

# ADR-014

## Title

Classification of Core and Optional Sensor Signals

# Date

2026-07-18

# Problem

Not all sensors follow the same acquisition strategy.

Core actuator sensors are continuously recorded, whereas vibration measurements
are intentionally duty-cycled.

Treating every signal as mandatory would incorrectly classify otherwise valid
cycles as invalid.

# Decision

Signals are grouped according to their functional role within the preprocessing
pipeline.

The pipeline distinguishes between

cycle reference signals
core required signals
optional or duty-cycled signals

Position is used as the cycle reference signal.

Core actuator signals are required for validating actuator behaviour.

Duty-cycled signals, such as vibration, are evaluated independently and do not
invalidate an otherwise valid actuator cycle solely because they are absent.

# Benefits
Reflects the experimental acquisition strategy.
Prevents unnecessary data loss.
Enables different validation strategies for different sensor groups.
Improves scalability for future experiments.

# Status

Accepted.