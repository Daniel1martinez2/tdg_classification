# FWOD Conversion Logic

## Overview

This document describes the conversion pipeline that transforms MIDI drum patterns into the FWOD (Flattened Weighted Onset Distribution) representation used for genre classification.

---

## 1. Conversion Pipeline

The conversion follows three main stages:

```
MIDI File → HV-List → FWOD → Dataset (Excel)
```

### Stage 1: MIDI to HV-List

A MIDI drum file contains timestamped note events with velocity information. The conversion process:

1. **Quantization**: The continuous MIDI timeline is divided into discrete 16th-note steps. Each bar (measure) contains exactly **16 steps**.

2. **Note Extraction**: For each note event, we extract:
   - The drum instrument (MIDI note number)
   - The velocity (force/intensity), normalized to [0, 1]
   - The step position within the bar

3. **Bar Segmentation**: The entire MIDI file is split into individual bars of 16 steps each.

4. **Density Filter**: Bars with fewer than 5 active steps are discarded (too sparse to be meaningful patterns).

**Result**: An HV-List (Hit-Velocity List) — a list of 16 steps, where each step contains pairs of (instrument, velocity).

#### Example HV-List (one bar):
```
Step 0:  [(36, 0.78), (42, 0.55)]   → Kick + Hi-hat
Step 1:  []                          → Silence
Step 2:  [(42, 0.48)]                → Hi-hat only
Step 3:  []                          → Silence
Step 4:  [(38, 0.82), (42, 0.51)]   → Snare + Hi-hat
...
Step 15: [(42, 0.45)]                → Hi-hat only
```

---

### Stage 2: HV-List to FWOD

The FWOD representation flattens the multi-instrument HV-List into a single value per step using **weighted aggregation**.

#### Instrument Categories and Weights

| Category | Instruments | Weight | Rationale |
|----------|-------------|--------|-----------|
| **Low** | Kick, Floor Tom, Low Conga | **3** | Foundation of the rhythm |
| **Mid** | Snare, Clap, Rimshot, Toms | **2** | Backbeat elements |
| **High** | Hi-hat, Cymbals, Shakers | **1** | Time-keeping elements |

#### Flattening Formula

For each step, the FWOD value is calculated as:

```
FWOD[step] = Σ (velocity × weight) for all instruments at that step
```

After processing all 16 steps, the entire vector is **normalized** by dividing by the maximum value, ensuring all values fall within [0, 1].

#### Example Conversion

Given Step 0 with: Kick (velocity=0.78) + Hi-hat (velocity=0.55)

```
FWOD[0] = (0.78 × 3) + (0.55 × 1) = 2.34 + 0.55 = 2.89
```

After normalization (assuming max across all steps is 2.89):
```
FWOD[0] = 2.89 / 2.89 = 1.0
```

**Result**: A 16-dimensional vector representing one bar of drumming.

---

### Stage 3: FWOD to Dataset

Each FWOD vector becomes one row in the dataset with:
- **file**: Source MIDI file path
- **sequence**: Bar index within the file
- **class**: Genre label (from folder name)
- **step_0 to step_15**: The 16 FWOD values

---

## 2. Pattern Aggregation

To capture longer temporal dependencies, multiple consecutive bars can be combined into a single sample.

### Non-Overlapping (Inclusive = False)

Bars are grouped into non-overlapping chunks of size N.

**Example with N=4 patterns:**
```
Bars:    [1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12]
          \_________/     \_________/     \__________/
Sample:       1               2                3
```

- Sample 1 = Bars 1-4
- Sample 2 = Bars 5-8
- Sample 3 = Bars 9-12

**Characteristics:**
- Each bar appears in exactly **one** sample
- Fewer total samples
- No data overlap between samples
- Cleaner for statistical independence

---

### Overlapping (Inclusive = True)

A sliding window of size N moves one step at a time.

**Example with N=4 patterns:**
```
Bars:    [1] [2] [3] [4] [5] [6] [7] [8] [9]
          \_________/
              \_________/
                  \_________/
                      \_________/
                          \_________/
                              \_________/
Sample:       1     2     3     4     5     6
```

- Sample 1 = Bars 1-4
- Sample 2 = Bars 2-5
- Sample 3 = Bars 3-6
- Sample 4 = Bars 4-7
- Sample 5 = Bars 5-8
- Sample 6 = Bars 6-9

**Characteristics:**
- Each bar appears in up to **N** samples
- More total samples (data augmentation effect)
- Samples share data (not statistically independent)
- Captures transitions between patterns

---

### Feature Expansion

When N patterns are combined, the feature vector grows from 16 to **16 × N** dimensions:

| Aggregation | Features | Description |
|-------------|----------|-------------|
| 1 pattern | 16 | Single bar |
| 2 patterns | 32 | Two consecutive bars |
| 4 patterns | 64 | Four consecutive bars |
| 5 patterns | 80 | Five consecutive bars |

The features are simply concatenated:
```
[bar1_step0, bar1_step1, ..., bar1_step15, bar2_step0, ..., barN_step15]
```

---

## 3. Summary

| Stage | Input | Output | Transformation |
|-------|-------|--------|----------------|
| 1 | MIDI file | HV-Lists | Quantize to 16-step bars |
| 2 | HV-List | FWOD (16 values) | Weighted flattening |
| 3 | FWOD vectors | Excel dataset | Add metadata + labels |
| 4 (optional) | Base dataset | Aggregated dataset | Combine N patterns |

### Why FWOD?

1. **Dimensionality Reduction**: From variable multi-instrument data to fixed 16 dimensions
2. **Perceptual Weighting**: Low frequencies (kick) have more impact on perceived rhythm
3. **Normalization**: Makes patterns comparable regardless of original velocity range
4. **Genre Invariance**: Focuses on rhythmic structure rather than specific instruments

### Why Aggregation?

1. **Temporal Context**: Single bars may lack distinguishing characteristics
2. **Pattern Cycles**: Many genres have 2-bar or 4-bar cyclic structures
3. **Feature Richness**: More features can help classifiers find complex patterns
4. **Trade-off**: More features vs. fewer samples (especially with non-overlapping)
