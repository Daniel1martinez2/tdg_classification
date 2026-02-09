# Project Structure - TDG Classification

## General Objective

This project aims to **classify drum patterns by music genre** using machine learning. It takes MIDI drum patterns from the Magenta Groove MIDI Dataset and converts them into a custom representation called **FWOD** (Flattened Weighted Onset Distribution).

The research goal is to improve upon a previous paper ([arxiv.org/pdf/2407.17536](https://arxiv.org/pdf/2407.17536)) that only classified 4 genres (funk, jazz, latin, rock). This project expands classification to **20 genres** using the FWOD representation, achieving **80% accuracy** with a KNN classifier.

**In simple terms:** Given a MIDI drum loop, the trained model predicts whether it sounds like jazz, rock, funk, latin, hiphop, etc., based on its rhythmic pattern and velocity characteristics.

---

## Directory Tree

```
tdg_classification/
│
├── Notebooks (Code)
│   ├── PDG Magenta to FWOD.ipynb
│   ├── classification_data_creation.ipynb
│   ├── classification_task.ipynb
│   ├── models_pipeline_test.ipynb        ← MAIN RESULTS
│   ├── multiple_pattern_db.ipynb
│   └── main.ipynb
│
├── Data Files
│   ├── fwod_representations.xlsx
│   ├── fwod_representations_clean.xlsx   ← PRIMARY DATASET
│   ├── data_pattern_2.xlsx
│   ├── data_pattern_2_inclusive.xlsx
│   ├── data_pattern_4_inclusive.xlsx
│   ├── data_pattern_5_inclusive.xlsx
│   ├── e1_all_hvs.pkl
│   └── Model_performance.xlsx
│
├── Configuration
│   ├── .gitignore
│   └── CLAUDE.md
│
└── External Data (not in repo - gitignored)
    ├── /dataset
    ├── /magenta midi
    └── /tap to drum taps
```

---

## File Descriptions

### Notebooks (Processing Pipeline)

| File | Purpose | Run Order |
|------|---------|-----------|
| `PDG Magenta to FWOD.ipynb` | **Initial prototype**. Contains the core functions for MIDI parsing (`midifile2hv_list`) and FWOD conversion (`flatten_hv_list`). Documents the theoretical approach. | Reference |
| `classification_data_creation.ipynb` | **Data generation pipeline**. Parses all MIDI files from the Magenta dataset, converts them to FWOD representation, and exports to Excel. Creates the primary dataset. | 1st |
| `multiple_pattern_db.ipynb` | **Pattern aggregation**. Combines multiple consecutive patterns (2, 4, 5 bars) into single rows with more features. Creates overlapping and non-overlapping variants. | 2nd (optional) |
| `classification_task.ipynb` | **Basic experiments**. Initial classification tests using Random Forest, SVM, and XGBoost. Exploratory analysis. | 3rd |
| `models_pipeline_test.ipynb` | **Comprehensive ML pipeline**. Tests all algorithms (KNN, SVM, XGBoost, LightGBM, LSTM, RNN) with Bayesian hyperparameter tuning. **This file produces the final results.** | 4th |
| `main.ipynb` | **Tap-to-Drum experiments**. Contains TTD (Tap-to-Drum) experiment data and analysis. Secondary research track. | Separate |

### Data Files

| File | Description |
|------|-------------|
| `fwod_representations.xlsx` | Raw FWOD dataset extracted from MIDI files |
| `fwod_representations_clean.xlsx` | **Primary dataset** - Cleaned version with 19,774 rows, 16 step features, 20 genre classes |
| `data_pattern_2.xlsx` | Aggregated dataset combining 2 consecutive patterns (32 features) |
| `data_pattern_2_inclusive.xlsx` | Same as above but with overlapping windows |
| `data_pattern_4_inclusive.xlsx` | 4 patterns combined with overlapping windows (64 features) |
| `data_pattern_5_inclusive.xlsx` | 5 patterns combined with overlapping windows (80 features) |
| `e1_all_hvs.pkl` | Pickled HV (Hit-Velocity) lists from experiment 1 |
| `Model_performance.xlsx` | Summary of model evaluation results |

### Configuration

| File | Description |
|------|-------------|
| `.gitignore` | Excludes `/dataset`, `/magenta midi`, `/tap to drum taps` directories |
| `CLAUDE.md` | AI assistant guidance for working with this codebase |

---

## Results Summary

The final classification results are produced by **`models_pipeline_test.ipynb`**:

| Model | Test Accuracy |
|-------|---------------|
| **KNN** | **80.0%** |
| XGBoost | 77.1% |
| RNN (Bidirectional) | 76.5% |
| SVM (RBF) | 69.6% |
| Random Forest | 65.7% |
| LightGBM | 65.4% |
| LSTM | 56.5% |

---

## Workflow to Reproduce Results

```
1. Obtain Magenta Groove MIDI Dataset → /magenta midi/

2. Run: classification_data_creation.ipynb
   → Generates: fwod_representations_clean.xlsx

3. (Optional) Run: multiple_pattern_db.ipynb
   → Generates: data_pattern_*.xlsx files

4. Run: models_pipeline_test.ipynb
   → Produces: Final classification results
```
