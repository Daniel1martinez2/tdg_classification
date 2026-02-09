# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **drum pattern genre classification** project using FWOD (Flattened Weighted Onset Distribution) representation to classify musical drum patterns from MIDI files. The project analyzes the Magenta Groove MIDI Dataset and classifies patterns into 20 music genres.

**Research Goal**: Improve upon existing approaches (reference: https://arxiv.org/pdf/2407.17536) which use only 4 classes, by expanding to 20 genres using the FWOD representation.

## Architecture

```
MIDI Files (Magenta Dataset, 20 genres)
    ↓
midifile2hv_list() - MIDI parsing with GM drum mapping
    ↓
HV-Lists (hit-velocity pairs, 16-step quantized bars)
    ↓
flatten_hv_list() - Weighted instrument flattening
    ↓
FWOD Representation (16 normalized values per bar)
    ↓
Pattern Aggregation (optional: combine N patterns)
    ↓
ML Pipeline (KNN, SVM, XGBoost, LightGBM, LSTM, RNN)
```

**Instrument Weighting**: Low drums (kick) × 3, mid drums (snare) × 2, high drums (hi-hat) × 1

## Key Files

| File | Purpose |
|------|---------|
| `classification_data_creation.ipynb` | MIDI parsing and FWOD dataset generation |
| `models_pipeline_test.ipynb` | Comprehensive ML pipeline with hyperparameter tuning |
| `classification_task.ipynb` | Basic classification experiments |
| `multiple_pattern_db.ipynb` | Pattern aggregation with overlapping windows |
| `fwod_representations_clean.xlsx` | Primary dataset (19,774 rows, 16 step features, 20 classes) |

## Dependencies

Python libraries used (no requirements.txt - install as needed):
- **MIDI**: mido
- **Data**: pandas, numpy
- **ML**: scikit-learn, xgboost, lightgbm, scikit-optimize (BayesSearchCV)
- **Deep Learning**: tensorflow/keras
- **Stats**: scipy, statsmodels
- **Visualization**: matplotlib

## Data Not in Repository

These directories are gitignored and must be obtained separately:
- `/dataset` - Raw drum pattern data
- `/magenta midi` - Magenta Groove MIDI Dataset (20 genres)
- `/tap to drum taps` - Tap-to-drum experiment data

## 20 Genre Classes

pop, neworleans, boska, blues, afrocuban, latin, sano, reggae, afrobeat, rock, dance, hiphop, punk, gospel, highlife, soul, middleeastern, country, funk, jazz

## Model Performance (Best Results)

| Model | Test Accuracy |
|-------|---------------|
| KNN | 80.0% |
| RNN (Bidirectional) | 76.5% |
| XGBoost | 77.1% |
| SVM (RBF) | 69.6% |

## Development Notes

- All code is in Jupyter notebooks (no modular Python packages)
- Run notebooks in order: `classification_data_creation.ipynb` → `models_pipeline_test.ipynb`
- Pattern aggregation variants: non-overlapping vs inclusive (overlapping windows)
- Dataset has class imbalance: Rock (1205 samples) vs smaller classes (~50-100 samples)
