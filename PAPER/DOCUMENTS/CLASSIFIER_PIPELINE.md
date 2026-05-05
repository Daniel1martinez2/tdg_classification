# Classification Pipeline

## Overview

This document describes the machine learning pipeline for drum pattern genre classification using the FWOD (Flattened Weighted Onset Distribution) representation.

---

## 1. Pipeline Flow

```mermaid
flowchart TD
    subgraph INPUT["Input Layer"]
        D1["1-pattern<br/>16 features"]
        D2["2-pattern<br/>32 features"]
        D3["4-pattern<br/>64 features"]
        D4["5-pattern<br/>80 features"]
    end

    subgraph FILTER["Class Filtering"]
        F{{"State of Art<br/>Comparison Mode?"}}
        F -->|Yes| F4["Filter to 4 classes:<br/>funk, jazz, latin, rock"]
        F -->|No| F18["Keep all 18 classes"]
    end

    subgraph SPLIT["Pre-defined Splits"]
        TRAIN["Train Set<br/>70%"]
        TEST["Test Set<br/>15%"]
        HOLDOUT["Holdout Set<br/>15%"]
    end

    subgraph MODELS["Model Pool"]
        M1["KNN"]
        M2["RandomForest"]
        M3["XGBoost"]
        M4["LightGBM"]
    end

    subgraph OPTUNA["Hyperparameter Optimization"]
        OPT["Optuna Bayesian Search<br/>30 trials"]
        CV["5-Fold Cross-Validation<br/>Metric: Macro F1"]
        OPT --> CV
        CV --> BEST["Best Hyperparameters"]
    end

    subgraph EVAL["Evaluation"]
        TRAIN_FINAL["Train Final Model"]
        EVAL_TEST["Evaluate on Test"]
        EVAL_HOLD["Evaluate on Holdout"]
    end

    subgraph OUTPUT["Results"]
        METRICS["Metrics per Model/Dataset"]
        EXCEL["Excel Export"]
        COMPARE["State of Art Comparison"]
    end

    D1 & D2 & D3 & D4 --> F
    F4 & F18 --> SPLIT
    TRAIN --> MODELS
    MODELS --> OPTUNA
    BEST --> TRAIN_FINAL
    TRAIN_FINAL --> EVAL_TEST
    EVAL_TEST --> EVAL_HOLD
    TEST --> EVAL_TEST
    HOLDOUT --> EVAL_HOLD
    EVAL_HOLD --> METRICS
    METRICS --> EXCEL
    METRICS --> COMPARE
```

---

## 2. Datasets

The pipeline evaluates four dataset configurations, each representing different temporal aggregations:

| Dataset | Features | Description | Temporal Context |
|---------|----------|-------------|------------------|
| 1-pattern | 16 | Single bar | 1 measure |
| 2-pattern | 32 | Two consecutive bars | 2 measures |
| 4-pattern | 64 | Four consecutive bars | 4 measures |
| 5-pattern | 80 | Five consecutive bars | 5 measures |

**Rationale**: Longer aggregations capture cyclic patterns common in genres (e.g., 2-bar or 4-bar phrases).

---

## 3. Class Filtering

### State of Art Mode (4 classes)
For comparison with the reference paper, the pipeline filters to:
- **funk**
- **jazz**
- **latin**
- **rock**

This matches the experimental setup of "Improved Symbolic Drum Style Classification with Grammar-Based Hierarchical Representations".

### Full Mode (18 classes)
All available genres in the dataset:
`afrobeat, afrocuban, blues, country, dance, funk, gospel, highlife, hiphop, jazz, latin, middleeastern, neworleans, pop, punk, reggae, rock, soul`

---

## 4. Data Splitting Strategy

The split is performed at the **file level** to prevent data leakage:

```mermaid
flowchart LR
    subgraph FILE["MIDI File"]
        P1["Pattern 1"]
        P2["Pattern 2"]
        P3["Pattern 3"]
        P4["..."]
    end

    FILE -->|"All patterns together"| SPLIT{"Stratified<br/>File Split"}
    SPLIT -->|70%| TRAIN["Train"]
    SPLIT -->|15%| TEST["Test"]
    SPLIT -->|15%| HOLDOUT["Holdout"]
```

**Key Guarantee**: All patterns from the same source file belong to the same split, eliminating correlation leakage.

---

## 5. Models

| Model | Type | Key Characteristics |
|-------|------|---------------------|
| **KNN** | Instance-based | Distance metric learning, no assumptions about data distribution |
| **RandomForest** | Ensemble (Bagging) | Robust to overfitting, handles feature interactions |
| **XGBoost** | Ensemble (Boosting) | Gradient boosting, regularization, handles imbalanced data |
| **LightGBM** | Ensemble (Boosting) | Faster training, leaf-wise growth, memory efficient |

---

## 6. Hyperparameter Optimization

### Optuna Configuration
- **Trials**: 30 per model
- **Sampler**: TPESampler (Tree-structured Parzen Estimator)
- **Optimization**: Bayesian (sequential model-based)
- **Objective**: Maximize Macro F1 Score
- **Validation**: 5-fold cross-validation on training set

### Why Bayesian Optimization (TPE)?

```mermaid
flowchart LR
    subgraph RANDOM["Random Search"]
        R1["Trial 1"] --> R2["Trial 2"] --> R3["Trial 3"]
        R1 -.->|"No learning"| R3
    end

    subgraph BAYESIAN["Bayesian (TPE)"]
        B1["Trial 1"] --> B2["Trial 2"] --> B3["Trial 3"]
        B1 -->|"Updates model"| B2
        B2 -->|"Updates model"| B3
    end
```

| Aspect | Random Search | Bayesian (TPE) |
|--------|--------------|----------------|
| Strategy | Independent samples | Informed by history |
| Efficiency | Low | High |
| Convergence | Slow | Fast |
| Exploration/Exploitation | Only exploration | Balanced |

**TPE** construye dos distribuciones:
- $l(x)$: Hiperparámetros que dieron buenos resultados
- $g(x)$: Hiperparámetros que dieron malos resultados

Maximiza: $\frac{l(x)}{g(x)}$ para elegir el siguiente trial

### Search Spaces

```mermaid
mindmap
  root((Hyperparameters))
    KNN
      n_neighbors: 1-30
      weights: uniform/distance
      metric: euclidean/manhattan/minkowski
    RandomForest
      n_estimators: 50-300
      max_depth: 5-50
      min_samples_split: 2-10
      min_samples_leaf: 1-5
    XGBoost
      n_estimators: 50-300
      max_depth: 3-10
      learning_rate: 0.01-0.3
      subsample: 0.5-1.0
      colsample_bytree: 0.5-1.0
    LightGBM
      n_estimators: 50-300
      max_depth: 3-15
      learning_rate: 0.01-0.3
      num_leaves: 20-100
      subsample: 0.5-1.0
```

---

## 7. Metrics

### Primary Metric: Macro F1 Score

$$\text{Macro F1} = \frac{1}{N} \sum_{i=1}^{N} F1_i$$

Where $F1_i$ is the F1 score for class $i$:

$$F1_i = 2 \cdot \frac{\text{Precision}_i \cdot \text{Recall}_i}{\text{Precision}_i + \text{Recall}_i}$$

**Why Macro F1?**
- Treats all classes equally regardless of sample size
- Penalizes poor performance on minority classes
- Same metric used in state of art paper (enables direct comparison)

### Secondary Metric: Accuracy

$$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}$$

Reported for reference but not used for model selection.

---

## 8. Evaluation Protocol

```mermaid
sequenceDiagram
    participant Train as Training Set
    participant Optuna as Optuna
    participant Model as Model
    participant Test as Test Set
    participant Hold as Holdout Set

    Train->>Optuna: 5-fold CV search
    Optuna->>Optuna: 30 trials (Macro F1)
    Optuna->>Model: Best hyperparameters
    Train->>Model: Fit final model
    Model->>Test: Predict
    Test->>Test: Calculate Test Macro F1
    Model->>Hold: Predict
    Hold->>Hold: Calculate Holdout Macro F1
```

### Three-Level Evaluation

| Level | Data | Purpose |
|-------|------|---------|
| **CV Score** | Training (5-fold) | Hyperparameter selection |
| **Test Score** | Test set | Model comparison |
| **Holdout Score** | Holdout set | Final unbiased estimate |

---

## 9. State of Art Comparison

### Reference Paper
- **Title**: "Improved Symbolic Drum Style Classification with Grammar-Based Hierarchical Representations"
- **Best Result**: Macro F1 = 0.66
- **Method**: Transformer + TBPE (Tree-Based Positional Encoding)
- **Classes**: funk, jazz, latin, rock

### Comparison Criteria
| Aspect | Our Pipeline | State of Art |
|--------|-------------|--------------|
| Classes | 4 (filtered) | 4 |
| Metric | Macro F1 | Macro F1 |
| Split | 70/15/15 (file-level) | 80/10/10 |
| Features | FWOD (16-80) | Grammar tokens |

---

## 10. Output

### Console Output
- Rich formatted tables per model
- Best configuration per model
- Overall best configuration
- State of art comparison panel

### Excel Export
Three sheets:
1. **All Results**: Complete results matrix (Dataset × Model)
2. **Best Per Model**: Best dataset configuration for each model
3. **Overall Best**: Single best configuration with state of art comparison

---

## 11. Execution Flow Summary

```mermaid
flowchart TB
    START([Start]) --> LOAD["Load 4 Datasets"]
    LOAD --> FILTER["Apply Class Filter<br/>(if State of Art mode)"]

    FILTER --> LOOP1{{"For each Dataset"}}
    LOOP1 --> LOOP2{{"For each Model"}}

    LOOP2 --> OPT["Optuna Optimization<br/>(30 trials, 5-fold CV)"]
    OPT --> TRAIN["Train with Best Params"]
    TRAIN --> EVAL["Evaluate Test + Holdout"]
    EVAL --> STORE["Store Results"]

    STORE --> LOOP2
    LOOP2 -->|Done| LOOP1
    LOOP1 -->|Done| SUMMARY["Generate Summary Tables"]

    SUMMARY --> EXPORT["Export to Excel"]
    EXPORT --> COMPARE["Show State of Art<br/>Comparison"]
    COMPARE --> END([End])
```

---

## 12. Reproducibility

| Parameter | Value |
|-----------|-------|
| Random State | 42 |
| CV Folds | 5 |
| Optuna Trials | 30 |
| Split Ratios | 70/15/15 |

All experiments use the same random seed for reproducible results.
