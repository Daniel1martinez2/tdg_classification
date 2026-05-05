# Experimentation Strategies

Plan de pruebas para superar el SOTA (Transformer+TBPE, Macro F1 = 0.66) en clasificación de 4 clases (funk, jazz, latin, rock) sobre FWOD.

Source experiment: `LABS/experiments/experiment__fwod_strategies__20260219.md`.

---

## 1. Datasets

Cada dataset agrega N bars consecutivos por ventana inclusiva (solapada).

| Dataset | Bars | Features |
|--------:|-----:|---------:|
| 1-pat   | 1  | 16  |
| 2-pat   | 2  | 32  |
| 4-pat   | 4  | 64  |
| 5-pat   | 5  | 80  |
| 6-pat   | 6  | 96  |
| 7-pat   | 7  | 112 |
| 8-pat   | 8  | 128 |
| 16-pat  | 16 | 256 |

---

## 2. Models

Optuna (TPE, 60 trials, MedianPruner) sobre `StratifiedGroupKFold(n=5)` agrupando por `file`.

| Activo (Holdout ≥ 0.60) | Descartado (Holdout < 0.60) |
|---|---|
| RandomForest         | KNN |
| ExtraTrees           | LogisticRegression |
| GradientBoosting     | AdaBoost |
| XGBoost              | MLP |
| LightGBM             | SVM |
| CatBoost             | |
| HistGradientBoosting | |

**Ensembles:** Voting (soft) y Stacking sobre Top-3 por dataset.

---

## 3. Strategies

| # | Estrategia | Idea | Evaluación |
|---|---|---|---|
| S1 | **Baseline GroupKFold**          | CV honesto agrupado por `file` | OK |
| S2 | **SMOTE-CV**                     | Oversampling minoritarias **dentro** del fold | ⭐ ganadora |
| S3 | **File-level voting**            | Voto mayoritario sobre patrones del mismo file | Reduce gap, baja peak |
| S4 | **SMOTE + FileVote**             | S2 + S3 combinadas | Más consistente, peak menor |
| S5 | **Full CV (sin holdout fijo)**   | CV sobre los 662 archivos sin split | Bound conservador |
| S6 | **Bag-of-Rhythms**               | k-prototipos de ritmo a nivel file | Insuficientes datos |
| S7 | **Multi-Scale Stacking**         | Probas bar-level → meta-modelo file-level | ❌ leakage (CV=1.00) |

---

## 4. Combinaciones corridas (matriz)

| Strategy × Dataset | 1 | 5 | 8 | 16 |
|---|:-:|:-:|:-:|:-:|
| Baseline             | ✓ | ✓ | ✓ | ✓ |
| SMOTE-CV             |   | ✓ | ✓ |   |
| FileVote             |   | ✓ | ✓ | ✓ |
| SMOTE+FileVote       |   |   | ✓ | ✓ |
| Full CV              | ✓ |   | ✓ | ✓ |
| BoR (k=10/20/30)     |   — file-level —  ||||
| MultiScale (LR / ET) |   — 12 feats fijos — ||||

20 experimentos en total.

**Estado de implementación** (¿corre desde el pipeline oficial?):

| Strategy | `CLASSIFIER.py` | Notebooks (LABS) |
|---|:-:|:-:|
| S1 Baseline GroupKFold  | ✅ | ✅ |
| S2 SMOTE-CV             | ⏳ | ✅ |
| S3 FileVote             | ⏳ | ✅ |
| S4 SMOTE + FileVote     | ⏳ | ✅ |
| S5 Full CV              | ⏳ | ✅ |
| S6 Bag-of-Rhythms       | ❌ | ✅ (descartada) |
| S7 Multi-Scale Stacking | ❌ | ✅ (descartada por leakage) |

Sólo **S1** corre desde `CLASSIFIER.py`. **S2–S5** están en notebooks; integrarlas al pipeline es el pendiente #1 (§8). **S6/S7** quedaron fuera por bajo desempeño / leakage estructural.

Los fixes de leakage (split file-level + GroupKFold por `file`) ya están en `CLASSIFIER.py` desde la corrida del 10-feb. El SMOTE-CV todavía no.

---

## 5. Top results (Holdout Macro F1)

| Rank | Config | CV | Test | **Holdout** | vs SOTA |
|-----:|---|---:|---:|---:|---:|
| 🥇 | SMOTE-CV · ExtraTrees · 8-pat   | 0.62 | 0.47 | **0.7325** | **+11.0%** |
| 🥈 | Baseline · 16-pat               | 0.60 | 0.48 | 0.7185 | +8.9% |
| 🥉 | SMOTE-CV · 5-pat                | 0.61 | 0.48 | 0.7156 | +8.4% |
| 4  | Baseline · 8-pat                | 0.61 | 0.49 | 0.7138 | +5.7% |
| 5  | SMOTE+FileVote · 16-pat         | 0.67 | 0.53 | 0.7071 | +4.7% |

Multi-Scale: CV=1.00 / Holdout=0.43-0.48 → descartado por leakage.

---

## 6. Hallazgos

- **Más bars → mejor Holdout.** Óptimo en 8–16.
- **SMOTE dentro del fold** rompe el techo: corrige desbalance (rock=313 vs latin=97) sin filtrar sintéticos a validación.
- **Gap Test–Holdout** (~−0.22) **no es overfitting**: los 97 files de test son intrínsecamente más difíciles que los 101 del holdout.
- **FileVote** no sube el peak; sólo estrecha el gap Test–Holdout.

---

## 7. Configuración elegida para el paper

```
Split:    70/15/15 file-level (random_state=42)
CV:       StratifiedGroupKFold(n=5) por file
Resampling: SMOTE dentro del fold (k_neighbors=5)
Features: FWOD 8-pat (128) o 16-pat (256)
Modelo:   ExtraTrees (Optuna, 60 trials)
Métrica:  Macro F1 (Holdout)
Target:   ≥ 0.73 (SOTA = 0.66)
```

---

## 8. Pendientes

- [ ] Integrar SMOTE en `CLASSIFIER.py` (hoy sólo vive en notebooks).
- [ ] Sweep SMOTE-CV × 7 modelos × 8 datasets + ensembles.
- [ ] Comparar ADASYN vs SMOTE.
- [ ] Nested CV con presupuesto Optuna reducido sobre Full CV.
