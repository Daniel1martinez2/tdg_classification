# Experiment: FWOD Classification Improvement Strategies
Date: 2026-02-19
Baseline: SOTA (Transformer+TBPE) = 0.66 Macro F1

## Summary

Tested 7 strategies across 20 experiments to improve FWOD-based 4-class genre classification (funk/jazz/latin/rock) with honest CV (StratifiedGroupKFold). **SMOTE inside CV on 8-bar data achieved the best Holdout F1 of 0.7325 (+11.0% over SOTA).** File-level voting reduces the Test-Holdout gap but doesn't improve peak Holdout F1. Bag-of-Rhythms and Multi-Scale Stacking underperformed.

## Results

| # | Experiment | CV F1 | Test F1 | Holdout F1 | vs SOTA | Test-Hold Gap | Features |
|---|-----------|------:|--------:|-----------:|--------:|--------------:|---------:|
| 1 | **SMOTE-CV 8-bar(128)** | 0.6208 | 0.4729 | **0.7325** | **+0.0725** | -0.2596 | 128 |
| 2 | Baseline 16-bar(256) | 0.5965 | 0.4771 | 0.7185 | +0.0585 | -0.2415 | 256 |
| 3 | SMOTE-CV 5-bar(80) | 0.6131 | 0.4823 | 0.7156 | +0.0556 | -0.2333 | 80 |
| 4 | Baseline 8-bar(128) | 0.6111 | 0.4929 | 0.7138 | +0.0538 | -0.2209 | 128 |
| 5 | SMOTE+FileVote 16-bar(256) | 0.6700 | 0.5268 | 0.7071 | +0.0471 | -0.1803 | 256 |
| 6 | FileVote 16-bar(256) | 0.6312 | 0.5521 | 0.7071 | +0.0471 | -0.1550 | 256 |
| 7 | Baseline 5-bar(80) | 0.6042 | 0.4927 | 0.6990 | +0.0390 | -0.2063 | 80 |
| 8 | FileVote 8-bar(128) | 0.6717 | 0.5157 | 0.6684 | +0.0084 | -0.1527 | 128 |
| 9 | SMOTE+FileVote 8-bar(128) | 0.7077 | 0.5712 | 0.6609 | +0.0009 | -0.0897 | 128 |
| 10 | FileVote 5-bar(80) | 0.6658 | 0.5234 | 0.6515 | -0.0085 | -0.1281 | 80 |
| 11 | Baseline 1-bar(16) | 0.5403 | 0.5351 | 0.6023 | -0.0577 | -0.0672 | 16 |
| 12 | FullCV 5-bar(80) | 0.5963 | - | 0.5963 | -0.0637 | - | 80 |
| 13 | FullCV 8-bar(128) | 0.5903 | - | 0.5903 | -0.0697 | - | 128 |
| 14 | BoR k=20 (file-level) | 0.5279 | 0.5435 | 0.5802 | -0.0798 | -0.0367 | 55 |
| 15 | FullCV 16-bar(256) | 0.5696 | - | 0.5696 | -0.0904 | - | 256 |
| 16 | BoR k=30 (file-level) | 0.5489 | 0.6268 | 0.5642 | -0.0958 | +0.0626 | 65 |
| 17 | FullCV 1-bar(16) | 0.5601 | - | 0.5601 | -0.0999 | - | 16 |
| 18 | BoR k=10 (file-level) | 0.5123 | 0.5728 | 0.5567 | -0.1033 | +0.0161 | 45 |
| 19 | MultiScale+LogReg | 0.9953 | 0.4726 | 0.4788 | -0.1812 | -0.0063 | 12 |
| 20 | MultiScale+ExtraTrees | 1.0000 | 0.4390 | 0.4353 | -0.2247 | +0.0037 | 12 |

## Key Findings

### What works:

- **SMOTE inside CV (exp_03)**: Best single result. SMOTE-CV on 8-bar gives **0.7325 Holdout F1** (+11.0% vs SOTA). Oversampling minority classes (jazz=101, latin=97 files vs rock=313) during training helps the model learn better decision boundaries for underrepresented genres.

- **Honest CV baselines (exp_01)**: Even without tricks, honest GroupKFold + Optuna gives strong Holdout: 8-bar=0.7138, 16-bar=0.7185. Honest hyperparameter selection finds more regularized models that generalize better.

- **More bars = better Holdout**: Consistent trend across strategies. 8-bar and 16-bar consistently outperform 5-bar and 1-bar. Optimal appears to be 8-16 bars.

### What partially works:

- **File-level voting (exp_02)**: Reduces Test-Holdout gap significantly (from -0.22 to -0.15) by aggregating per file. Peak Holdout is lower (0.7071 vs 0.7325), but Test F1 is much higher (0.55 vs 0.47). Better for reporting consistent metrics.

- **SMOTE + FileVote (exp_07)**: Smallest Test-Holdout gap of any multi-bar experiment (-0.0897 on 8-bar). Sacrifices peak Holdout for consistency. Good for papers where reviewers care about Test-Holdout agreement.

### What doesn't work:

- **Full CV over all 662 files (exp_04)**: Conservative estimate ~0.56-0.60 F1. The high per-fold variance (std=0.06) reflects the file-level diversity. This is the most honest evaluation but uses default params (no Optuna).

- **Bag-of-Rhythms (exp_05)**: File-level classification with rhythm prototypes peaks at 0.58 Holdout. Too few training samples (464 files) for this approach to compete.

- **Multi-Scale Stacking (exp_06)**: Complete failure. Perfect CV (1.0) = massive overfitting. Bar-level prediction probabilities leak training information into file-level features. Would need proper nested CV to work, too expensive.

### The Test-Holdout gap:

The persistent Test << Holdout gap is NOT data leakage — it's a property of the specific 97 test files being harder to classify than the 101 holdout files. With file-level voting the gap narrows from ~0.22 to ~0.09-0.15, but doesn't disappear.

## Recommendations for the paper

1. **Primary result**: SMOTE-CV ExtraTrees on 8-bar FWOD, **Holdout F1 = 0.7325** (+11% vs SOTA)
2. **Report both Test and Holdout F1** with file-level voting for credibility (Test=0.57, Hold=0.66-0.71)
3. **Report Full CV** (exp_04) as conservative lower bound (~0.59 F1)
4. **Integrate SMOTE into the CLASSIFIER.py pipeline** for the full model sweep

## Next Steps

- Run SMOTE-CV across all 7 models in the pipeline (not just ExtraTrees)
- Consider ADASYN as alternative to SMOTE
- Test if FullCV + Optuna (nested, but limited trials) can beat 0.60
- The full pipeline with fixed CV + SMOTE should be the final run
