"""
Genre Classifier Pipeline
==========================
Machine learning pipeline for drum pattern genre classification using FWOD representation.
Uses Optuna for hyperparameter optimization with file-level stratified splits.

Comparable to state of art paper:
- Uses Macro F1 Score as primary metric
- Can filter to 4 classes: funk, jazz, latin, rock
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Type

import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, AdaBoostClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

optuna.logging.set_verbosity(optuna.logging.WARNING)
console = Console()

# State of art paper uses these 4 classes
STATE_OF_ART_CLASSES = ['funk', 'jazz', 'latin', 'rock']


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Pipeline configuration."""
    random_state: int = 42
    n_trials: int = 60
    cv_folds: int = 8
    n_jobs: int = -1


# =============================================================================
# MODEL DEFINITIONS (Strategy Pattern)
# =============================================================================

class BaseModel(ABC):
    """Abstract base for all models."""

    name: str = "base"

    @abstractmethod
    def get_optuna_params(self, trial: optuna.Trial) -> Dict:
        """Define hyperparameter search space."""
        pass

    @abstractmethod
    def create_model(self, params: Dict):
        """Create model instance with given params."""
        pass


class RandomForestModel(BaseModel):
    name = "RandomForest"

    def get_optuna_params(self, trial: optuna.Trial) -> Dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", "None"]),
        }

    def create_model(self, params: Dict, random_state: int = 42):
        params = {**params}
        if params["class_weight"] == "None":
            params["class_weight"] = None
        return RandomForestClassifier(**params, random_state=random_state, n_jobs=-1)


class SVMModel(BaseModel):
    name = "SVM"

    def get_optuna_params(self, trial: optuna.Trial) -> Dict:
        return {
            "C": trial.suggest_float("C", 0.1, 100, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "poly"]),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", "None"]),
        }

    def create_model(self, params: Dict, random_state: int = 42):
        params = {**params}
        if params["class_weight"] == "None":
            params["class_weight"] = None
        return SVC(**params, random_state=random_state, probability=True)


class KNNModel(BaseModel):
    name = "KNN"

    def get_optuna_params(self, trial: optuna.Trial) -> Dict:
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),
        }

    def create_model(self, params: Dict, random_state: int = 42):
        return KNeighborsClassifier(**params, n_jobs=-1)


class XGBoostModel(BaseModel):
    name = "XGBoost"

    def get_optuna_params(self, trial: optuna.Trial) -> Dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    def create_model(self, params: Dict, random_state: int = 42):
        return XGBClassifier(**params, random_state=random_state, n_jobs=-1, verbosity=0)


class LightGBMModel(BaseModel):
    name = "LightGBM"

    def get_optuna_params(self, trial: optuna.Trial) -> Dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    def create_model(self, params: Dict, random_state: int = 42):
        return LGBMClassifier(**params, random_state=random_state, n_jobs=-1, verbose=-1)


class CatBoostModel(BaseModel):
    name = "CatBoost"

    def get_optuna_params(self, trial: optuna.Trial) -> Dict:
        return {
            "iterations": trial.suggest_int("iterations", 100, 500),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "auto_class_weights": trial.suggest_categorical("auto_class_weights", ["Balanced", "SqrtBalanced", "None"]),
        }

    def create_model(self, params: Dict, random_state: int = 42):
        params = {**params}
        if params["auto_class_weights"] == "None":
            params.pop("auto_class_weights")
        return CatBoostClassifier(**params, random_state=random_state, verbose=0)


class ExtraTreesModel(BaseModel):
    name = "ExtraTrees"

    def get_optuna_params(self, trial: optuna.Trial) -> Dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", "None"]),
        }

    def create_model(self, params: Dict, random_state: int = 42):
        params = {**params}
        if params["class_weight"] == "None":
            params["class_weight"] = None
        return ExtraTreesClassifier(**params, random_state=random_state, n_jobs=-1)


class GradientBoostingModel(BaseModel):
    name = "GradientBoosting"

    def get_optuna_params(self, trial: optuna.Trial) -> Dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }

    def create_model(self, params: Dict, random_state: int = 42):
        return GradientBoostingClassifier(
            **params, random_state=random_state,
            n_iter_no_change=20, validation_fraction=0.1, tol=1e-4
        )


class HistGradientBoostingModel(BaseModel):
    name = "HistGradientBoosting"

    def get_optuna_params(self, trial: optuna.Trial) -> Dict:
        return {
            "max_iter": trial.suggest_int("max_iter", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", "None"]),
        }

    def create_model(self, params: Dict, random_state: int = 42):
        params = {**params}
        if params["class_weight"] == "None":
            params["class_weight"] = None
        return HistGradientBoostingClassifier(
            **params, random_state=random_state,
            early_stopping=True, n_iter_no_change=20, validation_fraction=0.1
        )


class MLPModel(BaseModel):
    name = "MLP"

    def get_optuna_params(self, trial: optuna.Trial) -> Dict:
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f"n_units_l{i}", 32, 256))
        return {
            "hidden_layer_sizes": tuple(layers),
            "n_layers": n_layers,
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        }

    def create_model(self, params: Dict, random_state: int = 42):
        params = {**params}
        # Remove Optuna-specific keys that aren't MLPClassifier params
        params.pop("n_layers", None)
        for key in list(params.keys()):
            if key.startswith("n_units_l"):
                params.pop(key)
        return MLPClassifier(**params, random_state=random_state, max_iter=500, early_stopping=True)


class AdaBoostModel(BaseModel):
    name = "AdaBoost"

    def get_optuna_params(self, trial: optuna.Trial) -> Dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 2.0, log=True),
        }

    def create_model(self, params: Dict, random_state: int = 42):
        return AdaBoostClassifier(**params, random_state=random_state, algorithm="SAMME")


class LogisticRegressionModel(BaseModel):
    name = "LogisticRegression"

    def get_optuna_params(self, trial: optuna.Trial) -> Dict:
        return {
            "C": trial.suggest_float("C", 1e-3, 100, log=True),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", "None"]),
        }

    def create_model(self, params: Dict, random_state: int = 42):
        params = {**params}
        if params["class_weight"] == "None":
            params["class_weight"] = None
        return LogisticRegression(**params, random_state=random_state, max_iter=1000, n_jobs=-1)


# =============================================================================
# DATA LOADER
# =============================================================================

class DataLoader:
    """Loads and prepares data using pre-defined splits."""

    def __init__(self, filepath: str, filter_classes: Optional[List[str]] = None):
        self.filepath = filepath
        self.filter_classes = filter_classes
        self.df = None
        self.feature_cols = None
        self.classes = None
        self.label_encoder = LabelEncoder()

    def load(self) -> "DataLoader":
        # Prefer parquet (much faster) if available, fallback to xlsx
        parquet_path = self.filepath.replace('.xlsx', '.parquet')
        if os.path.exists(parquet_path):
            self.df = pd.read_parquet(parquet_path)
        else:
            self.df = pd.read_excel(self.filepath)

        # Filter to specific classes if requested (for state of art comparison)
        if self.filter_classes:
            original_count = len(self.df)
            self.df = self.df[self.df['class'].isin(self.filter_classes)].copy()
            print(f"Filtered to {self.filter_classes}: {original_count} -> {len(self.df)} samples")

        self.feature_cols = [c for c in self.df.columns if c.startswith(('step_', 'feature_'))]
        self.classes = sorted(self.df['class'].unique())

        # Encode labels to numeric
        self.df['class_encoded'] = self.label_encoder.fit_transform(self.df['class'])

        print(f"Loaded: {self.filepath}")
        print(f"  Samples: {len(self.df)} | Features: {len(self.feature_cols)} | Classes: {len(self.classes)}")
        return self

    def get_split(self, split: str, encoded: bool = True) -> tuple:
        """Get X, y for a specific split (train/test/holdout)."""
        mask = self.df['split'] == split
        X = self.df.loc[mask, self.feature_cols].values
        y = self.df.loc[mask, 'class_encoded' if encoded else 'class'].values
        return X, y

    def get_groups(self, split: str) -> np.ndarray:
        """Get file-level group labels for a split (for GroupKFold CV)."""
        mask = self.df['split'] == split
        return self.df.loc[mask, 'file'].values

    def decode_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """Convert encoded labels back to string labels."""
        return self.label_encoder.inverse_transform(y_encoded)

    @property
    def train(self) -> tuple:
        return self.get_split('train')

    @property
    def train_groups(self) -> np.ndarray:
        return self.get_groups('train')

    @property
    def test(self) -> tuple:
        return self.get_split('test')

    @property
    def holdout(self) -> tuple:
        return self.get_split('holdout')


# =============================================================================
# OPTIMIZER
# =============================================================================

class StudyEarlyStopping:
    """Callback to stop Optuna study if no improvement after `patience` trials."""

    def __init__(self, patience: int = 20):
        self.patience = patience

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if study.best_trial.number + self.patience <= trial.number:
            study.stop()


class HyperparameterOptimizer:
    """Optuna-based hyperparameter optimization with early stopping at 3 levels:
    - Study level: stop entire search if no improvement for N trials
    - Trial level: prune bad trials after a few CV folds (MedianPruner)
    - Model level: early stopping for boosting models (handled in ModelTrainer)
    """

    STUDY_PATIENCE = 10  # Stop study if no improvement in 25 consecutive trials

    def __init__(self, model_def: BaseModel, config: Config):
        self.model_def = model_def
        self.config = config
        self.best_params = None
        self.best_score = None

    def optimize(self, X_train: np.ndarray, y_train: np.ndarray,
                 groups: Optional[np.ndarray] = None) -> Dict:
        from sklearn.base import clone
        from sklearn.model_selection import StratifiedGroupKFold

        def objective(trial: optuna.Trial) -> float:
            params = self.model_def.get_optuna_params(trial)
            model = self.model_def.create_model(params, self.config.random_state)

            # StratifiedGroupKFold: ensures patterns from same file stay
            # together in the same fold, preventing overlapping-window leakage
            sgkf = StratifiedGroupKFold(n_splits=self.config.cv_folds)

            scores = []
            for step, (train_idx, val_idx) in enumerate(sgkf.split(X_train, y_train, groups)):
                X_t, X_v = X_train[train_idx], X_train[val_idx]
                y_t, y_v = y_train[train_idx], y_train[val_idx]

                model_clone = clone(model)
                model_clone.fit(X_t, y_t)
                score = f1_score(y_v, model_clone.predict(X_v), average='macro')
                scores.append(score)

                # Report intermediate score and check if trial should be pruned
                trial.report(np.mean(scores), step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(scores)

        sampler = optuna.samplers.TPESampler(seed=self.config.random_state)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        # Early stop the study if no improvement for STUDY_PATIENCE trials
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            show_progress_bar=True,
            callbacks=[StudyEarlyStopping(patience=self.STUDY_PATIENCE)]
        )

        self.best_params = study.best_params
        self.best_score = study.best_value

        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_total = len(study.trials)
        stopped_early = n_total < self.config.n_trials
        status = f"  Trials: {n_complete} complete, {n_pruned} pruned"
        if stopped_early:
            status += f" | Study stopped early at trial {n_total}/{self.config.n_trials} (no improvement for {self.STUDY_PATIENCE} trials)"
        print(status)

        return self.best_params


# =============================================================================
# TRAINER
# =============================================================================

@dataclass
class TrainingResult:
    """Container for training results."""
    model_name: str
    best_params: Dict
    cv_f1: float
    test_f1: float
    holdout_f1: Optional[float]
    test_accuracy: float
    holdout_accuracy: Optional[float]
    model: object


class ModelTrainer:
    """Trains a single model with optimization."""

    def __init__(self, model_def: BaseModel, config: Config):
        self.model_def = model_def
        self.config = config

    # Models that support early stopping with eval_set
    EARLY_STOP_MODELS = {"XGBoost", "LightGBM", "CatBoost"}

    def train(self, data: DataLoader, evaluate_holdout: bool = False) -> TrainingResult:
        X_train, y_train = data.train
        X_test, y_test = data.test
        train_groups = data.train_groups

        print(f"\n{'='*60}")
        print(f"Training: {self.model_def.name}")
        print(f"{'='*60}")

        # Optimize hyperparameters (file-aware CV to prevent overlapping-window leakage)
        optimizer = HyperparameterOptimizer(self.model_def, self.config)
        best_params = optimizer.optimize(X_train, y_train, groups=train_groups)

        print(f"Best params: {best_params}")
        print(f"CV Macro F1: {optimizer.best_score:.4f}")

        # Train final model with early stopping for boosting models
        model = self.model_def.create_model(best_params, self.config.random_state)

        if self.model_def.name in self.EARLY_STOP_MODELS:
            # Group-aware split for early stopping validation (no file leakage)
            from sklearn.model_selection import GroupShuffleSplit
            gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=self.config.random_state)
            tr_idx, val_idx = next(gss.split(X_train, y_train, train_groups))
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            fit_params = {}
            if self.model_def.name == "XGBoost":
                fit_params = {"eval_set": [(X_val, y_val)], "verbose": False}
                model.set_params(early_stopping_rounds=50)
            elif self.model_def.name == "LightGBM":
                import lightgbm as lgb
                fit_params = {"eval_set": [(X_val, y_val)],
                              "callbacks": [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]}
            elif self.model_def.name == "CatBoost":
                fit_params = {"eval_set": (X_val, y_val), "early_stopping_rounds": 50}
            model.fit(X_tr, y_tr, **fit_params)
            print(f"  (Early stopping with 10% validation split)")
        else:
            model.fit(X_train, y_train)

        # Evaluate on test
        y_pred_test = model.predict(X_test)
        test_f1 = f1_score(y_test, y_pred_test, average='macro')
        test_acc = accuracy_score(y_test, y_pred_test)
        print(f"Test Macro F1: {test_f1:.4f} | Test Accuracy: {test_acc:.4f}")

        # Evaluate on holdout (optional)
        holdout_f1 = None
        holdout_acc = None
        if evaluate_holdout:
            X_holdout, y_holdout = data.holdout
            y_pred_holdout = model.predict(X_holdout)
            holdout_f1 = f1_score(y_holdout, y_pred_holdout, average='macro')
            holdout_acc = accuracy_score(y_holdout, y_pred_holdout)
            print(f"Holdout Macro F1: {holdout_f1:.4f} | Holdout Accuracy: {holdout_acc:.4f}")

        return TrainingResult(
            model_name=self.model_def.name,
            best_params=best_params,
            cv_f1=optimizer.best_score,
            test_f1=test_f1,
            holdout_f1=holdout_f1,
            test_accuracy=test_acc,
            holdout_accuracy=holdout_acc,
            model=model
        )


# =============================================================================
# PIPELINE
# =============================================================================

class ClassificationPipeline:
    """Main pipeline orchestrator."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.results: List[TrainingResult] = []
        self.data: Optional[DataLoader] = None

    def load_data(self, filepath: str) -> "ClassificationPipeline":
        self.data = DataLoader(filepath).load()
        return self

    def run(self, models: List[BaseModel], evaluate_holdout: bool = False) -> "ClassificationPipeline":
        if not self.data:
            raise ValueError("Data not loaded. Call load_data() first.")

        for model_def in models:
            trainer = ModelTrainer(model_def, self.config)
            result = trainer.train(self.data, evaluate_holdout)
            self.results.append(result)

        return self

    def summary(self) -> pd.DataFrame:
        """Print and return results summary."""
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY (Macro F1 Score)")
        print(f"{'='*60}")

        rows = [{
            "Model": r.model_name,
            "CV F1": f"{r.cv_f1:.4f}",
            "Test F1": f"{r.test_f1:.4f}",
            "Holdout F1": f"{r.holdout_f1:.4f}" if r.holdout_f1 else "N/A",
            "Test Acc": f"{r.test_accuracy:.4f}",
            "Holdout Acc": f"{r.holdout_accuracy:.4f}" if r.holdout_accuracy else "N/A"
        } for r in self.results]

        df = pd.DataFrame(rows).sort_values("Test F1", ascending=False)
        print(df.to_string(index=False))
        return df

    def get_best_model(self) -> TrainingResult:
        return max(self.results, key=lambda r: r.test_f1)

    def classification_report(self, model_name: str, split: str = "test") -> str:
        result = next((r for r in self.results if r.model_name == model_name), None)
        if not result:
            raise ValueError(f"Model '{model_name}' not found.")

        X, y_true_encoded = self.data.get_split(split, encoded=True)
        y_pred_encoded = result.model.predict(X)

        # Decode to original labels for readable report
        y_true = self.data.decode_labels(y_true_encoded)
        y_pred = self.data.decode_labels(y_pred_encoded)
        return classification_report(y_true, y_pred)


# =============================================================================
# MULTI-DATASET RUNNER
# =============================================================================

@dataclass
class DatasetResult:
    """Results for a single dataset."""
    dataset_name: str
    n_samples: int
    n_features: int
    n_classes: int
    model_name: str
    best_params: Dict
    cv_f1: float
    test_f1: float
    holdout_f1: float
    test_accuracy: float
    holdout_accuracy: float


class MultiDatasetRunner:
    """Runs models across multiple datasets and compares results.

    Supports incremental Excel saving and resuming from a previous run.
    """

    def __init__(self, config: Config, models: List[BaseModel],
                 filter_classes: Optional[List[str]] = None,
                 output_path: Optional[str] = None,
                 resume_from: Optional[str] = None):
        self.config = config
        self.models = models
        self.filter_classes = filter_classes
        self.output_path = output_path
        self.results: List[DatasetResult] = []
        self._completed_keys: set = set()  # (dataset_name, model_name) pairs already done

        # Resume from previous run if provided
        if resume_from and os.path.exists(resume_from):
            self._load_previous_results(resume_from)
            # Use same file for incremental saves if no explicit output_path
            if not self.output_path:
                self.output_path = resume_from

    def _load_previous_results(self, filepath: str):
        """Load results from a previous Excel file to skip already-completed experiments."""
        try:
            df = pd.read_excel(filepath, sheet_name='All Results')
            loaded = 0
            for _, row in df.iterrows():
                result = DatasetResult(
                    dataset_name=row['Dataset'],
                    n_samples=int(row['Samples']),
                    n_features=int(row['Features']),
                    n_classes=int(row['Classes']),
                    model_name=row['Model'],
                    best_params=row['Best Params'] if 'Best Params' in row else {},
                    cv_f1=float(row['CV F1']),
                    test_f1=float(row['Test F1']),
                    holdout_f1=float(row['Holdout F1']),
                    test_accuracy=float(row['Test Acc']),
                    holdout_accuracy=float(row['Holdout Acc']),
                )
                self.results.append(result)
                self._completed_keys.add((row['Dataset'], row['Model']))
                loaded += 1
            console.print(f"[bold green]Resumed {loaded} results from: {filepath}[/bold green]")
            console.print(f"[bold green]Experiments already completed: {len(self._completed_keys)}[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Failed to load previous results: {e}[/bold red]")

    def _is_completed(self, dataset_name: str, model_name: str) -> bool:
        """Check if a (dataset, model) experiment was already completed."""
        return (dataset_name, model_name) in self._completed_keys

    def _save_incremental(self):
        """Save current results to Excel after each experiment."""
        if not self.output_path:
            return
        try:
            self.export_excel(self.output_path)
        except Exception as e:
            console.print(f"[bold red]Warning: incremental save failed: {e}[/bold red]")

    def run(self, dataset_paths: Dict[str, str]) -> "MultiDatasetRunner":
        if self.filter_classes:
            console.print(f"\n[bold yellow]Filtering to classes: {self.filter_classes}[/bold yellow]")

        # Count total experiments for progress tracking
        total = sum(1 for name in dataset_paths if os.path.exists(dataset_paths[name])) * len(self.models)
        skipped = sum(1 for name in dataset_paths for m in self.models
                      if self._is_completed(name, m.name))
        remaining = total - skipped
        if skipped > 0:
            console.print(f"[bold yellow]Skipping {skipped}/{total} already completed. Remaining: {remaining}[/bold yellow]")

        completed_count = 0
        for name, path in dataset_paths.items():
            if not os.path.exists(path):
                print(f"Skipping {name}: file not found")
                continue

            # Check if ALL models for this dataset are already done
            models_to_run = [m for m in self.models if not self._is_completed(name, m.name)]
            if not models_to_run:
                console.print(f"[dim]Skipping {name}: all models already completed[/dim]")
                continue

            print(f"\n{'#'*70}")
            print(f"# DATASET: {name}")
            print(f"{'#'*70}")

            data = DataLoader(path, filter_classes=self.filter_classes).load()

            for model_def in models_to_run:
                completed_count += 1
                console.print(f"[bold cyan]  [{completed_count}/{remaining}] {model_def.name} on {name}[/bold cyan]")

                trainer = ModelTrainer(model_def, self.config)
                result = trainer.train(data, evaluate_holdout=True)

                new_result = DatasetResult(
                    dataset_name=name,
                    n_samples=len(data.df),
                    n_features=len(data.feature_cols),
                    n_classes=len(data.classes),
                    model_name=result.model_name,
                    best_params=result.best_params,
                    cv_f1=result.cv_f1,
                    test_f1=result.test_f1,
                    holdout_f1=result.holdout_f1 or 0.0,
                    test_accuracy=result.test_accuracy,
                    holdout_accuracy=result.holdout_accuracy or 0.0
                )
                self.results.append(new_result)
                self._completed_keys.add((name, model_def.name))

                # Save after each experiment
                self._save_incremental()

        return self

    def summary(self) -> pd.DataFrame:
        """Generate comparison table."""
        rows = [{
            "Dataset": r.dataset_name,
            "Features": r.n_features,
            "Samples": r.n_samples,
            "Classes": r.n_classes,
            "Model": r.model_name,
            "CV F1": r.cv_f1,
            "Test F1": r.test_f1,
            "Holdout F1": r.holdout_f1,
            "Test Acc": r.test_accuracy,
            "Holdout Acc": r.holdout_accuracy,
            "Best Params": str(r.best_params)
        } for r in self.results]

        df = pd.DataFrame(rows)
        return df

    def print_summary(self):
        """Print formatted summary with rich tables."""
        df = self.summary()

        console.print("\n")
        console.rule("[bold blue]RESULTS COMPARISON - MACRO F1 SCORE (State of Art Metric)", style="blue")
        console.print("\n")

        # Table per model
        for model_name in df['Model'].unique():
            model_df = df[df['Model'] == model_name].copy()
            model_df = model_df.sort_values('Holdout F1', ascending=False)

            # Find best holdout for highlighting
            best_holdout = model_df['Holdout F1'].max()

            table = Table(
                title=f"[bold cyan]{model_name}[/bold cyan]",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold magenta"
            )
            table.add_column("Dataset", style="white", min_width=30)
            table.add_column("Feat", justify="center", style="dim")
            table.add_column("Samples", justify="center", style="dim")
            table.add_column("CV F1", justify="center", style="yellow")
            table.add_column("Test F1", justify="center", style="yellow")
            table.add_column("Holdout F1", justify="center", style="bold green")
            table.add_column("Holdout Acc", justify="center", style="dim")

            for _, row in model_df.iterrows():
                holdout_style = "bold green" if row['Holdout F1'] == best_holdout else "green"
                table.add_row(
                    row['Dataset'],
                    str(row['Features']),
                    f"{row['Samples']:,}",
                    f"{row['CV F1']:.4f}",
                    f"{row['Test F1']:.4f}",
                    f"[{holdout_style}]{row['Holdout F1']:.4f}[/{holdout_style}]",
                    f"{row['Holdout Acc']:.4f}"
                )

            console.print(table)
            console.print("\n")

        # Best overall per model
        console.rule("[bold green]BEST CONFIGURATION PER MODEL", style="green")

        best_table = Table(box=box.DOUBLE_EDGE, show_header=True, header_style="bold white on blue")
        best_table.add_column("Model", style="cyan", min_width=15)
        best_table.add_column("Best Dataset", style="white", min_width=30)
        best_table.add_column("Holdout F1", justify="center", style="bold green")
        best_table.add_column("Holdout Acc", justify="center", style="dim")
        best_table.add_column("Best Params", style="dim", max_width=50)

        for model_name in df['Model'].unique():
            model_results = [r for r in self.results if r.model_name == model_name]
            best = max(model_results, key=lambda r: r.holdout_f1)
            params_str = str(best.best_params)[:50] + "..." if len(str(best.best_params)) > 50 else str(best.best_params)
            best_table.add_row(
                best.model_name,
                best.dataset_name,
                f"{best.holdout_f1:.4f}",
                f"{best.holdout_accuracy:.4f}",
                params_str
            )

        console.print(best_table)

        # Overall best
        overall_best = max(self.results, key=lambda r: r.holdout_f1)
        console.print("\n")
        console.print(Panel(
            f"[bold white]Dataset:[/bold white] {overall_best.dataset_name}\n"
            f"[bold white]Model:[/bold white] {overall_best.model_name}\n"
            f"[bold white]Classes:[/bold white] {overall_best.n_classes}\n"
            f"[bold white]Holdout Macro F1:[/bold white] [bold green]{overall_best.holdout_f1:.4f}[/bold green]\n"
            f"[bold white]Holdout Accuracy:[/bold white] {overall_best.holdout_accuracy:.4f}\n"
            f"[bold white]Best Params:[/bold white] {overall_best.best_params}",
            title="[bold yellow]OVERALL BEST CONFIGURATION[/bold yellow]",
            border_style="yellow"
        ))

        # State of art comparison
        console.print("\n")
        console.print(Panel(
            f"[bold white]State of Art (Paper):[/bold white] Macro F1 = 0.66 (Transformer + TBPE)\n"
            f"[bold white]Our Best Result:[/bold white] Macro F1 = [bold green]{overall_best.holdout_f1:.4f}[/bold green] ({overall_best.model_name})",
            title="[bold cyan]STATE OF ART COMPARISON[/bold cyan]",
            border_style="cyan"
        ))

        return df

    def export_excel(self, output_path: str) -> str:
        """Export results to Excel file. If path already ends in .xlsx, use as-is."""
        df = self.summary()

        if not output_path.endswith('.xlsx'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{output_path}_{timestamp}.xlsx"

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Full results
            df.to_excel(writer, sheet_name='All Results', index=False)

            # Best per model summary
            best_rows = []
            for model_name in df['Model'].unique():
                model_results = [r for r in self.results if r.model_name == model_name]
                best = max(model_results, key=lambda r: r.holdout_f1)
                best_rows.append({
                    "Model": best.model_name,
                    "Best Dataset": best.dataset_name,
                    "Features": best.n_features,
                    "Samples": best.n_samples,
                    "Classes": best.n_classes,
                    "CV F1": best.cv_f1,
                    "Test F1": best.test_f1,
                    "Holdout F1": best.holdout_f1,
                    "Holdout Acc": best.holdout_accuracy,
                    "Best Params": str(best.best_params)
                })
            pd.DataFrame(best_rows).to_excel(writer, sheet_name='Best Per Model', index=False)

            # Overall best
            overall_best = max(self.results, key=lambda r: r.holdout_f1)
            pd.DataFrame([{
                "Model": overall_best.model_name,
                "Dataset": overall_best.dataset_name,
                "Features": overall_best.n_features,
                "Classes": overall_best.n_classes,
                "Holdout F1": overall_best.holdout_f1,
                "Holdout Acc": overall_best.holdout_accuracy,
                "State of Art F1": 0.66,
                "Improvement": overall_best.holdout_f1 - 0.66,
                "Best Params": str(overall_best.best_params)
            }]).to_excel(writer, sheet_name='Overall Best', index=False)

        console.print(f"\n[bold green]Results exported to: {output_path}[/bold green]")
        return output_path


# =============================================================================
# ENSEMBLE EVALUATOR
# =============================================================================

class EnsembleEvaluator:
    """Evaluates ensemble methods (Voting, Stacking) using best individual models."""

    def __init__(self, config: Config):
        self.config = config
        self.results: List[DatasetResult] = []

    def evaluate(self, runner: MultiDatasetRunner, dataset_paths: Dict[str, str]) -> "EnsembleEvaluator":
        """Build ensembles from top 3 models per dataset and evaluate."""
        from sklearn.model_selection import cross_val_score, StratifiedGroupKFold

        for name, path in dataset_paths.items():
            if not os.path.exists(path):
                continue

            # Skip if ensembles already completed for this dataset
            if runner._is_completed(name, "Voting (Top3)") and runner._is_completed(name, "Stacking (Top3)"):
                console.print(f"[dim]Skipping ensembles for {name}: already completed[/dim]")
                continue

            # Get results for this dataset, sorted by holdout F1
            dataset_results = sorted(
                [r for r in runner.results if r.dataset_name == name],
                key=lambda r: r.holdout_f1, reverse=True
            )
            if len(dataset_results) < 3:
                print(f"Skipping ensembles for {name}: need at least 3 models")
                continue

            top3 = dataset_results[:3]
            data = DataLoader(path, filter_classes=runner.filter_classes).load()
            X_train, y_train = data.train
            X_test, y_test = data.test
            X_holdout, y_holdout = data.holdout
            train_groups = data.train_groups
            sgkf = StratifiedGroupKFold(n_splits=self.config.cv_folds)

            # Rebuild top 3 models with their best params
            model_map = {
                "KNN": KNNModel, "RandomForest": RandomForestModel, "SVM": SVMModel,
                "XGBoost": XGBoostModel, "LightGBM": LightGBMModel, "CatBoost": CatBoostModel,
                "ExtraTrees": ExtraTreesModel, "GradientBoosting": GradientBoostingModel,
                "HistGradientBoosting": HistGradientBoostingModel, "MLP": MLPModel,
                "AdaBoost": AdaBoostModel, "LogisticRegression": LogisticRegressionModel,
            }
            estimators = []
            for r in top3:
                model_cls = model_map.get(r.model_name)
                if model_cls:
                    model_instance = model_cls().create_model(r.best_params, self.config.random_state)
                    estimators.append((r.model_name, model_instance))

            if len(estimators) < 3:
                continue

            ensemble_names = [r.model_name for r in top3]
            console.print(f"\n[bold cyan]Ensembles for {name} using: {ensemble_names}[/bold cyan]")

            # --- Soft Voting ---
            try:
                voting = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
                cv_scores = cross_val_score(voting, X_train, y_train, cv=sgkf,
                                            scoring='f1_macro', n_jobs=self.config.n_jobs,
                                            groups=train_groups)
                voting.fit(X_train, y_train)

                test_f1 = f1_score(y_test, voting.predict(X_test), average='macro')
                holdout_f1 = f1_score(y_holdout, voting.predict(X_holdout), average='macro')
                test_acc = accuracy_score(y_test, voting.predict(X_test))
                holdout_acc = accuracy_score(y_holdout, voting.predict(X_holdout))

                print(f"  Voting: CV={cv_scores.mean():.4f} | Test={test_f1:.4f} | Holdout={holdout_f1:.4f}")

                self.results.append(DatasetResult(
                    dataset_name=name, n_samples=len(data.df), n_features=len(data.feature_cols),
                    n_classes=len(data.classes), model_name="Voting (Top3)",
                    best_params={"estimators": ensemble_names, "voting": "soft"},
                    cv_f1=cv_scores.mean(), test_f1=test_f1, holdout_f1=holdout_f1,
                    test_accuracy=test_acc, holdout_accuracy=holdout_acc
                ))
            except Exception as e:
                print(f"  Voting failed: {e}")

            # --- Stacking ---
            try:
                stacking = StackingClassifier(
                    estimators=estimators,
                    final_estimator=LogisticRegression(max_iter=1000, random_state=self.config.random_state),
                    cv=5, n_jobs=-1
                )
                cv_scores = cross_val_score(stacking, X_train, y_train, cv=sgkf,
                                            scoring='f1_macro', n_jobs=self.config.n_jobs,
                                            groups=train_groups)
                stacking.fit(X_train, y_train)

                test_f1 = f1_score(y_test, stacking.predict(X_test), average='macro')
                holdout_f1 = f1_score(y_holdout, stacking.predict(X_holdout), average='macro')
                test_acc = accuracy_score(y_test, stacking.predict(X_test))
                holdout_acc = accuracy_score(y_holdout, stacking.predict(X_holdout))

                print(f"  Stacking: CV={cv_scores.mean():.4f} | Test={test_f1:.4f} | Holdout={holdout_f1:.4f}")

                self.results.append(DatasetResult(
                    dataset_name=name, n_samples=len(data.df), n_features=len(data.feature_cols),
                    n_classes=len(data.classes), model_name="Stacking (Top3)",
                    best_params={"estimators": ensemble_names, "meta": "LogisticRegression"},
                    cv_f1=cv_scores.mean(), test_f1=test_f1, holdout_f1=holdout_f1,
                    test_accuracy=test_acc, holdout_accuracy=holdout_acc
                ))
            except Exception as e:
                print(f"  Stacking failed: {e}")

        return self


# =============================================================================
# MAIN
# =============================================================================

# ============== CONFIGURE MODELS TO RUN ==============
# Comment/uncomment to enable/disable models

MODELS_TO_RUN: List[BaseModel] = [
    # --- 7 Competitive Models (Holdout F1 > 0.60) ---
    # Tree Ensembles
    RandomForestModel(),
    ExtraTreesModel(),
    # Gradient Boosting
    GradientBoostingModel(),
    XGBoostModel(),
    LightGBMModel(),
    CatBoostModel(),
    HistGradientBoostingModel(),
    # --- Dropped (Holdout F1 < 0.60 across all datasets) ---
    # KNNModel(),            # 0.58 max
    # LogisticRegressionModel(),  # 0.55 max
    # AdaBoostModel(),       # 0.58 max
    # MLPModel(),            # 0.55 max
    # SVMModel(),            # 0.59 max + very slow
]

# =====================================================

# Set to True to run ensemble methods after individual models
RUN_ENSEMBLES = True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="FWOD Genre Classification Pipeline")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a previous results .xlsx to resume from")
    parser.add_argument("--all-classes", action="store_true",
                        help="Use all 20 classes instead of 4-class state-of-art comparison")
    args = parser.parse_args()

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(PROJECT_ROOT, "PAPER", "DATA")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "PAPER", "RESULTS")

    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # All available datasets
    DATASETS = {
        "1-pattern (16 feat)": os.path.join(DATA_DIR, "fwod_representations.xlsx"),
        "2-pattern inclusive (32 feat)": os.path.join(DATA_DIR, "fwod_pattern_2_inclusive.xlsx"),
        "4-pattern inclusive (64 feat)": os.path.join(DATA_DIR, "fwod_pattern_4_inclusive.xlsx"),
        "5-pattern inclusive (80 feat)": os.path.join(DATA_DIR, "fwod_pattern_5_inclusive.xlsx"),
        "6-pattern inclusive (96 feat)": os.path.join(DATA_DIR, "fwod_pattern_6_inclusive.xlsx"),
        "7-pattern inclusive (112 feat)": os.path.join(DATA_DIR, "fwod_pattern_7_inclusive.xlsx"),
        "8-pattern inclusive (128 feat)": os.path.join(DATA_DIR, "fwod_pattern_8_inclusive.xlsx"),
        "16-pattern inclusive (256 feat)": os.path.join(DATA_DIR, "fwod_pattern_16_inclusive.xlsx"),
    }

    config = Config(
        random_state=42,
        n_trials=60,
        cv_folds=8,
        n_jobs=-1
    )

    # ============== STATE OF ART COMPARISON MODE ==============
    USE_STATE_OF_ART_CLASSES = not args.all_classes
    # ==========================================================

    filter_classes = STATE_OF_ART_CLASSES if USE_STATE_OF_ART_CLASSES else None

    # Determine output file path
    classes_suffix = "_4classes" if USE_STATE_OF_ART_CLASSES else "_all_classes"
    if args.resume:
        output_file = args.resume
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(RESULTS_DIR, f"classification_results{classes_suffix}_{timestamp}.xlsx")

    console.print("\n")
    console.rule("[bold magenta]FWOD GENRE CLASSIFICATION PIPELINE", style="magenta")
    if USE_STATE_OF_ART_CLASSES:
        console.print(f"[bold yellow]Mode: State of Art Comparison (4 classes: {STATE_OF_ART_CLASSES})[/bold yellow]")
    console.print(f"[bold yellow]Metric: Macro F1 Score[/bold yellow]")
    console.print(f"[bold yellow]Config: {config.n_trials} trials, {config.cv_folds} CV folds[/bold yellow]")
    console.print(f"[bold yellow]Models: {len(MODELS_TO_RUN)} individual + {'ensembles' if RUN_ENSEMBLES else 'no ensembles'}[/bold yellow]")
    console.print(f"[bold yellow]Output: {output_file}[/bold yellow]")
    if args.resume:
        console.print(f"[bold green]Resuming from: {args.resume}[/bold green]")
    console.print("\n")

    runner = MultiDatasetRunner(
        config, MODELS_TO_RUN,
        filter_classes=filter_classes,
        output_path=output_file,
        resume_from=args.resume
    )
    runner.run(DATASETS)

    # Run ensemble methods
    if RUN_ENSEMBLES:
        console.print("\n")
        console.rule("[bold magenta]ENSEMBLE METHODS", style="magenta")
        ensemble_eval = EnsembleEvaluator(config)
        ensemble_eval.evaluate(runner, DATASETS)
        # Merge ensemble results into runner and save
        runner.results.extend(ensemble_eval.results)
        runner._save_incremental()

    runner.print_summary()

    # Final export (ensures Best Per Model and Overall Best sheets are up to date)
    runner.export_excel(output_file)
    console.print(f"\n[bold green]Final results saved to: {output_file}[/bold green]")


if __name__ == "__main__":
    main()
