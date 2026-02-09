"""
TDG Classification Pipeline
===========================
Drum pattern genre classification using FWOD (Flattened Weighted Onset Distribution).

This pipeline:
1. Parses MIDI files from the Magenta Groove MIDI Dataset
2. Converts them to HV-list representation (hit-velocity pairs)
3. Flattens to FWOD representation (16 normalized values per bar)
4. Optionally aggregates multiple patterns
5. Trains and evaluates ML models for genre classification

Reference: https://arxiv.org/pdf/2407.17536
"""

# Suppress TensorFlow warnings (must be before imports)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
import pickle
import warnings
from typing import Dict, List, Tuple, Optional

import mido
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# TensorFlow imports are lazy-loaded in train_lstm() and train_rnn() methods
# to avoid hanging on macOS during module import

warnings.filterwarnings('ignore')


# =============================================================================
# CONSTANTS
# =============================================================================

# General MIDI drum mapping
GM_DICT = {
    22: ["Closed Hi-Hat edge", "high", 42, "CH", 3, 42, 42, 42],
    26: ["Open Hi-Hat edge", "high", 46, "OH", 4, 46, 46, 42],
    35: ["Acoustic Bass Drum", "low", 36, "K", 1, 36, 36, 36],
    36: ["Bass Drum 1", "low", 36, "K", 1, 36, 36, 36],
    37: ["Side Stick", "mid", 37, "RS", 6, 37, 37, 38],
    38: ["Acoustic Snare", "mid", 38, "SN", 2, 38, 38, 38],
    39: ["Hand Clap", "mid", 39, "CP", 5, 39, 39, 38],
    40: ["Electric Snare", "mid", 38, "SN", 2, 38, 38, 38],
    41: ["Low Floor Tom", "low", 45, "LT", 7, 45, 45, 36],
    42: ["Closed Hi Hat", "high", 42, "CH", 3, 42, 42, 42],
    43: ["High Floor Tom", "mid", 45, "HT", 8, 45, 45, 38],
    44: ["Pedal Hi-Hat", "high", 46, "OH", 4, 46, 46, 42],
    45: ["Low Tom", "low", 45, "LT", 7, 45, 45, 36],
    46: ["Open Hi-Hat", "high", 46, "OH", 4, 46, 46, 42],
    47: ["Low-Mid Tom", "low", 47, "MT", 7, 45, 47, 36],
    48: ["Hi-Mid Tom", "mid", 47, "MT", 7, 50, 50, 38],
    49: ["Crash Cymbal 1", "high", 49, "CC", 4, 46, 42, 42],
    50: ["High Tom", "mid", 50, "HT", 8, 50, 50, 38],
    51: ["Ride Cymbal 1", "high", 51, "RC", -1, 42, 51, 42],
    52: ["Chinese Cymbal", "high", 52, "", -1, 46, 51, 42],
    53: ["Ride Bell", "high", 53, "", -1, 42, 51, 42],
    54: ["Tambourine", "high", 54, "", -1, 42, 69, 42],
    55: ["Splash Cymbal", "high", 55, "OH", 4, 46, 42, 42],
    56: ["Cowbell", "high", 56, "CB", -1, 37, 56, 42],
    57: ["Crash Cymbal 2", "high", 57, "CC", 4, 46, 42, 42],
    58: ["Vibraslap", "mid", 58, "VS", 6, 37, 37, 42],
    59: ["Ride Cymbal 2", "high", 59, "RC", 3, 42, 51, 42],
    60: ["Hi Bongo", "high", 60, "LB", 8, 45, 63, 42],
    61: ["Low Bongo", "mid", 61, "HB", 7, 45, 64, 38],
    62: ["Mute Hi Conga", "mid", 62, "MC", 8, 50, 62, 38],
    63: ["Open Hi Conga", "high", 63, "HC", 8, 50, 63, 42],
    64: ["Low Conga", "low", 64, "LC", 7, 45, 64, 36],
    65: ["High Timbale", "mid", 65, "", 8, 45, 63, 38],
    66: ["Low Timbale", "low", 66, "", 7, 45, 64, 36],
    67: ["High Agogo", "high", 67, "", -1, 37, 56, 42],
    68: ["Low Agogo", "mid", 68, "", -1, 37, 56, 38],
    69: ["Cabasa", "high", 69, "MA", -1, 42, 69, 42],
    70: ["Maracas", "high", 69, "MA", -1, 42, 69, 42],
    71: ["Short Whistle", "high", 71, "", -1, 37, 56, 42],
    72: ["Long Whistle", "high", 72, "", -1, 37, 56, 42],
    73: ["Short Guiro", "high", 73, "", -1, 42, 42, 42],
    74: ["Long Guiro", "high", 74, "", -1, 46, 46, 42],
    75: ["Claves", "high", 75, "", -1, 37, 75, 42],
    76: ["Hi Wood Block", "high", 76, "", 8, 50, 63, 42],
    77: ["Low Wood Block", "mid", 77, "", 7, 45, 64, 38],
    78: ["Mute Cuica", "high", 78, "", -1, 50, 62, 42],
    79: ["Open Cuica", "high", 79, "", -1, 45, 63, 42],
    80: ["Mute Triangle", "high", 80, "", -1, 37, 75, 42],
    81: ["Open Triangle", "high", 81, "", -1, 37, 75, 42],
}

# Instrument categories for weighted flattening
LOWS = [35, 36, 41, 45, 47, 64, 66]
MIDS = [37, 38, 39, 40, 43, 48, 50, 61, 62, 65, 68, 77]
HIGHS = [22, 26, 42, 44, 46, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
         63, 67, 69, 70, 71, 72, 73, 74, 75, 76, 78, 79, 80, 81]

# Instrument weights (low=3x, mid=2x, high=1x)
WEIGHT_LOW = 3
WEIGHT_MID = 2
WEIGHT_HIGH = 1


# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def midifile2hv_list(file_name: str, mapping: str = "allinstruments") -> List[List]:
    """
    Convert a MIDI file to HV-list representation.

    Args:
        file_name: Path to the MIDI file
        mapping: Instrument mapping type ("allinstruments", "16instruments", "8instruments", "3instruments")

    Returns:
        List of 16-step patterns, each containing (note, velocity) tuples per step
    """
    pattern = []
    mid = mido.MidiFile(file_name)
    sixteenth = mid.ticks_per_beat / 4
    acc = 0

    # Select column based on mapping
    column_map = {
        "allinstruments": 2,
        "16instruments": 6,
        "8instruments": 5,
        "3instruments": 7
    }
    column = column_map.get(mapping, 2)

    for i, track in enumerate(mid.tracks):
        for msg in track:
            acc += msg.time
            if msg.type == "note_on" and msg.velocity != 0:
                if msg.note in GM_DICT.keys():
                    midinote = GM_DICT[msg.note][column]
                    rounded_step = int((acc / sixteenth) + 0.45)
                    midivelocity = msg.velocity / 127
                    pattern.append((int(acc / sixteenth), midinote, midivelocity))

    if len(pattern) == 0:
        return []

    # Round pattern length to next multiple of 16
    if (rounded_step / 16) - (rounded_step // 16) != 0:
        pattern_len_in_steps = (rounded_step // 16) * 16 + 16
    else:
        pattern_len_in_steps = (rounded_step // 16) * 16

    # Create empty list of lists
    output_pattern = [[]] * pattern_len_in_steps

    # Group instruments and velocities by step
    for step in range(len(output_pattern)):
        output_pattern.append([(x[1], x[2]) for x in pattern if x[0] == step])
        output_pattern[step] = list(set(output_pattern[step]))
        output_pattern[step].sort()

    # Split pattern every 16 steps
    hv_lists_split = []
    for x in range(len(output_pattern) // 16):
        patt_fragment = output_pattern[x * 16:(x * 16) + 16]
        patt_density = sum([1 for x in patt_fragment if x != []])

        # Filter out patterns with less than 4 events
        if patt_density > 4:
            hv_lists_split.append(patt_fragment)

    return hv_lists_split


def flatten_hv_list(hv_list: List) -> np.ndarray:
    """
    Flatten an HV-list to FWOD representation.

    Applies weighted flattening: low instruments (3x), mid (2x), high (1x).

    Args:
        hv_list: 16-step HV-list with (note, velocity) tuples

    Returns:
        Normalized 16-element numpy array (FWOD representation)
    """
    flat = np.zeros([len(hv_list), 1])

    for i, step in enumerate(hv_list):
        step_weight = 0
        for onset in step:
            if onset[0] in LOWS:
                step_weight += onset[1] * WEIGHT_LOW
            elif onset[0] in MIDS:
                step_weight += onset[1] * WEIGHT_MID
            else:
                step_weight += onset[1] * WEIGHT_HIGH
        flat[i] = step_weight

    if max(flat) > 0:
        flat = flat / max(flat)

    return flat


def list_midi_files(directory: str) -> Dict[str, List[str]]:
    """
    List all MIDI files organized by genre folder.

    Args:
        directory: Path to the root directory containing genre folders

    Returns:
        Dictionary mapping genre names to lists of MIDI file paths
    """
    def filter_midi(file_list):
        return [x for x in file_list if x.endswith(".mid")]

    def list_all_elements(dir_path):
        elements = []
        for root, dirs, files in os.walk(dir_path):
            for name in files:
                elements.append(os.path.join(root, name))
        return filter_midi(elements)

    folders = [name for name in os.listdir(directory)
               if os.path.isdir(os.path.join(directory, name))]

    all_midi_files = {}
    for genre in folders:
        all_midi_files[genre] = list_all_elements(os.path.join(directory, genre))

    return all_midi_files


def create_fwod_dataset(midi_directory: str, output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Create FWOD dataset from MIDI files.

    Args:
        midi_directory: Path to directory containing genre folders with MIDI files
        output_file: Optional path to save the dataset as Excel file

    Returns:
        DataFrame with FWOD representations and class labels
    """
    all_midi_files = list_midi_files(midi_directory)
    fwod_representations = []

    for genre, midi_files in all_midi_files.items():
        for midi_path in midi_files:
            try:
                hv_lists = midifile2hv_list(midi_path, "allinstruments")
                for pattern_idx, hv_list in enumerate(hv_lists):
                    fwod = flatten_hv_list(hv_list)
                    element = {
                        'file': midi_path,
                        'sequence': pattern_idx,
                        'class': genre
                    }
                    for i in range(len(fwod)):
                        element[f'step_{i}'] = fwod[i][0]
                    fwod_representations.append(element)
            except Exception as e:
                print(f"Error processing {midi_path}: {e}")

    df = pd.DataFrame(fwod_representations)

    if output_file:
        df.to_excel(output_file, index=False)
        print(f"Dataset saved to {output_file}")

    return df


def create_aggregated_dataset(
    input_file: str,
    output_file: str,
    pattern_quantity: int,
    inclusive: bool = False
) -> pd.DataFrame:
    """
    Create aggregated dataset by combining multiple consecutive patterns.

    Args:
        input_file: Path to input FWOD Excel file
        output_file: Path to save output Excel file
        pattern_quantity: Number of patterns to combine (2, 4, 5, etc.)
        inclusive: Whether to use overlapping windows

    Returns:
        Aggregated DataFrame
    """
    data = pd.read_excel(input_file)

    required_columns = {"file", "sequence", "class"}
    step_columns = {f"step_{i}" for i in range(16)}
    all_columns = required_columns.union(step_columns)

    if not all_columns.issubset(data.columns):
        raise ValueError(f"Input dataset must contain columns: {all_columns}")

    new_rows = []

    for file, group in data.groupby("file"):
        group = group.sort_values("sequence")
        step_size = 1 if inclusive else pattern_quantity
        num_rows = len(group)

        for i in range(0, num_rows - pattern_quantity + 1, step_size):
            chunk = group.iloc[i:i + pattern_quantity]

            if len(chunk["class"].unique()) > 1:
                continue

            aggregated_row = {
                "file": file,
                "sequence": "-".join(map(str, chunk["sequence"])),
                "class": chunk["class"].iloc[0]
            }

            for j, row in enumerate(chunk.itertuples(index=False)):
                for k in range(16):
                    feature_name = f"feature_{j * 16 + k}"
                    aggregated_row[feature_name] = getattr(row, f"step_{k}")

            new_rows.append(aggregated_row)

    new_data = pd.DataFrame(new_rows)
    new_data.to_excel(output_file, index=False)
    print(f"Aggregated database saved to {output_file}")

    return new_data


# =============================================================================
# MODEL TRAINING FUNCTIONS
# =============================================================================

class DrumPatternClassifier:
    """
    Machine learning pipeline for drum pattern classification.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.results = {}

        # Define hyperparameter search spaces
        self.search_spaces = {
            "random_forest": {
                "n_estimators": Integer(50, 300),
                "max_depth": Integer(5, 50)
            },
            "svm": {
                "C": Real(0.1, 10),
                "gamma": Real(0.01, 1)
            },
            "xgboost": {
                "n_estimators": Integer(50, 300),
                "max_depth": Integer(3, 10),
                "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
                "subsample": Real(0.5, 1.0),
                "colsample_bytree": Real(0.5, 1.0)
            },
            "lightgbm": {
                "num_leaves": Integer(20, 100),
                "max_depth": Integer(3, 10),
                "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
                "n_estimators": Integer(50, 300),
                "subsample": Real(0.5, 1.0),
                "colsample_bytree": Real(0.5, 1.0)
            },
            "knn": {
                "n_neighbors": Integer(1, 30),
                "weights": Categorical(["uniform", "distance"]),
                "metric": Categorical(["minkowski", "manhattan", "euclidean"])
            }
        }

    def load_data(self, filepath: str, feature_prefix: str = "feature_") -> Tuple:
        """
        Load dataset and prepare train/test splits.

        Args:
            filepath: Path to Excel file with FWOD data
            feature_prefix: Prefix for feature columns ("feature_" or "step_")

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        dataset = pd.read_excel(filepath)

        X = dataset.filter(like=feature_prefix).values
        y = dataset['class'].values

        # Encode labels for models that require numeric labels
        y_encoded = self.label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=self.random_state, stratify=y_encoded
        )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.n_classes = len(np.unique(y_encoded))

        print(f"Dataset loaded: {len(X_train)} train, {len(X_test)} test samples")
        print(f"Number of classes: {self.n_classes}")
        print(f"Number of features: {X.shape[1]}")

        return X_train, X_test, y_train, y_test

    def train_random_forest(self, n_iter: int = 20) -> dict:
        """Train Random Forest with Bayesian optimization."""
        print("\n" + "="*50)
        print("Training Random Forest...")

        rf_model = RandomForestClassifier(random_state=self.random_state)

        opt_rf = BayesSearchCV(
            rf_model,
            self.search_spaces['random_forest'],
            n_iter=n_iter,
            cv=5,
            scoring='accuracy',
            random_state=self.random_state
        )
        opt_rf.fit(self.X_train, self.y_train)

        print(f"Best parameters: {opt_rf.best_params_}")
        print(f"Best CV accuracy: {opt_rf.best_score_:.4f}")

        y_pred = opt_rf.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Test accuracy: {test_accuracy:.4f}")

        self.models['random_forest'] = opt_rf.best_estimator_
        self.results['random_forest'] = {
            'best_params': opt_rf.best_params_,
            'cv_accuracy': opt_rf.best_score_,
            'test_accuracy': test_accuracy
        }

        return self.results['random_forest']

    def train_svm(self, n_iter: int = 20) -> dict:
        """Train SVM with Bayesian optimization."""
        print("\n" + "="*50)
        print("Training SVM...")

        svm_model = SVC(kernel="rbf", random_state=self.random_state)

        opt_svm = BayesSearchCV(
            svm_model,
            self.search_spaces['svm'],
            n_iter=n_iter,
            cv=5,
            scoring='accuracy',
            random_state=self.random_state
        )
        opt_svm.fit(self.X_train, self.y_train)

        print(f"Best parameters: {opt_svm.best_params_}")
        print(f"Best CV accuracy: {opt_svm.best_score_:.4f}")

        y_pred = opt_svm.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Test accuracy: {test_accuracy:.4f}")

        self.models['svm'] = opt_svm.best_estimator_
        self.results['svm'] = {
            'best_params': opt_svm.best_params_,
            'cv_accuracy': opt_svm.best_score_,
            'test_accuracy': test_accuracy
        }

        return self.results['svm']

    def train_xgboost(self, n_iter: int = 20) -> dict:
        """Train XGBoost with Bayesian optimization."""
        print("\n" + "="*50)
        print("Training XGBoost...")

        xgb_model = XGBClassifier(
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric="logloss"
        )

        opt_xgb = BayesSearchCV(
            xgb_model,
            self.search_spaces['xgboost'],
            n_iter=n_iter,
            cv=5,
            scoring='accuracy',
            random_state=self.random_state
        )
        opt_xgb.fit(self.X_train, self.y_train)

        print(f"Best parameters: {opt_xgb.best_params_}")
        print(f"Best CV accuracy: {opt_xgb.best_score_:.4f}")

        y_pred = opt_xgb.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Test accuracy: {test_accuracy:.4f}")

        self.models['xgboost'] = opt_xgb.best_estimator_
        self.results['xgboost'] = {
            'best_params': opt_xgb.best_params_,
            'cv_accuracy': opt_xgb.best_score_,
            'test_accuracy': test_accuracy
        }

        return self.results['xgboost']

    def train_lightgbm(self, n_iter: int = 20) -> dict:
        """Train LightGBM with Bayesian optimization."""
        print("\n" + "="*50)
        print("Training LightGBM...")

        lgb_model = LGBMClassifier(random_state=self.random_state, verbose=-1)

        opt_lgb = BayesSearchCV(
            lgb_model,
            self.search_spaces['lightgbm'],
            n_iter=n_iter,
            cv=5,
            scoring='accuracy',
            random_state=self.random_state
        )
        opt_lgb.fit(self.X_train, self.y_train)

        print(f"Best parameters: {opt_lgb.best_params_}")
        print(f"Best CV accuracy: {opt_lgb.best_score_:.4f}")

        y_pred = opt_lgb.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Test accuracy: {test_accuracy:.4f}")

        self.models['lightgbm'] = opt_lgb.best_estimator_
        self.results['lightgbm'] = {
            'best_params': opt_lgb.best_params_,
            'cv_accuracy': opt_lgb.best_score_,
            'test_accuracy': test_accuracy
        }

        return self.results['lightgbm']

    def train_knn(self, n_iter: int = 20) -> dict:
        """Train KNN with Bayesian optimization."""
        print("\n" + "="*50)
        print("Training KNN...")

        knn_model = KNeighborsClassifier()

        opt_knn = BayesSearchCV(
            knn_model,
            self.search_spaces['knn'],
            n_iter=n_iter,
            cv=5,
            scoring='accuracy',
            random_state=self.random_state
        )
        opt_knn.fit(self.X_train, self.y_train)

        print(f"Best parameters: {opt_knn.best_params_}")
        print(f"Best CV accuracy: {opt_knn.best_score_:.4f}")

        y_pred = opt_knn.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Test accuracy: {test_accuracy:.4f}")

        self.models['knn'] = opt_knn.best_estimator_
        self.results['knn'] = {
            'best_params': opt_knn.best_params_,
            'cv_accuracy': opt_knn.best_score_,
            'test_accuracy': test_accuracy
        }

        return self.results['knn']

    def train_lstm(self, epochs: int = 100, batch_size: int = 32) -> dict:
        """Train LSTM neural network."""
        print("\n" + "="*50)
        print("Training LSTM...")
        print("Loading TensorFlow...")

        # Lazy import TensorFlow to avoid hanging on module load
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
        from tensorflow.keras.regularizers import l2
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.callbacks import EarlyStopping

        # Reshape for LSTM (samples, timesteps, features)
        X_train_lstm = self.X_train.reshape((self.X_train.shape[0], self.X_train.shape[1], 1))
        X_test_lstm = self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))

        y_train_cat = to_categorical(self.y_train, num_classes=self.n_classes)
        y_test_cat = to_categorical(self.y_test, num_classes=self.n_classes)

        model = Sequential([
            LSTM(64, activation='relu', return_sequences=True,
                 input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
            BatchNormalization(),
            Dropout(0.4),
            LSTM(64, activation='relu', return_sequences=True, kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.4),
            LSTM(32, activation='relu'),
            Dense(64, activation='swish', kernel_regularizer=l2(0.01)),
            Dense(self.n_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(
            X_train_lstm, y_train_cat,
            validation_data=(X_test_lstm, y_test_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )

        test_loss, test_accuracy = model.evaluate(X_test_lstm, y_test_cat, verbose=0)
        print(f"Test accuracy: {test_accuracy:.4f}")

        self.models['lstm'] = model
        self.results['lstm'] = {
            'test_accuracy': test_accuracy,
            'history': history.history
        }

        return self.results['lstm']

    def train_rnn(self, epochs: int = 50, batch_size: int = 32) -> dict:
        """Train Bidirectional RNN."""
        print("\n" + "="*50)
        print("Training Bidirectional RNN...")
        print("Loading TensorFlow...")

        # Lazy import TensorFlow to avoid hanging on module load
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.callbacks import EarlyStopping

        # Reshape for RNN
        X_train_rnn = self.X_train.reshape((self.X_train.shape[0], self.X_train.shape[1], 1))
        X_test_rnn = self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))

        y_train_cat = to_categorical(self.y_train, num_classes=self.n_classes)
        y_test_cat = to_categorical(self.y_test, num_classes=self.n_classes)

        model = Sequential([
            Bidirectional(LSTM(128, activation='tanh', return_sequences=True),
                         input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])),
            Dropout(0.3),
            LSTM(64, activation='tanh', return_sequences=False),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(self.n_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(
            X_train_rnn, y_train_cat,
            validation_data=(X_test_rnn, y_test_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )

        test_loss, test_accuracy = model.evaluate(X_test_rnn, y_test_cat, verbose=0)
        print(f"Test accuracy: {test_accuracy:.4f}")

        self.models['rnn'] = model
        self.results['rnn'] = {
            'test_accuracy': test_accuracy,
            'history': history.history
        }

        return self.results['rnn']

    def train_all_models(self, n_iter: int = 20) -> dict:
        """Train all models and return results."""
        # self.train_random_forest(n_iter)
        # self.train_svm(n_iter)
        # self.train_xgboost(n_iter)
        # self.train_lightgbm(n_iter)
        self.train_knn(n_iter)
        # self.train_lstm()
        # self.train_rnn()

        return self.results

    def print_summary(self):
        """Print summary of all model results."""
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)

        results_list = []
        for model_name, result in self.results.items():
            results_list.append({
                'Model': model_name,
                'Test Accuracy': result['test_accuracy']
            })

        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values('Test Accuracy', ascending=False)
        print(results_df.to_string(index=False))

        return results_df

    def get_classification_report(self, model_name: str) -> str:
        """Get detailed classification report for a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")

        model = self.models[model_name]

        if model_name in ['lstm', 'rnn']:
            X_test_reshaped = self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))
            y_pred = np.argmax(model.predict(X_test_reshaped), axis=1)
        else:
            y_pred = model.predict(self.X_test)

        y_test_decoded = self.label_encoder.inverse_transform(self.y_test)
        y_pred_decoded = self.label_encoder.inverse_transform(y_pred)

        return classification_report(y_test_decoded, y_pred_decoded)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution pipeline.

    Usage:
        1. Data Creation (if MIDI files available):
           df = create_fwod_dataset("magenta midi", "fwod_representations.xlsx")

        2. Pattern Aggregation (optional):
           create_aggregated_dataset(
               "fwod_representations_clean.xlsx",
               "data_pattern_5_inclusive.xlsx",
               pattern_quantity=5,
               inclusive=True
           )

        3. Model Training:
           classifier = DrumPatternClassifier()
           classifier.load_data("data_pattern_5_inclusive.xlsx")
           classifier.train_all_models()
           classifier.print_summary()
    """
    print("TDG Classification Pipeline")
    print("="*60)
    print("\nThis pipeline classifies drum patterns into 20 music genres")
    print("using FWOD (Flattened Weighted Onset Distribution) representation.")

    # Train and evaluate models using existing dataset
    classifier = DrumPatternClassifier()
    classifier.load_data("data_pattern_5_inclusive.xlsx")
    classifier.train_all_models(n_iter=20)
    classifier.print_summary()

    # Get detailed report for best model
    print(classifier.get_classification_report('knn'))


if __name__ == "__main__":
    main()
