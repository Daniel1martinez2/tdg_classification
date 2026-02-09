"""
FWOD Converter
==============
Converts MIDI drum patterns to FWOD (Flattened Weighted Onset Distribution) representation.

This module implements the conversion pipeline described in the paper:
    MIDI File → HV-List → FWOD Representation

The FWOD representation applies weighted flattening based on instrument categories:
    - Low instruments (kick, toms): weight = 3
    - Mid instruments (snare, claps): weight = 2
    - High instruments (hi-hat, cymbals): weight = 1

Author: Daniel Martinez
"""

import os
from typing import Dict, List, Optional, Tuple

import mido
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# =============================================================================
# CONSTANTS
# =============================================================================

# Instrument category weights
WEIGHT_LOW = 3
WEIGHT_MID = 2
WEIGHT_HIGH = 1

# Instrument categories (General MIDI drum note numbers)
INSTRUMENTS_LOW = [35, 36, 41, 45, 47, 64, 66]
INSTRUMENTS_MID = [37, 38, 39, 40, 43, 48, 50, 61, 62, 65, 68, 77]
INSTRUMENTS_HIGH = [
    22, 26, 42, 44, 46, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
    63, 67, 69, 70, 71, 72, 73, 74, 75, 76, 78, 79, 80, 81
]

# General MIDI Drum Map
# Format: note_number -> [name, category, simplified_note, abbrev, ...]
GM_DRUM_MAP = {
    22: ["Closed Hi-Hat edge", "high", 42],
    26: ["Open Hi-Hat edge", "high", 46],
    35: ["Acoustic Bass Drum", "low", 36],
    36: ["Bass Drum 1", "low", 36],
    37: ["Side Stick", "mid", 37],
    38: ["Acoustic Snare", "mid", 38],
    39: ["Hand Clap", "mid", 39],
    40: ["Electric Snare", "mid", 38],
    41: ["Low Floor Tom", "low", 45],
    42: ["Closed Hi Hat", "high", 42],
    43: ["High Floor Tom", "mid", 45],
    44: ["Pedal Hi-Hat", "high", 46],
    45: ["Low Tom", "low", 45],
    46: ["Open Hi-Hat", "high", 46],
    47: ["Low-Mid Tom", "low", 47],
    48: ["Hi-Mid Tom", "mid", 47],
    49: ["Crash Cymbal 1", "high", 49],
    50: ["High Tom", "mid", 50],
    51: ["Ride Cymbal 1", "high", 51],
    52: ["Chinese Cymbal", "high", 52],
    53: ["Ride Bell", "high", 53],
    54: ["Tambourine", "high", 54],
    55: ["Splash Cymbal", "high", 55],
    56: ["Cowbell", "high", 56],
    57: ["Crash Cymbal 2", "high", 57],
    58: ["Vibraslap", "mid", 58],
    59: ["Ride Cymbal 2", "high", 59],
    60: ["Hi Bongo", "high", 60],
    61: ["Low Bongo", "mid", 61],
    62: ["Mute Hi Conga", "mid", 62],
    63: ["Open Hi Conga", "high", 63],
    64: ["Low Conga", "low", 64],
    65: ["High Timbale", "mid", 65],
    66: ["Low Timbale", "low", 66],
    67: ["High Agogo", "high", 67],
    68: ["Low Agogo", "mid", 68],
    69: ["Cabasa", "high", 69],
    70: ["Maracas", "high", 69],
    71: ["Short Whistle", "high", 71],
    72: ["Long Whistle", "high", 72],
    73: ["Short Guiro", "high", 73],
    74: ["Long Guiro", "high", 74],
    75: ["Claves", "high", 75],
    76: ["Hi Wood Block", "high", 76],
    77: ["Low Wood Block", "mid", 77],
    78: ["Mute Cuica", "high", 78],
    79: ["Open Cuica", "high", 79],
    80: ["Mute Triangle", "high", 80],
    81: ["Open Triangle", "high", 81],
}


# =============================================================================
# CORE CONVERSION FUNCTIONS
# =============================================================================

def midi_to_hv_list(filepath: str) -> List[List[List[Tuple[int, float]]]]:
    """
    Convert a MIDI drum file to HV-list representation.

    The HV-list (Hit-Velocity list) represents each 16-step bar as a list of
    (note_number, normalized_velocity) tuples per step.

    Args:
        filepath: Path to the MIDI file.

    Returns:
        List of 16-step patterns. Each pattern is a list of 16 steps,
        where each step contains tuples of (note, velocity).

    Example:
        >>> patterns = midi_to_hv_list("drums.mid")
        >>> patterns[0][0]  # First pattern, first step
        [(36, 0.78), (42, 0.65)]  # Kick and hi-hat on step 0
    """
    midi_file = mido.MidiFile(filepath)
    ticks_per_sixteenth = midi_file.ticks_per_beat / 4

    # Collect all note events with timing
    events = []
    accumulated_time = 0

    for track in midi_file.tracks:
        for msg in track:
            accumulated_time += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                if msg.note in GM_DRUM_MAP:
                    step = int(accumulated_time / ticks_per_sixteenth)
                    velocity = msg.velocity / 127.0  # Normalize to [0, 1]
                    mapped_note = GM_DRUM_MAP[msg.note][2]
                    events.append((step, mapped_note, velocity))

    if not events:
        return []

    # Calculate pattern length (round to next multiple of 16)
    max_step = max(e[0] for e in events)
    pattern_length = ((max_step // 16) + 1) * 16

    # Create step-based representation
    step_events = [[] for _ in range(pattern_length)]
    for step, note, velocity in events:
        if step < pattern_length:
            step_events[step].append((note, velocity))

    # Remove duplicates and sort within each step
    for i in range(len(step_events)):
        step_events[i] = sorted(list(set(step_events[i])))

    # Split into 16-step patterns, filtering sparse patterns
    patterns = []
    for bar_idx in range(pattern_length // 16):
        bar_start = bar_idx * 16
        bar_end = bar_start + 16
        bar_pattern = step_events[bar_start:bar_end]

        # Count non-empty steps (pattern density)
        density = sum(1 for step in bar_pattern if step)

        # Keep patterns with at least 5 active steps
        if density >= 5:
            patterns.append(bar_pattern)

    return patterns


def hv_list_to_fwod(hv_list: List[List[Tuple[int, float]]]) -> np.ndarray:
    """
    Convert HV-list to FWOD (Flattened Weighted Onset Distribution).

    Applies weighted flattening based on instrument categories:
        - Low (kick, toms): weight = 3
        - Mid (snare, claps): weight = 2
        - High (hi-hat, cymbals): weight = 1

    Args:
        hv_list: A 16-step pattern with (note, velocity) tuples per step.

    Returns:
        Normalized 16-element numpy array representing the FWOD.

    Example:
        >>> fwod = hv_list_to_fwod(pattern)
        >>> fwod.shape
        (16,)
        >>> fwod.max()
        1.0
    """
    fwod = np.zeros(16)

    for step_idx, step in enumerate(hv_list):
        weighted_sum = 0.0
        for note, velocity in step:
            if note in INSTRUMENTS_LOW:
                weighted_sum += velocity * WEIGHT_LOW
            elif note in INSTRUMENTS_MID:
                weighted_sum += velocity * WEIGHT_MID
            else:
                weighted_sum += velocity * WEIGHT_HIGH
        fwod[step_idx] = weighted_sum

    # Normalize to [0, 1]
    max_val = fwod.max()
    if max_val > 0:
        fwod = fwod / max_val

    return fwod


def midi_to_fwod(filepath: str) -> List[np.ndarray]:
    """
    Convert a MIDI file directly to FWOD representations.

    This is a convenience function that combines midi_to_hv_list and hv_list_to_fwod.

    Args:
        filepath: Path to the MIDI file.

    Returns:
        List of FWOD arrays, one per 16-step bar in the MIDI file.
    """
    hv_lists = midi_to_hv_list(filepath)
    return [hv_list_to_fwod(hv) for hv in hv_lists]


# =============================================================================
# DATASET CREATION
# =============================================================================

def create_fwod_dataset(
    midi_directory: str,
    output_path: Optional[str] = None,
    train_ratio: float = 0.7,
    test_ratio: float = 0.15,
    holdout_ratio: float = 0.15,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Create a complete FWOD dataset from a directory of MIDI files organized by genre.

    Includes stratified train/test/holdout split at the FILE level to prevent data leakage.
    All patterns from the same MIDI file will be in the same split.

    Expected directory structure:
        midi_directory/
            genre1/
                file1.mid
                file2.mid
            genre2/
                file3.mid
            ...

    Args:
        midi_directory: Root directory containing genre subdirectories.
        output_path: Optional path to save the dataset as Excel file.
        train_ratio: Proportion of files for training (default: 0.7).
        test_ratio: Proportion of files for testing (default: 0.15).
        holdout_ratio: Proportion of files for holdout (default: 0.15).
        random_state: Random seed for reproducibility (default: 42).

    Returns:
        DataFrame with columns:
            - file: Source MIDI file path
            - sequence: Bar index within the file
            - class: Genre label
            - split: 'train', 'test', or 'holdout'
            - step_0 to step_15: FWOD values
    """
    assert abs(train_ratio + test_ratio + holdout_ratio - 1.0) < 0.001, \
        "Ratios must sum to 1.0"

    # Step 1: Collect all MIDI files with their genres
    print("Step 1: Collecting MIDI files...")
    file_genre_pairs = []

    genres = [
        d for d in os.listdir(midi_directory)
        if os.path.isdir(os.path.join(midi_directory, d))
    ]

    for genre in genres:
        genre_path = os.path.join(midi_directory, genre)
        midi_files = _find_midi_files(genre_path)
        for midi_path in midi_files:
            file_genre_pairs.append((midi_path, genre))

    files = [f[0] for f in file_genre_pairs]
    labels = [f[1] for f in file_genre_pairs]

    print(f"  - Total files: {len(files)}")
    print(f"  - Genres: {len(set(labels))}")

    # Step 2: Stratified split at FILE level
    print("\nStep 2: Stratified split at FILE level...")

    # Count files per genre
    from collections import Counter
    genre_counts = Counter(labels)

    # Minimum files needed per genre for 3-way stratified split
    # With 70/15/15 split, need at least 7 files to have 1+ in each split
    MIN_FILES_FOR_SPLIT = 7

    # Separate files: genres with <MIN files go directly to train (can't stratify)
    files_small_genres = []
    labels_small_genres = []
    files_to_split = []
    labels_to_split = []

    for f, l in zip(files, labels):
        if genre_counts[l] < MIN_FILES_FOR_SPLIT:
            files_small_genres.append(f)
            labels_small_genres.append(l)
        else:
            files_to_split.append(f)
            labels_to_split.append(l)

    if files_small_genres:
        small_genres = sorted(set(labels_small_genres))
        print(f"  - Genres with <{MIN_FILES_FOR_SPLIT} files (assigned to train only):")
        for g in small_genres:
            print(f"      {g}: {genre_counts[g]} files")

    # First split: train vs (test + holdout)
    test_holdout_ratio = test_ratio + holdout_ratio
    files_train, files_temp, labels_train, labels_temp = train_test_split(
        files_to_split, labels_to_split,
        test_size=test_holdout_ratio,
        stratify=labels_to_split,
        random_state=random_state
    )

    # Second split: test vs holdout (from the temp set)
    holdout_proportion = holdout_ratio / test_holdout_ratio
    files_test, files_holdout, labels_test, labels_holdout = train_test_split(
        files_temp, labels_temp,
        test_size=holdout_proportion,
        stratify=labels_temp,
        random_state=random_state
    )

    # Add small genre files to train
    files_train.extend(files_small_genres)
    labels_train.extend(labels_small_genres)

    # Create file -> split mapping
    file_to_split = {}
    for f in files_train:
        file_to_split[f] = 'train'
    for f in files_test:
        file_to_split[f] = 'test'
    for f in files_holdout:
        file_to_split[f] = 'holdout'

    print(f"  - Train files: {len(files_train)} ({len(files_train)/len(files)*100:.1f}%)")
    print(f"  - Test files: {len(files_test)} ({len(files_test)/len(files)*100:.1f}%)")
    print(f"  - Holdout files: {len(files_holdout)} ({len(files_holdout)/len(files)*100:.1f}%)")

    # Step 3: Process all MIDI files and create records
    print("\nStep 3: Converting MIDI to FWOD...")
    records = []

    for midi_path, genre in file_genre_pairs:
        try:
            fwod_patterns = midi_to_fwod(midi_path)
            split = file_to_split[midi_path]

            for seq_idx, fwod in enumerate(fwod_patterns):
                record = {
                    'file': midi_path,
                    'sequence': seq_idx,
                    'class': genre,
                    'split': split
                }
                for step_idx, value in enumerate(fwod):
                    record[f'step_{step_idx}'] = value
                records.append(record)

        except Exception as e:
            print(f"  Error processing {midi_path}: {e}")

    df = pd.DataFrame(records)

    # Step 4: Print split statistics
    print("\nStep 4: Split statistics (patterns)...")
    split_counts = df.groupby(['split', 'class']).size().unstack(fill_value=0)
    print(f"\n  Patterns per split:")
    for split in ['train', 'test', 'holdout']:
        count = len(df[df['split'] == split])
        pct = count / len(df) * 100
        print(f"    - {split}: {count} ({pct:.1f}%)")

    if output_path:
        df.to_excel(output_path, index=False)
        print(f"\nDataset saved: {output_path}")
        print(f"  - Total patterns: {len(df)}")
        print(f"  - Genres: {df['class'].nunique()}")
        print(f"  - Columns: {list(df.columns)}")

    return df


def create_aggregated_dataset(
    input_path: str,
    output_path: str,
    n_patterns: int,
    overlapping: bool = False
) -> pd.DataFrame:
    """
    Create an aggregated dataset by combining consecutive patterns.

    This increases the feature dimensionality from 16 to 16*n_patterns,
    potentially capturing longer temporal dependencies.

    Preserves the 'split' column from the base dataset (train/test/holdout).

    Args:
        input_path: Path to the base FWOD dataset (Excel).
        output_path: Path to save the aggregated dataset.
        n_patterns: Number of consecutive patterns to combine.
        overlapping: If True, use sliding window; if False, non-overlapping chunks.

    Returns:
        Aggregated DataFrame with 16*n_patterns features per row.
    """
    data = pd.read_excel(input_path)
    records = []

    # Check if split column exists
    has_split = 'split' in data.columns

    for file_path, group in data.groupby('file'):
        group = group.sort_values('sequence')
        n_rows = len(group)

        step = 1 if overlapping else n_patterns

        for i in range(0, n_rows - n_patterns + 1, step):
            chunk = group.iloc[i:i + n_patterns]

            # Verify all rows have the same class
            if chunk['class'].nunique() != 1:
                continue

            record = {
                'file': file_path,
                'sequence': f"{chunk.iloc[0]['sequence']}-{chunk.iloc[-1]['sequence']}",
                'class': chunk.iloc[0]['class']
            }

            # Preserve split column if it exists
            if has_split:
                record['split'] = chunk.iloc[0]['split']

            # Concatenate features
            for j, (_, row) in enumerate(chunk.iterrows()):
                for k in range(16):
                    record[f'feature_{j * 16 + k}'] = row[f'step_{k}']

            records.append(record)

    df = pd.DataFrame(records)
    df.to_excel(output_path, index=False)

    print(f"Aggregated dataset saved: {output_path}")
    print(f"  - Patterns combined: {n_patterns}")
    print(f"  - Overlapping: {overlapping}")
    print(f"  - Total samples: {len(df)}")
    print(f"  - Features per sample: {16 * n_patterns}")

    # Print split statistics if available
    if has_split and 'split' in df.columns:
        print(f"  - Split distribution:")
        for split in ['train', 'test', 'holdout']:
            count = len(df[df['split'] == split])
            pct = count / len(df) * 100
            print(f"      {split}: {count} ({pct:.1f}%)")

    return df


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _find_midi_files(directory: str) -> List[str]:
    """Recursively find all .mid files in a directory."""
    midi_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith('.mid'):
                midi_files.append(os.path.join(root, f))
    return midi_files


def get_instrument_category(note: int) -> str:
    """Get the category (low/mid/high) for a MIDI drum note."""
    if note in INSTRUMENTS_LOW:
        return 'low'
    elif note in INSTRUMENTS_MID:
        return 'mid'
    else:
        return 'high'


def get_instrument_weight(note: int) -> int:
    """Get the weight for a MIDI drum note based on its category."""
    if note in INSTRUMENTS_LOW:
        return WEIGHT_LOW
    elif note in INSTRUMENTS_MID:
        return WEIGHT_MID
    else:
        return WEIGHT_HIGH


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("FWOD Converter")
    print("=" * 60)

    # Paths relative to project root
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MIDI_DIR = os.path.join(PROJECT_ROOT, "magenta midi")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "PAPER", "DATA")

    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"MIDI directory: {MIDI_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Check if MIDI directory exists
    if not os.path.exists(MIDI_DIR):
        print(f"\nError: MIDI directory not found: {MIDI_DIR}")
        print("Please ensure the 'magenta midi' folder exists in the project root.")
        exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Create base FWOD dataset
    print("\n" + "=" * 60)
    print("STEP 1: Creating FWOD dataset from MIDI files...")
    print("=" * 60)

    output_base = os.path.join(OUTPUT_DIR, "fwod_representations.xlsx")
    df_base = create_fwod_dataset(MIDI_DIR, output_base)

    # Step 2: Create aggregated datasets (2, 4, 5 patterns)
    print("\n" + "=" * 60)
    print("STEP 2: Creating aggregated datasets...")
    print("=" * 60)

    for n_patterns in [2, 4, 5]:
        output_agg = os.path.join(OUTPUT_DIR, f"fwod_pattern_{n_patterns}_inclusive.xlsx")
        print(f"\nCreating {n_patterns}-pattern aggregated dataset...")
        create_aggregated_dataset(output_base, output_agg, n_patterns, overlapping=True)

    print("\n" + "=" * 60)
    print("DONE! All datasets created in PAPER/DATA/")
    print("=" * 60)
