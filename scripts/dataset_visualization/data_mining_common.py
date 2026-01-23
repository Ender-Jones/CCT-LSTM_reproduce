"""Common utilities for UBFC-Phys dataset processing and data mining.

This module provides shared constants, path management, and I/O functions
for EDA and BVP/PPG signal processing scripts.

Dataset: UBFC-Phys (https://search-data.ubfc.fr/FR-18008901306731-2022-05-05)
"""

from pathlib import Path

import pandas as pd


# =============================================================================
# Constants: Sampling Rates
# =============================================================================

EDA_SAMPLING_RATE_HZ = 4   # EDA signals are sampled at 4 Hz
BVP_SAMPLING_RATE_HZ = 64  # BVP signals are sampled at 64 Hz


# =============================================================================
# Constants: Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH_FILE = SCRIPT_DIR / "UBFC_path.txt"
OUTPUT_DIR = SCRIPT_DIR / "eda_results"
DATA_MINING_OUTPUT_DIR = SCRIPT_DIR / "data_mining"


# =============================================================================
# Dataset Path Management
# =============================================================================

def ensure_dataset_path_file_exists() -> None:
    """Verify that the dataset path file exists.

    Raises:
        FileNotFoundError: If UBFC_path.txt does not exist.

    Outputs:
        None.
    """
    if not DATASET_PATH_FILE.exists():
        raise FileNotFoundError(f"The file {DATASET_PATH_FILE} does not exist")


def prompt_dataset_path() -> Path:
    """Prompt user to enter the dataset path interactively.

    Returns:
        Path object pointing to the user-provided dataset directory.

    Outputs:
        None.
    """
    dataset_path = input(
        "Enter the path to the dataset (to Data folder): ").strip()
    return Path(dataset_path)


def save_dataset_path(dataset_path: Path) -> None:
    """Save the dataset path to UBFC_path.txt for future use.

    Args:
        dataset_path: Path to UBFC-Phys/Data folder.

    Outputs:
        Writes to DATASET_PATH_FILE (UBFC_path.txt).
    """
    DATASET_PATH_FILE.write_text(str(dataset_path), encoding="utf-8")


def read_dataset_path() -> Path:
    """Read the dataset path from UBFC_path.txt.

    Returns:
        Path object pointing to the saved dataset directory.

    Outputs:
        None.
    """
    return Path(DATASET_PATH_FILE.read_text(encoding="utf-8").splitlines()[0].strip())


def read_pathfile_or_ask_for_path() -> Path:
    """Read dataset path from file, or prompt user if file doesn't exist.

    Attempts to read the path from UBFC_path.txt. If the file doesn't exist,
    prompts the user to enter the path and saves it for future use.

    Returns:
        Path object pointing to the UBFC-Phys/Data folder.

    Outputs:
        May write to DATASET_PATH_FILE if user provides a new path.
    """
    try:
        ensure_dataset_path_file_exists()
        return read_dataset_path()
    except FileNotFoundError:
        dataset_path = prompt_dataset_path()
        save_dataset_path(dataset_path)
        return dataset_path


# =============================================================================
# Subject Directory Utilities
# =============================================================================

def list_subject_paths(dataset_path: Path) -> list[Path]:
    """List all subject directories in the dataset.

    Scans the dataset directory for folders starting with 's' (e.g., s1, s2, ...).

    Args:
        dataset_path: Path to UBFC-Phys/Data folder.

    Returns:
        Sorted list of Path objects for each subject directory.

    Outputs:
        None.
    """
    subject_paths: list[Path] = []
    for entry in dataset_path.iterdir():
        if entry.is_dir() and entry.name.startswith("s"):
            subject_paths.append(entry)
    return sorted(subject_paths, key=lambda p: p.name)


# =============================================================================
# EDA Data I/O
# =============================================================================

def list_eda_paths(subject_path: Path) -> list[Path]:
    """List all EDA CSV files in a subject directory.

    Finds files matching pattern 'eda_*.csv' (e.g., eda_s1_T1.csv).

    Args:
        subject_path: Path to a subject folder (e.g., .../Data/s1).

    Returns:
        Sorted list of Path objects for each EDA file.

    Outputs:
        None.
    """
    eda_paths: list[Path] = []
    for entry in subject_path.iterdir():
        if entry.is_file() and entry.name.startswith("eda_"):
            eda_paths.append(entry)
    return sorted(eda_paths, key=lambda p: p.name)


def read_eda_data(eda_path: Path) -> list[float]:
    """Read EDA signal values from a CSV file.

    Reads a single-column CSV file containing raw EDA values.

    Args:
        eda_path: Path to an EDA CSV file.

    Returns:
        List of float values representing the EDA signal.

    Outputs:
        None.
    """
    eda_data: list[float] = []
    with eda_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                eda_data.append(float(line))
    return eda_data


def has_expected_eda_files(eda_paths: list[Path], subject_path: Path) -> bool:
    """Check if subject has the expected number of EDA files.

    UBFC-Phys should have exactly 3 EDA files per subject (T1/T2/T3).
    Subjects that don't match are skipped to avoid breaking batch processing.

    Args:
        eda_paths: List of EDA file paths found for the subject.
        subject_path: Path to the subject folder (for logging).

    Returns:
        True if exactly 3 EDA files found, False otherwise.

    Outputs:
        Prints warning to stdout if file count is unexpected.
    """
    if len(eda_paths) != 3:
        print(
            f"[WARN] {subject_path}: expected 3 EDA files (T1/T2/T3), got {len(eda_paths)}. Skipping."
        )
        return False
    return True


# =============================================================================
# BVP/PPG Data I/O
# =============================================================================

def list_bvp_paths(subject_path: Path) -> list[Path]:
    """List all BVP/PPG CSV files in a subject directory.

    Finds files matching pattern 'bvp_*.csv' (e.g., bvp_s1_T1.csv).

    Args:
        subject_path: Path to a subject folder (e.g., .../Data/s1).

    Returns:
        Sorted list of Path objects for each BVP file.

    Outputs:
        None.
    """
    bvp_paths: list[Path] = []
    for entry in subject_path.iterdir():
        if entry.is_file() and entry.name.startswith("bvp_"):
            bvp_paths.append(entry)
    return sorted(bvp_paths, key=lambda p: p.name)


def read_bvp_data(bvp_path: Path) -> list[float]:
    """Read BVP/PPG signal values from a CSV file.

    Reads a single-column CSV file containing raw BVP values.

    Args:
        bvp_path: Path to a BVP CSV file.

    Returns:
        List of float values representing the BVP signal.

    Outputs:
        None.
    """
    bvp_data: list[float] = []
    with bvp_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                bvp_data.append(float(line))
    return bvp_data


def has_expected_bvp_files(bvp_paths: list[Path], subject_path: Path) -> bool:
    """Check if subject has the expected number of BVP files.

    UBFC-Phys should have exactly 3 BVP files per subject (T1/T2/T3).
    Subjects that don't match are skipped to avoid breaking batch processing.

    Args:
        bvp_paths: List of BVP file paths found for the subject.
        subject_path: Path to the subject folder (for logging).

    Returns:
        True if exactly 3 BVP files found, False otherwise.

    Outputs:
        Prints warning to stdout if file count is unexpected.
    """
    if len(bvp_paths) != 3:
        print(
            f"[WARN] {subject_path}: expected 3 BVP files (T1/T2/T3), got {len(bvp_paths)}. Skipping."
        )
        return False
    return True


# =============================================================================
# Manifest Utilities
# =============================================================================

def read_master_manifest(dataset_path: Path) -> dict[str, str]:
    """Read master_manifest.csv and return subject-to-group mapping.

    The manifest file is located at UBFC-Phys/master_manifest.csv,
    which is the parent directory of the Data folder.

    Args:
        dataset_path: Path to the Data folder (e.g., .../UBFC-Phys/Data).

    Returns:
        Dict mapping subject ID to group, e.g., {'s1': 'test', 's2': 'ctrl', ...}.

    Raises:
        FileNotFoundError: If master_manifest.csv is not found.

    Outputs:
        None.
    """
    manifest_path = dataset_path.parent / "master_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"master_manifest.csv not found at {manifest_path}")

    df = pd.read_csv(manifest_path)
    return dict(zip(df['subject'], df['group']))
