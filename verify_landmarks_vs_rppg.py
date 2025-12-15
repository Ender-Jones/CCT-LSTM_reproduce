import json
from pathlib import Path
import argparse
import tqdm

from preprocessing.file_path_gen import FilePathGen

"""
Purpose
-------
Verify temporal alignment between per-window Landmark JSON and rPPG JSON for each
subject/level. It checks two things: (1) window counts are equal; (2) each window's
start_frame and end_frame match one-to-one after sorting by start_frame.

When to run
-----------
- After you have generated both Landmark JSONs (via the landmark extractor) and rPPG JSONs
  (e.g., pre-exported from pyVHR). Run this script BEFORE converting to MTF at scale to
  ensure the two modalities are aligned and can form valid pairs.

Expected locations
------------------
- Landmark JSON:   <data_root>/sXX/landmarks/vid_{sXX}_{T*}_landmarks.json
- rPPG JSON:       <data_root>/sXX/rppg_signals/vid_{sXX}_{T*}_rppg.json

How to run
----------
1) Ensure `UBFC_data_path.txt` points to `<data_root>` (created by `main.py check`).
2) Execute:
   python verify_landmarks_vs_rppg.py

Outcome
-------
- For each subject/level pair, prints OK/MISMATCH and the reason.
- Prints a summary: total compared and #mismatches.

Notes
-----
- This is a read-only verification; it will not modify data.
- If there are mismatches, inspect the listed pairs and consider re-generating JSONs.
"""


def load_json_safely(path: Path):
    """Loads a JSON file, returning None if it doesn't exist or is invalid."""
    if not path.exists():
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"[WARN] Invalid JSON format in {path.name}. Skipping.")
        return None


def verify_alignment(landmark_data, rppg_data):
    """
    Compares two lists of window data for alignment.
    Checks for equal length and matching start/end frames for each window.
    """
    if len(landmark_data) != len(rppg_data):
        return False, f"window count mismatch ({len(landmark_data)} vs {len(rppg_data)})"

    # Sort both by start_frame to ensure order before comparison
    lmk_sorted = sorted(landmark_data, key=lambda w: int(w['start_frame']))
    rppg_sorted = sorted(rppg_data, key=lambda w: int(w['start_frame']))

    mismatched_indices = []
    for i, (lmk_win, rppg_win) in enumerate(zip(lmk_sorted, rppg_sorted)):
        lmk_start = int(lmk_win['start_frame'])
        rppg_start = int(rppg_win['start_frame'])
        lmk_end = int(lmk_win['end_frame'])
        rppg_end = int(rppg_win['end_frame'])

        if lmk_start != rppg_start or lmk_end != rppg_end:
            mismatched_indices.append(i)

    if mismatched_indices:
        return False, f"frame boundary mismatch at window indices: {mismatched_indices}"

    return True, f"aligned with {len(landmark_data)} windows"


def get_rppg_path(fpg: FilePathGen, subject_id: str, level: str) -> Path:
    """Constructs the path to the corresponding rPPG JSON file."""
    # Based on `rppg_extractor.py`, rppg_signals is a sibling to landmarks
    # e.g., .../s1/landmarks/... -> .../s1/rppg_signals/...
    subject_dir = fpg.datapath / subject_id
    video_stem = f"vid_{subject_id}_{level}"
    return subject_dir / 'rppg_signals' / f"{video_stem}_rppg.json"


def main():
    parser = argparse.ArgumentParser(
        description="Verify alignment between processed landmark files and rPPG signal files."
    )
    args = parser.parse_args()

    fpg = FilePathGen()
    subjects = fpg.get_subject_list()
    levels = ['T1', 'T2', 'T3']
    
    total_pairs = 0
    mismatch_count = 0

    print("--- Starting Alignment Verification ---")
    
    file_iterator = []
    for subject_id in subjects:
        for level in levels:
            file_iterator.append((subject_id, level))

    for subject_id, level in tqdm.tqdm(file_iterator, desc="Verifying Alignment"):
        landmark_path = fpg.get_landmark_path(subject_id, level)
        
        # Skip if the primary landmark file doesn't exist
        if not landmark_path.exists():
            continue

        rppg_path = get_rppg_path(fpg, subject_id, level)
        
        landmark_data = load_json_safely(landmark_path)
        rppg_data = load_json_safely(rppg_path)

        # Handle cases where one or both files are missing/invalid
        if landmark_data is None or rppg_data is None:
            print(f"[{subject_id}/{level}] SKIPPED: One or both JSON files are missing or invalid.")
            continue
            
        total_pairs += 1
        is_aligned, message = verify_alignment(landmark_data, rppg_data)

        if is_aligned:
            print(f"[{subject_id}/{level}] OK: {message}")
        else:
            print(f"[{subject_id}/{level}] MISMATCH: {message}")
            mismatch_count += 1
            
    print("\n--- Verification Summary ---")
    print(f"Total file pairs compared: {total_pairs}")
    print(f"Mismatched pairs found: {mismatch_count}")
    if mismatch_count == 0 and total_pairs > 0:
        print("All files are perfectly aligned.")
    else:
        print("Please review the mismatched files listed above.")


if __name__ == "__main__":
    main()
