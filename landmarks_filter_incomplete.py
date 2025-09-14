import json
from pathlib import Path
import argparse
import cv2
import tqdm
import shutil

from file_path_gen import FilePathGen


def get_video_fps(video_path: Path) -> float:
    """Safely get frames per second (FPS) of a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 0:
        raise ValueError(f"Invalid FPS ({fps}) for video: {video_path}")
    return fps


def get_expected_window_frames(fps: float, window_sec: int) -> int:
    """Calculate the expected number of frames in a full window, using floor."""
    return int(window_sec * fps)


def is_window_complete(window_data: dict, expected_frames: int) -> bool:
    """
    Check if a window is complete. A complete window must:
    1. Have the exact number of frames specified by `expected_frames`.
    2. Have frame indices that are contiguous and match start/end frames.
    """
    landmarks = window_data.get('landmarks', {})
    if not isinstance(landmarks, dict) or len(landmarks) != expected_frames:
        return False

    try:
        # Frame indices in JSON are strings, convert to int for sorting
        frame_indices = sorted([int(k) for k in landmarks.keys()])
    except (ValueError, TypeError):
        return False

    start_frame = int(window_data['start_frame'])
    end_frame = int(window_data['end_frame'])

    # Check for contiguity and boundary match
    if frame_indices[0] != start_frame or frame_indices[-1] != end_frame:
        return False
    if (end_frame - start_frame + 1) != expected_frames:
        return False

    return True


def process_landmark_file(landmark_json_path: Path, video_path: Path, window_sec: int, step_sec: int, apply: bool):
    """
    Processes a single landmark JSON file to filter out incomplete windows.
    - Moves original to a backup directory.
    - Writes cleaned data back to the original path.
    """
    try:
        fps = get_video_fps(video_path)
        expected_frames = get_expected_window_frames(fps, window_sec)
    except (IOError, ValueError) as e:
        print(f"[ERROR] Could not process video metadata for {video_path.name}: {e}")
        return

    # Define paths
    subject_dir = landmark_json_path.parent.parent
    backup_dir = subject_dir / 'landmarks_backup'
    backup_path = backup_dir / landmark_json_path.name

    # Handle file operations if in apply mode
    if apply:
        backup_dir.mkdir(exist_ok=True)
        # Move original file to backup location
        shutil.move(str(landmark_json_path), str(backup_path))
        source_path = backup_path
    else:
        source_path = landmark_json_path

    # Read the (now backed up) landmark data
    try:
        with open(source_path, 'r') as f:
            all_windows_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"[ERROR] Could not read or decode JSON from: {source_path}")
        return

    if not isinstance(all_windows_data, list):
        print(f"[ERROR] Expected a list of windows in {source_path}, but found {type(all_windows_data)}.")
        return

    # Filter for complete windows
    complete_windows = []
    # Sort windows by start time to ensure correct order before filtering
    sorted_windows = sorted(all_windows_data, key=lambda w: int(w['start_frame']))

    for window in sorted_windows:
        if is_window_complete(window, expected_frames):
            complete_windows.append(window)

    # Re-index window_id and correct end_frame for all complete windows
    for i, window in enumerate(complete_windows):
        window['window_id'] = i
        # Ensure end_frame is consistent with a full window length
        window['end_frame'] = int(window['start_frame']) + expected_frames - 1

    # Report changes
    before_count = len(all_windows_data)
    after_count = len(complete_windows)
    if before_count != after_count:
        print(f"[CHANGE] {landmark_json_path.name}: Filtered {before_count} -> {after_count} windows.")
    else:
        print(f"[OK] {landmark_json_path.name}: No changes needed ({before_count} windows).")

    # Write the cleaned data back to the original path if in apply mode
    if apply:
        with open(landmark_json_path, 'w') as f:
            json.dump(complete_windows, f, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description="Filter incomplete windows from existing landmark JSON files, "
                    "move originals to a backup folder, and save cleaned versions."
    )
    parser.add_argument("--window-sec", type=int, default=60, help="Window duration in seconds (must match rPPG).")
    parser.add_argument("--step-sec", type=int, default=5, help="Step/stride in seconds (must match rPPG).")
    parser.add_argument(
        "--apply", action="store_true",
        help="If set, performs file operations (backup and overwrite). Otherwise, runs in dry-run mode."
    )
    args = parser.parse_args()

    fpg = FilePathGen()
    subjects = fpg.get_subject_list()
    levels = ['T1', 'T2', 'T3']
    
    file_iterator = []
    for subject_id in subjects:
        for level in levels:
            file_iterator.append((subject_id, level))

    print(f"Scanning for landmark files... Window={args.window_sec}s, Step={args.step_sec}s.")
    if not args.apply:
        print("--- RUNNING IN DRY-RUN MODE --- (No files will be moved or changed)")
    
    for subject_id, level in tqdm.tqdm(file_iterator, desc="Processing Landmark Files"):
        landmark_path = fpg.get_landmark_path(subject_id, level)
        video_path = fpg.vid_path_gen(subject_id, level)

        if not landmark_path.exists():
            continue
        if not video_path.exists():
            print(f"[WARN] Landmark file exists but video is missing. Skipping {landmark_path.name}")
            continue

        process_landmark_file(landmark_path, video_path, args.window_sec, args.step_sec, args.apply)
    
    print("\nProcessing complete.")
    if not args.apply:
        print("To apply these changes, re-run with the --apply flag.")


if __name__ == "__main__":
    main()
