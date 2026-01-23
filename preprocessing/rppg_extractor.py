import json
from pathlib import Path
import cv2
import numpy as np
import tqdm

from pyVHR.extraction.sig_processing import SignalProcessing
from pyVHR.extraction.skin_extraction_methods import SkinExtractionFaceParsing
from pyVHR.BVP.methods import cpu_OMIT

"""
NOTE ABOUT THIS FILE (pyVHR-based helper, not runnable in this repo as-is)

Provenance
----------
This script is an auxiliary extractor adapted from the pyVHR project to replicate the
rPPG (BVP) signal generation used in the paper. It documents the approach and JSON
schema we used when communicating with the original authors. It is intentionally kept
inside this repository for reference and provenance.

Limitations (Why it may not run here)
-------------------------------------
- This module depends on pyVHR and its ecosystem (torch, face parsing, etc.).
  Those dependencies are NOT included or managed by this project’s environment.yml.
- Importing this module requires a working pyVHR installation and compatible
  models/drivers. Without that, simply importing it will fail.
- The code paths (e.g., dataset locations, device flags) are tailored for a separate
  environment and were used to pre-generate rPPG JSON files. In this repo, we only
  consume those JSON files to create MTF images.

Intended Use
------------
- Do NOT import or execute this file inside the current project environment unless
  you have installed pyVHR and validated its runtime. Treat this as documentation
  of the rPPG-extraction process rather than an integral component of the pipeline.
- The actual CCT-LSTM training here only needs the exported JSON files placed under
  `.../sXX/rppg_signals/vid_{subject}_{level}_rppg.json`.

Expected JSON Schema (per-window)
---------------------------------
[
  {
    "video_name": "vid_sXX_TY.avi",
    "window_id": <int>,
    "start_frame": <int>,
    "end_frame": <int>,
    "fps": <float>,
    "bvp_signal": [<float>, ...]  # 1D array length ~= (end_frame - start_frame + 1)
  },
  ...
]

How to reuse (if you really want to run it)
-------------------------------------------
1) Install pyVHR and its dependencies in a separate environment that you control.
2) Adjust dataset paths in `process_dataset()`.
3) Run this script to export rPPG JSONs. Then copy those JSONs to the structure
   expected by this repo (see FilePathGen/rppg path conventions).

Warning
-------
This script is provided for transparency and traceability only. The maintained and
supported pipeline in this repository starts from the already-exported rPPG JSONs
and converts them into MTF images.
"""


def get_dataset_path() -> Path:
    """Read dataset path from UBFC_data_path.txt config file.
    
    Returns:
        Path: The dataset root path (e.g., .../UBFC-Phys/Data).
        
    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    repo_root = Path(__file__).resolve().parent.parent
    config_path = repo_root / "UBFC_data_path.txt"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Please run 'python main.py check' first to set up the dataset path."
        )
    return Path(config_path.read_text().strip())


class RppgExtractor:
    def __init__(self, window_length_sec=60, step_length_sec=5, device='CPU'):
        """
        Initialize the rPPG extractor.

        Args:
            window_length_sec (int): Length of the sliding window in seconds.
            step_length_sec (int): Step size for window sliding in seconds.
            device (str): Compute device ('CPU' or 'GPU').
        """
        self.window_length_sec = window_length_sec
        self.step_length_sec = step_length_sec

        # --- Normalize device string to avoid case-sensitivity issues ---
        device_norm = str(device).upper()
        if device_norm not in ('CPU', 'GPU'):
            print(f"[WARN] Unknown device='{device}', falling back to 'CPU'")
            device_norm = 'CPU'

        # Device identifier for torch.load (lowercase)
        torch_device = 'cuda' if device_norm == 'GPU' else 'cpu'

        # --- Initialize pyVHR core components ---
        print("Initializing pyVHR components...")
        # Use lowercase during construction to avoid torch.device('CPU') error
        skin_extractor = SkinExtractionFaceParsing(device=torch_device)
        # Immediately override to uppercase to match library's internal device checks
        skin_extractor.device = device_norm
        self.signal_processor = SignalProcessing()
        self.signal_processor.set_skin_extractor(skin_extractor)
        print("pyVHR components initialized.")

    @staticmethod
    def _get_video_metadata(video_path):
        """Helper function: Get the FPS and total frame count of a video."""
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise IOError(f"Unable to open video file: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if fps == 0 or total_frames == 0:
            raise ValueError(f"Video file may be corrupted or has invalid metadata: {video_path}")
        return fps, total_frames

    def extract_and_save_signals_from_video(self, video_path, output_dir):
        """
        Process a single video, extract rPPG signals for all windows and save to file.

        Args:
            video_path (Path): Path to the input video file.
            output_dir (Path): Directory to save the output JSON file.
        """
        try:
            fps, total_frames = self._get_video_metadata(video_path)
        except (IOError, ValueError) as e:
            print(f"Error: {e}")
            return

        # Define output file path and check if it already exists
        output_dir.mkdir(parents=True, exist_ok=True)
        json_file_path = output_dir / f"{video_path.stem}_rppg.json"
        if json_file_path.exists():
            print(f"File already exists, skipping: {json_file_path}")
            return

        # --- Step 1: Extract the full RGB signal from the entire video at once ---
        # This is a time-consuming operation, but only performed once per video
        print(f"Extracting full RGB signal from video '{video_path.name}'...")
        try:
            # full_rgb_signal shape is (num_frames, 1, 3)
            full_rgb_signal = self.signal_processor.extract_holistic(str(video_path))
            # Remove the redundant middle dimension -> (num_frames, 3)
            # In extract_holistic mode, the selector count is always 1, so the
            # returned shape [num_frames, 1, rgb_channels] has a redundant dim at axis=1
            full_rgb_signal = np.squeeze(full_rgb_signal, axis=1)
        except Exception as e:
            print(f"!! Failed to extract RGB signal from video '{video_path.name}': {e}")
            return
        
        print(f"RGB signal extraction complete, shape: {full_rgb_signal.shape}")

        # --- Step 2: Apply sliding window and convert signals ---
        window_frames = int(self.window_length_sec * fps)
        # Note: step is calculated based on the non-overlapping portion
        step_frames = int(self.step_length_sec * fps)

        all_windows_data = []
        
        print("Applying sliding window and converting to BVP signal...")
        for window_id, start_frame in enumerate(range(0, total_frames, step_frames)):
            end_frame = start_frame + window_frames
            # If window exceeds video length, skip this window
            if end_frame > total_frames:
                continue

            # 2.1 Slice the window
            rgb_window = full_rgb_signal[start_frame:end_frame, :]

            # 2.2 Reshape signal to match cpu_OMIT input requirements
            # cpu_OMIT expects [num_estimators, rgb_channels, num_frames]
            # Current shape: [num_frames, rgb_channels] -> (window_frames, 3)
            # Transpose -> (3, window_frames)
            # Add dimension -> (1, 3, window_frames)
            rgb_window_transposed = rgb_window.T
            rgb_window_final = np.expand_dims(rgb_window_transposed, axis=0)

            # 2.3 Call OMIT for conversion
            # bvp_window shape is (1, window_frames)
            bvp_window = cpu_OMIT(rgb_window_final)
            # Flatten to 1D array
            bvp_signal_list = bvp_window.flatten().tolist()

            # 2.4 Collect data
            window_data = {
                'video_name': video_path.name,
                'window_id': window_id,
                'start_frame': start_frame,
                'end_frame': end_frame - 1,
                'fps': fps,
                'bvp_signal': bvp_signal_list
            }
            all_windows_data.append(window_data)
        
        # --- Step 3: Save to JSON file ---
        with open(json_file_path, 'w') as f:
            json.dump(all_windows_data, f, indent=4)
            
        print(f"rPPG signal successfully saved to: {json_file_path}")


def process_dataset():
    """
    Process the UBFC-Phys dataset to extract rPPG signals from all videos.
    
    This function reads the dataset path from UBFC_data_path.txt (created by
    'python main.py check'), iterates over all subject videos, and extracts
    rPPG signals using the pyVHR OMIT algorithm.
    """
    # --- 1. Configuration ---
    # Read dataset path from config file (same mechanism as FilePathGen)
    try:
        DATASET_ROOT_PATH = get_dataset_path()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Fixed parameters for paper reproduction
    WINDOW_LENGTH_SEC = 60
    STEP_LENGTH_SEC = 5
    DEVICE = "CPU"  

    # --- 2. Build video file list based on dataset structure ---
    if not DATASET_ROOT_PATH.exists():
        print(f"Error: Dataset root path does not exist: {DATASET_ROOT_PATH}")
        return

    print("Searching for video files based on UBFC-Phys structure...")
    all_video_paths = []
    levels = ['T1', 'T2', 'T3']
    
    # Get all subject directories (s1, s2, ...) and sort them naturally
    subject_dirs = sorted(
        [d for d in DATASET_ROOT_PATH.iterdir() if d.is_dir() and d.name.startswith('s')],
        key=lambda p: int(p.name[1:])
    )

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        for level in levels:
            video_path = subject_dir / f"vid_{subject_id}_{level}.avi"
            if video_path.is_file():
                all_video_paths.append(video_path)
            else:
                print(f"Warning: Video file does not exist, skipping: {video_path}")
    
    if not all_video_paths:
        print(f"No video files found in '{DATASET_ROOT_PATH}' matching the expected pattern.")
        return

    # --- 3. Execute extraction ---
    extractor = RppgExtractor(
        window_length_sec=WINDOW_LENGTH_SEC,
        step_length_sec=STEP_LENGTH_SEC,
        device=DEVICE
    )

    print(f"Found {len(all_video_paths)} video files. Starting processing...")
    for video_path in tqdm.tqdm(all_video_paths, desc="Overall Progress"):
        # Output directory will be inside each subject folder, named 'rppg_signals'
        # e.g., /.../s1/rppg_signals/
        subject_dir = video_path.parent
        output_dir = subject_dir / "rppg_signals"
        
        print(f"\nProcessing: {video_path}")
        extractor.extract_and_save_signals_from_video(video_path, output_dir)

    print("\n--- All videos processed ---")


if __name__ == '__main__':
    process_dataset()