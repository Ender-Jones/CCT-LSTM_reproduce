import argparse
import tqdm
import json
from pathlib import Path
import cv2
import numpy as np
from pyts.image import MarkovTransitionField

# import all necessary modules
from landmark_extractor import LandmarkExtractor
from file_path_gen import FilePathGen
from integrity_and_masterManifest import IntegrityCheckerAndManifestCreator
from pca_and_mtf import PCAandMTFProcessor
from rppg_extractor import RppgExtractor


def run_integrity_check():
    """the integrity check and manifest creation logic."""
    print("--- Running Dataset Integrity Check ---")
    checker = IntegrityCheckerAndManifestCreator()
    checker.running_script()


def run_landmark_extraction():
    """the landmark extraction logic."""
    print("--- Starting Landmark Extraction Pipeline ---")
    try:
        fpg = FilePathGen()
        # adjust window_length and step_length etc. as needed
        extractor = LandmarkExtractor(window_length=60, step_length=5)
    except FileNotFoundError as e:
        print(f"Error initializing: {e}. Run integrity check first.")
        return

    # get the list of subjects and all video paths
    subject_list = fpg.get_subject_list()
    all_video_paths = fpg.get_all_video_paths(subject_list)

    for video_path in tqdm.tqdm(all_video_paths, desc="Extracting Landmarks"):
        # check if the output file already exists
        output_path = video_path.parent / 'landmarks' / f"{video_path.stem}_landmarks.json"
        if output_path.exists():
            print(f"Skipping {video_path.name}, landmarks already extracted.")
            continue
        try:
            landmarks_data = extractor.extract_landmarks_from_video(video_path)
            extractor.save_landmarks_to_subject_folder(landmarks_data, video_path)
        except Exception as e:
            print(f"!!! ERROR processing {video_path.name}: {e}")
            continue

    print("--- Landmark Extraction Pipeline Finished ---")


# TODO: 该function完全没有测试和查看, 我们要慢慢来
def run_pca_mtf_pipeline():
    """The PCA and MTF image generation logic."""
    print("--- Starting PCA and MTF Image Generation ---")
    try:
        fpg = FilePathGen()
        processor = PCAandMTFProcessor()
    except FileNotFoundError as e:
        print(f"Error initializing: {e}. Run integrity check first.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during initialization: {e}")
        return

    subject_list = fpg.get_subject_list()
    levels = ['T1', 'T2', 'T3']

    for subject_id in tqdm.tqdm(subject_list, desc="Processing Subjects"):
        for level in levels:
            try:
                landmark_path = fpg.get_landmark_path(subject_id, level)
                if not landmark_path.exists():
                    print(f"Skipping {subject_id}/{level}: Landmark file not found.")
                    continue

                with open(landmark_path, 'r') as f:
                    all_windows_data = json.load(f)

                # Create output directory for the images
                output_dir = landmark_path.parent.parent / 'mtf_images'
                output_dir.mkdir(exist_ok=True)

                for window_data in all_windows_data:
                    window_id = window_data['window_id']
                    
                    # Define output path and check for existence
                    image_name = f"{subject_id}_{level}_window_{window_id}.png"
                    output_image_path = output_dir / image_name
                    if output_image_path.exists():
                        print(f"Skipping {subject_id}/{level}/window_{window_id}: Image already exists.")
                        continue

                    landmarks = window_data['landmarks']
                    if not landmarks:
                        print(f"Skipping window {window_id} for {subject_id}/{level}: No landmarks found.")
                        continue
                        
                    # Transform data and save the image
                    rgb_image = processor.transform(landmarks)
                    cv2.imwrite(str(output_image_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

            except Exception as e:
                print(f"!!! ERROR processing {subject_id}/{level}: {e}")
                continue
    
    print("--- PCA and MTF Image Generation Finished ---")


def run_rppg_pipeline(window_length_sec=60, step_length_sec=5, image_size=224, mtf_bins=8):
    """The rPPG signal extraction, windowing, and MTF image generation logic."""
    print("--- Starting rPPG Signal to MTF Image Generation ---")
    try:
        fpg = FilePathGen()
        rppg_ext = RppgExtractor()
        mtf = MarkovTransitionField(n_bins=mtf_bins)
    except FileNotFoundError as e:
        print(f"Error initializing: {e}. Run integrity check first.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during initialization: {e}")
        return

    subject_list = fpg.get_subject_list()
    all_video_paths = fpg.get_all_video_paths(subject_list)

    for video_path in tqdm.tqdm(all_video_paths, desc="Processing Videos for rPPG"):
        try:
            subject_id = video_path.parent.name
            level_id = video_path.stem.split('_')[-1]

            # Define and create the output directory for this subject
            output_dir = video_path.parent / 'rppg_mtf_images'
            output_dir.mkdir(exist_ok=True)

            # --- Step 1: Extract the full rPPG signal from the video ---
            signal, fps = rppg_ext.extract_signal(video_path)
            if signal is None or fps is None:
                print(f"Skipping {video_path.name}: Failed to extract rPPG signal.")
                continue

            # --- Step 2: Windowing the signal ---
            window_frames = int(window_length_sec * fps)
            step_frames = int(step_length_sec * fps)
            total_frames = len(signal)

            for window_id, start_frame in enumerate(range(0, total_frames, step_frames)):
                end_frame = start_frame + window_frames
                if end_frame > total_frames:
                    # Skip the last window if it's shorter than the required length
                    continue
                
                # Check if image already exists
                image_name = f"{subject_id}_{level_id}_rppg_window_{window_id}.png"
                output_image_path = output_dir / image_name
                if output_image_path.exists():
                    continue # Silently skip if already processed

                # --- Step 3: Apply MTF to the window ---
                signal_window = signal[start_frame:end_frame]
                
                # MTF expects (n_samples, n_timestamps), so we reshape
                mtf_image = mtf.fit_transform(signal_window.reshape(1, -1))[0]

                # --- Step 4: Resize, normalize, and save the image ---
                resized_image = cv2.resize(mtf_image, (image_size, image_size))
                
                # Normalize to 0-255 and convert to uint8 grayscale image
                normalized_image = cv2.normalize(resized_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                
                cv2.imwrite(str(output_image_path), normalized_image)

        except Exception as e:
            print(f"!!! ERROR processing {video_path.name} for rPPG: {e}")
            continue
    
    print("--- rPPG MTF Image Generation Finished ---")


def main():
    parser = argparse.ArgumentParser(description="Data processing pipeline for UBFC-Phys dataset.")

    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    parser_check = subparsers.add_parser('check', help='Run dataset integrity check and create manifest.')
    parser_check.set_defaults(func=run_integrity_check)

    parser_extract = subparsers.add_parser('extract', help='Extract face landmarks from all videos.')
    parser_extract.set_defaults(func=run_landmark_extraction)

    parser_process = subparsers.add_parser('process', help='Process landmarks into MTF images.')
    parser_process.set_defaults(func=run_pca_mtf_pipeline)

    parser_rppg = subparsers.add_parser('rppg', help='Extract rPPG signals and process into MTF images.')
    parser_rppg.set_defaults(func=run_rppg_pipeline)

    args = parser.parse_args()
    args.func()


if __name__ == '__main__':
    main()
