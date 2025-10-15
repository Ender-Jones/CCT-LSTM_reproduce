import argparse
import json

import cv2
import tqdm

from file_path_gen import FilePathGen
from integrity_and_masterManifest import IntegrityCheckerAndManifestCreator
# import all necessary modules
from landmark_extractor import LandmarkExtractor
from pca_and_mtf import PCAandMTFProcessor


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


def run_faceLandmark_pca_mtf_pipeline():
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
                output_dir = landmark_path.parent.parent / 'mtf_images' / 'landmark'
                output_dir.mkdir(parents=True, exist_ok=True)

                for window_data in all_windows_data:
                    window_id = window_data['window_id']

                    # Define output path and check for existence
                    # Use zero-padding for window_id to ensure correct alphabetical sorting
                    image_name = f"{subject_id}_{level}_window_{window_id:03d}.png"
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


def main():
    parser = argparse.ArgumentParser(description="Data processing pipeline for UBFC-Phys dataset.")

    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    parser_check = subparsers.add_parser('check', help='Run dataset integrity check and create manifest.')
    parser_check.set_defaults(func=run_integrity_check)

    parser_extract = subparsers.add_parser('extract', help='Extract face landmarks from all videos.')
    parser_extract.set_defaults(func=run_landmark_extraction)

    parser_process = subparsers.add_parser('process', help='Process Face landmarks into MTF images.')
    parser_process.set_defaults(func=run_faceLandmark_pca_mtf_pipeline)

    args = parser.parse_args()
    args.func()


if __name__ == '__main__':
    main()
