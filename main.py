import argparse
import tqdm

# import all necessary modules
from landmark_extractor import LandmarkExtractor
from file_path_gen import FilePathGen
from integrity_and_masterManifest import IntegrityCheckerAndManifestCreator


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


def main():
    parser = argparse.ArgumentParser(description="Data processing pipeline for UBFC-Phys dataset.")

    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    parser_check = subparsers.add_parser('check', help='Run dataset integrity check and create manifest.')
    parser_check.set_defaults(func=run_integrity_check)

    parser_extract = subparsers.add_parser('extract', help='Extract face landmarks from all videos.')
    parser_extract.set_defaults(func=run_landmark_extraction)

    args = parser.parse_args()
    args.func()


if __name__ == '__main__':
    main()
