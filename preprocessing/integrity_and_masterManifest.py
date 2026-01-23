import os
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog


class IntegrityCheckerAndManifestCreator:
    @staticmethod
    def _check_dataset_integrity_and_create_manifest(dataset_base_path: Path):
        """
        Args:
            dataset_base_path (Path): The dict expected to contain the "UBFC-Phys" folder.
        """
        ubfc_phys_path = dataset_base_path / "UBFC-Phys"
        print(f"Checking path: {ubfc_phys_path}")

        # 1. existence of "UBFC-Phys"
        if not ubfc_phys_path.is_dir():
            print(f"Error: No 'UBFC-Phys' folder under {dataset_base_path}'")
            return

        data_path = ubfc_phys_path / "Data"
        # 2. existence of "Data" folder under "UBFC-Phys"
        if not data_path.is_dir():
            print(f"Error: No 'Data' folder under '{ubfc_phys_path}'")
            return

        # expected subfolders as set
        expected_folders = {f"s{i}" for i in range(1, 57)}
        # existing subfolders in the "Data" folder
        existing_folders = {f.name for f in data_path.iterdir()}
        # check if all expected folders exist, if not, report missing ones
        missing_folders = expected_folders - existing_folders  # pythonic, yes!
        if missing_folders:
            print(f"Error: Missing subject folders: {', '.join(missing_folders)}")
            # stop execution if any folder is missing
            sys.exit(1)

        print("Found 56/56 subject folders, Checking integrity of each folder...")

        manifest_data = []
        # integrate every subfolder, check files, and create manifest.
        for i in tqdm(range(1, 57), desc="Processing subjects"):
            subject_id = f"s{i}"
            subject_path = data_path / subject_id

            if not subject_path.is_dir():
                print(f"Error, subject folder not found '{subject_id}'。")
                return

            # check every folder for files
            expected_files = []
            # video files check
            for task in ['T1', 'T2', 'T3']:
                expected_files.append(f"vid_{subject_id}_{task}.avi")
            # bvp files check.
            for task in ['T1', 'T2', 'T3']:
                expected_files.append(f"bvp_{subject_id}_{task}.csv")
            # eda files check.
            for task in ['T1', 'T2', 'T3']:
                expected_files.append(f"eda_{subject_id}_{task}.csv")
            # meta data check
            expected_files.append(f"info_{subject_id}.txt")
            # self-report check
            expected_files.append(f"selfReportedAnx_{subject_id}.csv")

            all_files_present = True
            for file_name in expected_files:
                if not (subject_path / file_name).is_file():
                    print(f"Error: In folder '{subject_id}' file '{file_name}' not found.")
                    all_files_present = False

            if not all_files_present:
                sys.exit(1)

            # making master manifest
            info_file_path = subject_path / f"info_{subject_id}.txt"
            with open(info_file_path, 'r') as f:
                # File format: subject_id, gender, group, date, time(? Not sure about this)split by \n
                # e.g. s1, m, test, 2019_02_07, 11_37_06
                lines = f.read().splitlines()
                if len(lines) < 5:
                    print(f"Error: File '{info_file_path}' incomplete or malformed.")
                    sys.exit(1)
                elif lines[2] != 'test' and lines[2] != 'ctrl':
                    print(f"Error: File '{info_file_path}' has unexpected group value: {lines[2]}.")
                    sys.exit(1)
                elif lines[0] != subject_id:
                    print(
                        f"Error: File '{info_file_path}' subject ID mismatch: expected '{subject_id}', found '{lines[0]}'.")
                    sys.exit(1)
                else:
                    group = lines[2]

            manifest_data.append({'subject': subject_id, 'group': group})

        manifest_df = pd.DataFrame(manifest_data)
        output_path = ubfc_phys_path / "master_manifest.csv"
        manifest_df.to_csv(output_path, index=False)

        # TODO(multilevel-labels): For multi-level stress classification (T1, T3-ctrl, T3-test),
        # downstream datasets should read this master_manifest.csv (columns: subject, group).

        # make a file to store UBFC_data path in the REPO ROOT.
        repo_root = Path(__file__).resolve().parent.parent
        config_output_path = repo_root / "UBFC_data_path.txt"
        
        with open(config_output_path, "w") as f:
            f.write(str(data_path))

        print("\n======================================================")
        print("Integrity Check Finished, Master Manifest created！")
        print(f"Master manifest path: {output_path}")
        print(f"Config file created at: {config_output_path}")
        print("Preview:")
        print(manifest_df.head())
        print("======================================================")

    def running_script(self):
        """Main function to run the integrity check and manifest creation."""
        root = tk.Tk()
        root.withdraw()  # hide the root window

        print("Please select the directory that contains the 'UBFC-Phys' folder.")
        selected_path = filedialog.askdirectory(title="Root Directory Selection")

        if selected_path:
            dataset_root_path = Path(selected_path)
            self._check_dataset_integrity_and_create_manifest(dataset_root_path)
        else:
            print("No directory selected. Exiting.")
            sys.exit(1)
