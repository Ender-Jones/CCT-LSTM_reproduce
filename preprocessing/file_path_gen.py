from pathlib import Path


class FilePathGen:
    # Dynamically generate file paths for UBFC-Phys dataset.
    def __init__(self, config_path=None):
        """
        Args:
            config_path: Path to the config file. If None, looks in the repo root.
        """
        repo_root = Path(__file__).resolve().parent.parent
        
        if config_path is None:
            self.datapath_config = repo_root / "UBFC_data_path.txt"
        else:
            self.datapath_config = Path(config_path)

        if not self.datapath_config.is_file():
            raise FileNotFoundError(
                f"Configuration file {self.datapath_config} not found.\n"
                f"Please run 'preprocessing/integrity_and_masterManifest.py' first."
            )

        # Read the data path from text file.
        with open(self.datapath_config, 'r') as f:
            self.datapath = Path(f.read().strip())

    def get_subject_list(self):
        """Get a list of all subject IDs in the dataset by iterating over the datapath folder names"""
        subject_list = [d.name for d in self.datapath.iterdir() if d.is_dir()]
        return subject_list

    def vid_path_gen(self, test_id, level):
        output_path = self.datapath / test_id / f'vid_{test_id}_{level}.avi'
        return output_path


    def eda_path_gen(self, test_id, level):
        output_path = self.datapath / test_id / f'eda_{test_id}_{level}.csv'
        return output_path

    def meta_info_path_gen(self, test_id):
        output_path = self.datapath / test_id / f'info_{test_id}.txt'
        return output_path

    def get_landmark_path(self, subject_id, level):
        # Generate the path for the landmark file.
        output_path = self.datapath / subject_id / 'landmarks' / f"vid_{subject_id}_{level}_landmarks.json"
        return output_path

    def get_rppg_path(self, subject_id, level):
        output_path = self.datapath / subject_id / 'rppg_signals' / f"vid_{subject_id}_{level}_rppg.json"
        return output_path

    def get_all_video_paths(self, subject_list, level=None):
        # Generate a list of all video paths for a given subject list and level.
        if level is None:
            level = ['T1', 'T2', 'T3']
        all_video_paths = []
        for subject_id in subject_list:
            for file_level in level:
                video_path = self.vid_path_gen(subject_id, file_level)
                if video_path.is_file():
                    all_video_paths.append(video_path)
                else:
                    print(f"Warning: Video file {video_path} does not exist.")
        return all_video_paths
