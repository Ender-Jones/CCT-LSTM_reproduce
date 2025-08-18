from pathlib import Path

class FIlePathGen:
    # Dynamically generate file paths for UBFC-Phys dataset.
    def __init__(self, config_path: str = "UBFC_data_path.txt"):
        self.datapath_config = Path(config_path)
        if not self.datapath_config.is_file():
            raise FileNotFoundError(
                f"Configuration file {config_path} not found."
                f"Please run 'integrity_and_masterManifest.py' first."
            )

        # Read the data path from text file.
        with open('UBFC_data_path.txt', 'r') as f:
            self.datapath = Path(f.read().strip())

    def get_subject_list(self):
        # Get a list of all subject IDs in the dataset.
        subject_list = [d.name for d in self.datapath.iterdir() if d.is_dir()]
        return subject_list

    def vid_path_gen(self, test_id, level):
        output_path = self.datapath / test_id / f'vid_{test_id}_{level}.avi'
        return output_path

    def bvp_path_gen(self, test_id, level):
        output_path = self.datapath / test_id / f'bvp_{test_id}_{level}.csv'
        return output_path

    def eda_path_gen(self, test_id, level):
        output_path = self.datapath / test_id / f'eda_{test_id}_{level}.csv'
        return output_path

    def meta_info_path_gen(self, test_id, level):
        output_path = self.datapath / test_id / f'info_{test_id}.txt'
        return output_path
