from pathlib import Path
from typing import List, Tuple, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from file_path_gen import FilePathGen


class SingleImageDataset(Dataset):
    """
    PyTorch Dataset for the CCT pre-training phase (Stage 1).
    In this mode, a sample is every single MTF picture.
    """

    def __init__(self, data_root: Path, modality: str, subject_list: List[str], transform: Callable = None):
        """
        Initializes the dataset.

        Args:
            data_root (Path): The root directory of the dataset (e.g., '/path/to/UBFC-Phys/Data').
            modality (str): The data modality to load. Must be 'landmark' or 'rppg'.
            subject_list (List[str]): A list of subject IDs (e.g., ['s1', 's5', 's10']) to include in this dataset split.
            transform (Callable, optional): Transformations to be applied to each image sample.
        """
        self.data_root = Path(data_root)
        self.modality = modality.lower()
        self.subject_list = subject_list
        self.transform = transform

        if self.modality not in ['landmark', 'rppg']:
            raise ValueError(f"Unknown modality: {self.modality}")

        self.samples = []  # flat list of (image_path, label)
        label_map = {'T1': 0, 'T2': 1, 'T3': 2}

        for subject_id in subject_list:
            mtf_dir = self.data_root / subject_id / 'mtf_images' / self.modality

            if not mtf_dir.is_dir():
                # This is a warning, not an error, as subjects might be in different CV folds.
                print(f"Warning: Directory not found for subject '{subject_id}', modality '{self.modality}'. Skipping.")
                continue

            # Sorting is not strictly necessary here, but it's good practice for reproducibility.
            image_paths = sorted(list(mtf_dir.glob('*.png')))

            for img_path in image_paths:
                try:
                    # Expecting filenames like: s1_T2_window_029.png
                    image_id = img_path.stem  # remove the suffix(.png)
                    parts = image_id.split('_')

                    level_str = parts[1]
                    label = label_map[level_str]

                    self.samples.append((img_path, label))

                except (IndexError, KeyError):
                    print(f"Warning: Could not parse label from filename, skipping: {img_path.name}")
                    continue

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieves a sample from the dataset at the specified index.
        Args:
            idx (int): The index of the sample to retrieve.
        Returns:
            A tuple containing the transformed image tensor and its corresponding integer label.
        """
        image_path, label = self.samples[idx]

        try:
            # Open the image using Pillow
            image = Image.open(image_path).convert('RGB')

            # Apply transformations if they are provided
            if self.transform:
                image = self.transform(image)

            return image, label

        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}. Skipping.")
            # To handle this gracefully, we can return the next valid sample
            # or raise an exception if strictness is required.
            # For robustness in training, returning a sample from a different index is a common approach.
            return self.__getitem__((idx + 1) % len(self))
        except Exception as e:
            print(f"Error loading or transforming image {image_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))


class VideoSequenceDataset(Dataset):
    """
    PyTorch Dataset for the CCT-LSTM training phase (Stage 2).
    In this mode, a sample consists of the sequence of all MTF image pairs from a single video.
    """

    MIN_SEQ_LENGTH = 10

    def __init__(self, data_root: Path, subject_list: List[str], transform: Callable = None):
        """
        Initializes the dataset by pre-loading and verifying all video sequence paths.

        Args:
            data_root (Path): The root directory of the dataset.
            subject_list (List[str]): A list of subject IDs to include in this dataset split.
            transform (Callable, optional): Transformations to be applied to each image.
        """
        self.data_root = Path(data_root)
        self.subject_list = subject_list
        self.transform = transform

        self.sequences = []  # A list of tuples: (landmark_paths, rppg_paths, label)
        label_map = {'T1': 0, 'T2': 1, 'T3': 2}

        print("Initializing VideoSequenceDataset, scanning for valid sequences...")
        for subject_id in subject_list:
            for level_str, label in label_map.items():
                landmark_dir = self.data_root / subject_id / 'mtf_images' / 'landmark'
                rppg_dir = self.data_root / subject_id / 'mtf_images' / 'rppg'

                # The file existence is checked here during initialization.
                if not landmark_dir.is_dir() or not rppg_dir.is_dir():
                    # TODO(multilevel-labels): 未来在做多等级压力分类（T1, T3-ctrl, T3-test）时，
                    # 这里需要结合 master_manifest.csv 中的 group(ctrl/test) 信息进行筛选。
                    print(
                        f"Warning: Missing modality directories for subject '{subject_id}'. "
                        f"landmark_dir_exists={landmark_dir.is_dir()}, rppg_dir_exists={rppg_dir.is_dir()}. Skipping."
                    )
                    continue

                landmark_paths = sorted(landmark_dir.glob(f'{subject_id}_{level_str}_window_*.png'))
                rppg_paths = sorted(rppg_dir.glob(f'{subject_id}_{level_str}_window_*.png'))

                if not landmark_paths and not rppg_paths:
                    continue

                # 解析 window_id -> 路径
                def _parse_window_id(path: Path) -> int:
                    try:
                        return int(path.stem.split('_')[-1])
                    except (IndexError, ValueError):
                        return -1

                lm_map = {wid: p for p in landmark_paths if (wid := _parse_window_id(p)) >= 0}
                rp_map = {wid: p for p in rppg_paths if (wid := _parse_window_id(p)) >= 0}

                # get the subset of window ids that are common to both landmark and rppg paths
                common_ids = sorted(set(lm_map.keys()) & set(rp_map.keys()))

                if len(common_ids) < self.MIN_SEQ_LENGTH:
                    if landmark_paths or rppg_paths:
                        print(
                            f"Warning: Insufficient paired windows for {subject_id}/{level_str}. "
                            f"paired={len(common_ids)}, landmark={len(landmark_paths)}, rppg={len(rppg_paths)}. Skipping."
                        )
                    continue

                seq_landmark = [lm_map[i] for i in common_ids]
                seq_rppg = [rp_map[i] for i in common_ids]
                self.sequences.append((seq_landmark, seq_rppg, label))

    def __len__(self) -> int:
        """Returns the total number of valid video sequences in the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieves a complete video sequence sample based on the given index.

        Args:
            idx (int): The index of the sample.

        Returns:
            A tuple containing the sequence tensor and its single label.
            The shape of the sequence tensor: (sequence_length, 2, channels, height, width)
        """
        landmark_paths, rppg_paths, label = self.sequences[idx]

        landmark_tensors = []
        rppg_tensors = []

        for landmark_path, rppg_path in zip(landmark_paths, rppg_paths):
            try:
                # Load images using Pillow
                landmark_image = Image.open(landmark_path).convert('RGB')
                rppg_image = Image.open(rppg_path).convert('RGB')

                # Apply transformations
                if self.transform:
                    landmark_image = self.transform(landmark_image)
                    rppg_image = self.transform(rppg_image)

                landmark_tensors.append(landmark_image)
                rppg_tensors.append(rppg_image)

            except FileNotFoundError:
                print(f"Error: Image file not found at {landmark_path} or {rppg_path}. Skipping this pair.")
                continue
            except Exception as e:
                print(f"Error loading or transforming images {landmark_path} or {rppg_path}: {e}. Skipping this pair.")
                continue
        
        # If a sequence ends up empty because all pairs failed to load, handle it.
        if not landmark_tensors or not rppg_tensors:
            print(f"Warning: No valid image pairs found for index {idx}. Returning next sample.")
            return self.__getitem__((idx + 1) % len(self))

        # Stack the lists into tensors of shape (N, C, H, W)
        # N = sequence length, C = channels, H = height, W = width
        landmark_sequence = torch.stack(landmark_tensors)
        rppg_sequence = torch.stack(rppg_tensors)

        # Stack the two modalities along a new dimension (dim=1)
        # to create the final tensor of shape (N, 2, C, H, W).
        final_sequence = torch.stack([landmark_sequence, rppg_sequence], dim=1)

        return final_sequence, label


def get_default_transforms() -> Callable:
    """
    Returns a composition of default transformations for the MTF images.
    The pipeline includes:
    1. Resizing the image to the standard 224x224 size.
    2. Converting the image to a PyTorch tensor.
    3. Normalizing the tensor channels with ImageNet's mean and standard deviation.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Normalizing with ImageNet stats is a common practice and a good starting point.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


if __name__ == '__main__':
    # A simple usage example and testing area.

    # Assume fpg.datapath points to the root of the dataset, e.g., .../UBFC-Phys/Data
    fpg = FilePathGen()
    DATA_ROOT = fpg.datapath

    # --- Test SingleImageDataset ---
    print("--- Testing SingleImageDataset ---")
    # Let's use subjects s1 to s10 for testing.
    test_subjects = [f's{i}' for i in range(1, 11)]
    try:
        # Load only the landmark modality
        landmark_dataset = SingleImageDataset(
            data_root=DATA_ROOT,
            modality='landmark',
            subject_list=test_subjects,
            transform=get_default_transforms()
        )
        print(f"Successfully created Landmark Dataset. Total samples: {len(landmark_dataset)}")
        # To get the first sample:
        # if len(landmark_dataset) > 0:
        #     img, lbl = landmark_dataset[0]
        #     print(f"Shape of the first sample: {img.shape}, Label: {lbl}")
    except Exception as e:
        print(f"Error creating SingleImageDataset: {e}")
        print("Please ensure you have generated MTF images for the 'landmark' modality and the path is correct.")

    # --- Test VideoSequenceDataset ---
    print("\n--- Testing VideoSequenceDataset ---")
    try:
        sequence_dataset = VideoSequenceDataset(
            data_root=DATA_ROOT,
            subject_list=test_subjects,
            transform=get_default_transforms()
        )
        print(f"Successfully created Video Sequence Dataset. Total videos: {len(sequence_dataset)}")
        # To get the first video sequence:
        # if len(sequence_dataset) > 0:
        #     seq, lbl = sequence_dataset[0]
        #     print(f"Shape of the first sequence: {seq.shape}, Label: {lbl}")
    except Exception as e:
        print(f"Error creating VideoSequenceDataset: {e}")
        print("Please ensure you have generated MTF images for both 'landmark' and 'rppg' modalities and the path is correct.")
