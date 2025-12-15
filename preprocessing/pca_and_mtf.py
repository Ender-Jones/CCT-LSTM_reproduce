import json

import cv2
import numpy as np
from sklearn.decomposition import PCA
from pyts.image import MarkovTransitionField


class PcaAndMtfProcessor:
    """
    Input:
    landmark dictionary, which is the value of the key 'landmarks' in the window, which is in the json file.
    Output: dictionary: (frame, 478 landmarks, 3 coordinates)
    then apply PCA to the dictionary, and get the pca_features.
    output: 3 x (1, 478) vector
    then apply MTF to the pca_features, and get the mtf_images.
    output: 3 x (478, 478) gray scale images
    rescale the mtf images to (224, 224) and use them as R, G, B channels to build a RGB image.
    output: (224, 224, 3) RGB image

    INFO: the parameter of MTF is set to 8, but the original paper is not clear.
    """

    def __init__(self, image_size=224, mtf_bins=8):
        self.pca = PCA(n_components=1)
        self.mtf = MarkovTransitionField(n_bins=mtf_bins, strategy='quantile')
        self.image_size = image_size

    def _reshape_to_numpy(self, window_landmarks_data):
        """
        Reshape the landmark dictionary of a window to a numpy array.
        Args:
            window_landmarks_data (dict): The 'landmarks' dictionary from a single window.
                                          e.g., {'frame_idx': [{'x':..., 'y':..., 'z':...}, ...], ...}
        Returns:
            np.ndarray: A numpy array of shape (num_frames, 478, 3).
        """
        # Sort frames by index to ensure temporal order
        sorted_frame_indices = sorted(window_landmarks_data.keys(), key=int)

        all_frames_landmarks = []
        for frame_idx in sorted_frame_indices:
            landmarks_for_frame = window_landmarks_data[frame_idx]
            if not landmarks_for_frame or len(landmarks_for_frame) != 478:
                # Handle cases with missing or incomplete landmarks for a frame
                # For now, we skip such frames, but a more robust solution might be interpolation
                print(f"Warning: Skipping frame {frame_idx} due to missing or incomplete landmarks.")
                continue

            # Convert list of dicts to a (478, 3) numpy array, which is all landmarks of a frame
            frame_array = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks_for_frame])
            all_frames_landmarks.append(frame_array)

        if not all_frames_landmarks:
            raise ValueError("No valid frames with landmarks found in the window data.")

        # stack all frames landmarks to a (num_frames, 478, 3) numpy array
        # which is all landmarks of all frames in a window
        return np.stack(all_frames_landmarks, axis=0)

    def _process_single_coordinate(self, coords_array):
        """
        Apply PCA and then MTF to a single coordinate array.
        Args:
            coords_array (np.ndarray): Array of shape (num_frames, 478) for a single coordinate (e.g., x).
        Returns:
            np.ndarray: A MTF image of shape (478, 478) or None if unstable.
        """
        # PCA works on samples as rows. Here, each landmark is a feature over time.
        # So we transpose to (478, num_frames) to apply PCA on the time dimension.
        pca_result = self.pca.fit_transform(coords_array.T).T  # Shape: (1, 478)

        # z-score on principal component; if unstable, return None to skip window
        ts = self._zscore_normalize(pca_result[0])
        if ts is None:
            return None

        # MTF expects a time series. The output is (n_samples, height, width).
        mtf_image = self.mtf.fit_transform(ts[None, :])[0]  # Shape: (len, len)
        return mtf_image

    def _reshape_to_rgb(self, mtf_images):
        """
        Resize the 3 MTF images and stack them into a single RGB image.
        Args:
            mtf_images (list[np.ndarray]): A list of 3 MTF images (for x, y, z) of shape (478, 478).
        Returns:
            np.ndarray: A single RGB image of shape (image_size, image_size, 3).
        """
        # The resizing is now handled by torchvision.transforms in the Dataset,
        # so we will save the full resolution images.
        # resized_images = [cv2.resize(img, (self.image_size, self.image_size)) for img in mtf_images]

        # Stack the channels along the last axis to form an RGB image
        rgb_image = np.stack(mtf_images, axis=-1)

        # Normalize to 0-255 and convert to uint8 for image representation
        # This normalization is a standard practice for saving image files.
        rgb_image = cv2.normalize(rgb_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        return rgb_image

    def _zscore_normalize(self, x: np.ndarray):
        """
        Z-score normalize before applying MTF.
        z = (x - mu) / sigma，其中 mu/sigma 为当前窗口的均值/标准差。
        Args:
            np.ndarray: The numpy array to normalize.
        Returns:
            np.ndarray: The normalized numpy array.
        """
        x = np.asarray(x, dtype=np.float32)
        mu = np.nanmean(x)
        sigma = np.nanstd(x)

        if not np.isfinite(mu) or sigma < 1e-8:
            return None
        return (x - mu) / sigma

    def _validate_series(self, x, expected_len=None, min_len_ratio=0.8, min_valid_ratio=0.95, min_std=1e-6):
        x = np.asarray(x, dtype=np.float32)
        if expected_len is not None and len(x) < expected_len * min_len_ratio:
            return False, "too_short"
        valid = np.isfinite(x)
        if valid.mean() < min_valid_ratio:
            return False, "too_many_nans"
        if np.std(x[valid]) < min_std or np.ptp(x[valid]) < 1e-6:
            return False, "too_flat"
        return True, None

    def transform(self, window_data):
        """
        Orchestrates the full transformation from window landmark data to an RGB image.
        Args:
            window_data (dict): The 'landmarks' dictionary from a single window's data.
        """
        # 1. Reshape the raw window data into a structured Numpy array
        numpy_array = self._reshape_to_numpy(window_data)

        # 2. Decompose the array into three separate arrays for x, y, z coordinates
        coords_x = numpy_array[:, :, 0]
        coords_y = numpy_array[:, :, 1]
        coords_z = numpy_array[:, :, 2]

        # 3. Apply PCA and MTF to each coordinate array individually
        mtf_x = self._process_single_coordinate(coords_x)
        mtf_y = self._process_single_coordinate(coords_y)
        mtf_z = self._process_single_coordinate(coords_z)
        # 任一通道不稳定则返回 None，调用方跳过该窗口
        if mtf_x is None or mtf_y is None or mtf_z is None:
            return None

        # 4. Resize the three MTF images and combine them into a single RGB image
        final_rgb_image = self._reshape_to_rgb([mtf_x, mtf_y, mtf_z])

        return final_rgb_image

    def process_landmark_to_mtf(self, current_landmark_path):
        """
        A convenience method to process landmark data directly.
        so I don't have to deal with all the logic in main.py
        Args:
            current_landmark_path (Path): The path to the current landmark file.
        Returns:
            np.ndarray: The resulting RGB image.
        """
        with open(current_landmark_path, 'r') as f:
            window_landmarks_data = json.load(f)

        # Create output directory for the images
        output_dir = current_landmark_path.parent.parent / 'mtf_images' / 'landmark'
        output_dir.mkdir(parents=True, exist_ok=True)

        for window_data in window_landmarks_data:
            window_id = window_data['window_id']

            # Define output path and check for existence
            # Use zero-padding for window_id to ensure correct alphabetical sorting
            # file name e.g.: vid_s21_T1_landmarks.json
            image_name = f"{current_landmark_path.parent.parent.name}_{current_landmark_path.stem.split('_')[2]}_window_{window_id:03d}.png"
            output_image_path = output_dir / image_name
            if output_image_path.exists():
                print(
                    f"Skipping {current_landmark_path.parent.parent.name}/"
                    f"{current_landmark_path.stem.split('_')[2]}/"
                    f"window_{window_id}: Image already exists.")
                continue

            landmarks = window_data['landmarks']
            if not landmarks:
                print(
                    f"Skipping window {window_id} for {current_landmark_path.parent.parent.name}/{current_landmark_path.stem.split('_')[1]}: No landmarks found.")
                continue

            # Transform data and save the image
            rgb_image = self.transform(landmarks)
            if rgb_image is None:
                print(
                    f"Skipping window {window_id} for {current_landmark_path.parent.parent.name}/"
                    f"{current_landmark_path.stem.split('_')[2]}: Unstable landmark series (z-score invalid).")
                continue
            cv2.imwrite(str(output_image_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

        return None

    def _process_rppg_to_mtf(self, rppgs, max_len: int = None):
        """
        将一段 rPPG/BVP 一维序列转换为 RGB MTF 图像。
        说明：下采样暂时关闭；若需开启，将 max_len 设为正整数并取消下方注释。
        """
        x = np.asarray(rppgs, dtype=np.float32)
        # 下采样暂时关闭
        # if max_len is not None and len(x) > max_len:
        #     idx = np.linspace(0, len(x) - 1, max_len).astype(int)
        #     x = x[idx]
        x = self._zscore_normalize(x)
        if x is None:
            return None
        # 使用当前的 MTF 配置（quantile + n_bins）
        img = self.mtf.fit_transform(x[None, :])[0]  # (n, n)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # 与 landmark 流保持一致：尺寸在 Dataset 中再统一 resize
        return np.stack([img, img, img], axis=-1)

    def process_rppg_to_mtf(self, current_rppg_path):
        """
        A convenience method to process rPPG data directly.
        so I don't have to deal with all the logic in main.py
        Args:
            current_rppg_path (Path): The path to the current rPPG file.
        Returns:
            np.ndarray: The resulting RGB image.
        """
        with open(current_rppg_path, 'r') as f:
            rppg_data = json.load(f)

        # Create output directory for rppg images
        output_dir = current_rppg_path.parent.parent / 'mtf_images' / 'rppg'
        output_dir.mkdir(parents=True, exist_ok=True)

        # 从路径解析 subject 与 level
        subject = current_rppg_path.parent.parent.name  # 'sXX'
        level = current_rppg_path.stem.split('_')[2]  # 'T1'|'T2'|'T3'

        for window in rppg_data:
            window_id = window.get('window_id')
            start_frame = window.get('start_frame')
            end_frame = window.get('end_frame')
            expected_len = None
            if start_frame is not None and end_frame is not None:
                expected_len = int(end_frame - start_frame + 1)

            image_name = f"{subject}_{level}_window_{int(window_id):03d}.png"
            output_image_path = output_dir / image_name
            if output_image_path.exists():
                print(
                    f"Skipping {subject}/{level}/window_{window_id}: Image already exists.")
                continue

            rppgs = window.get('bvp_signal', [])
            if not rppgs:
                print(f"Skipping window {window_id} for {subject}/{level}: No rPPG signal found.")
                continue

            ok, reason = self._validate_series(rppgs, expected_len=expected_len)
            if not ok:
                print(f"Skipping window {window_id} for {subject}/{level}: invalid series ({reason}).")
                continue

            rgb_image = self._process_rppg_to_mtf(rppgs, max_len=None)
            if rgb_image is None:
                print(f"Skipping window {window_id} for {subject}/{level}: Unstable rPPG series (z-score invalid).")
                continue
            cv2.imwrite(str(output_image_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

        # 批处理函数不返回图像
        return None
