import json
import file_path_gen
import cv2
import numpy as np
import tqdm
from sklearn.decomposition import PCA
from pyts.image import MarkovTransitionField


class PCAandMTFProcessor:
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
        self.mtf = MarkovTransitionField(n_bins=mtf_bins)
        self.image_size = image_size

    def load_landmark_json_data(self, subject_id, level):
        """The output should match original paper: (frame, 478 landmarks, 3 coordinates)"""
        landmark_path = self.fpg.get_landmark_path(subject_id, level)
        if not landmark_path.is_file():
            raise FileNotFoundError(f"Landmark file {landmark_path} does not exist.")

        with open(landmark_path, 'r') as f:
            landmarks_data = json.load(f)
            # the landmarks_data format:
            # [{}, {}, ...] where each dict contains 'video_name', 'window_id', 'start_frame', 'end_frame', 'landmarks'
            if not landmarks_data:
                raise ValueError(f"No landmark data found in {landmark_path}.")

        return landmarks_data

    def _reshape_to_numpy(self, window_data):

        pass

    def _apply_pca(self, window_data):

        pass

    def _apply_mtf(self, window_data):

        pass

    def transform(self, window_data):
        """
        1. reshape the window_data to (frame, 478 landmarks, 3 coordinates)
        2. disassemble the numpy_array to 3 x (frame, 478)
        """
        # 1. 重塑
        numpy_array = self._reshape_to_numpy(window_data)

        # 2. 分解坐标
        coords_x = numpy_array[:, :, 0]
        coords_y = numpy_array[:, :, 1]
        coords_z = numpy_array[:, :, 2]
        # 3. 应用PCA
        pca_features = self._apply_pca(coords_x, coords_y, coords_z)
        # 4. 应用MTF
        mtf_images = self._apply_mtf(pca_features)
        # 5. 重塑为RGB图像
        final_rgb_image = self._reshape_to_rgb(mtf_images)

        return final_rgb_image
