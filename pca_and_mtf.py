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
            np.ndarray: A MTF image of shape (478, 478).
        """
        # PCA works on samples as rows. Here, each landmark is a feature over time.
        # So we transpose to (478, num_frames) to apply PCA on the time dimension.
        pca_result = self.pca.fit_transform(coords_array.T).T  # Shape: (1, 478)
        
        # MTF expects a time series. The output is (n_samples, height, width).
        mtf_image = self.mtf.fit_transform(pca_result)[0]  # Shape: (478, 478)
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
        
        # 4. Resize the three MTF images and combine them into a single RGB image
        final_rgb_image = self._reshape_to_rgb([mtf_x, mtf_y, mtf_z])

        return final_rgb_image
