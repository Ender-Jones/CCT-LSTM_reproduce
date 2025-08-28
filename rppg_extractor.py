import sys
from pathlib import Path
import numpy as np

# Add the local pyVHR clone to the path to allow imports.
# This assumes the 'pyVHR' folder is in the same directory as this script's parent.
pyvhr_path = Path(__file__).parent / 'pyVHR'
if pyvhr_path.exists() and str(pyvhr_path) not in sys.path:
    sys.path.insert(0, str(pyvhr_path))

try:
    # Now we can import the necessary components from pyVHR
    from pyVHR.analysis.video_analysis import VideoAnalysis
    from pyVHR.BPM.BPM_methods import get_BPM
    from pyVHR.utils.errors import get_error_method
except ImportError:
    print("---")
    print("ERROR: pyVHR library not found.")
    print(f"Attempted to look for it in: {pyvhr_path}")
    print("Please make sure you have cloned the pyVHR repository into your project's root folder.")
    print("---")
    sys.exit(1)


class RppgExtractor:
    """
    Extracts rPPG (BVP) signals from video files using the pyVHR library.
    This class is tailored to use the OMIT method as referenced in the Ziaratnia et al. paper.

    NOTE ON DEPENDENCIES:
    The pyVHR library's default face detector is 'dlib'. However, 'dlib' requires a C++
    build environment (including CMake) to be installed, which can be complex to set up.
    To ensure our project is easy to run and to bypass these compilation issues, we have
    explicitly configured this extractor to use the 'mediapipe' face detector instead.
    MediaPipe is a modern, high-performance library from Google that is already used in the
    landmark extraction phase of this project.

    This change in face detector does NOT affect the core rPPG algorithm. The face detector's
    role is simply to locate the face region. The actual rPPG signal is still generated
    using the paper's specified 'OMIT' method. This is a valid engineering choice that
    maintains the scientific integrity of the core methodology replication.
    """

    def __init__(self):
        """
        Initializes the RppgExtractor.
        No special configuration is needed here as pyVHR parameters
        are set directly in the extraction method.
        """
        pass

    def extract_signal(self, video_path):
        """
        Extracts the Blood Volume Pulse (BVP) signal from a single video file using the OMIT method.
        This implementation explicitly uses the 'mediapipe' face detector to avoid
        the dependency on dlib and its complex installation requirements.

        Args:
            video_path (pathlib.Path or str): The path to the video file.

        Returns:
            tuple[np.ndarray, float] or tuple[None, None]: A tuple containing:
                - np.ndarray: The 1D BVP (rPPG) signal as a numpy array. Returns None if processing fails.
                - float: The frames per second (FPS) of the video. Returns None if processing fails.
        """
        try:
            video_path_str = str(video_path)
            
            # 1. Initialize VideoAnalysis with mediapipe as the face detector
            # The first parameter is the number of parallel jobs, -1 means using all available cores.
            v_analysis = VideoAnalysis(-1, 'mediapipe')

            # 2. Run the analysis to extract raw skin pixel data
            # This step performs face detection, skin segmentation, and signal extraction frame by frame.
            v_analysis.run(video_path_str)

            # 3. Get the frames per second (FPS) of the video
            fps = v_analysis.get_fps()

            # 4. Compute the BVP signal using the OMIT method
            # We pass the extracted raw signals and FPS to the get_BPM function
            # and specify the desired rPPG method.
            bvp_OMIT = get_BPM(v_analysis.get_signals(), fps, method='OMIT')

            # The get_BPM method returns a list of signals. For OMIT, it's typically one signal.
            # We return the first element of that list.
            if bvp_OMIT and len(bvp_OMIT) > 0:
                return bvp_OMIT[0], fps
            else:
                print(f"Warning: OMIT method did not return a valid BVP signal for {video_path.name}.")
                return None, None

        except Exception as e:
            print(f"!!! ERROR processing video {video_path.name} with pyVHR: {e}")
            # This can happen if no face is detected in the video, among other reasons.
            return None, None

