import mediapipe as mp
import file_path_gen

class LandmarkExtractor:
    def __init__(self,
                 window_length = 60,
                 step_length = 5,
                 static_image_mode=False,
                 max_num_faces=1,
                 refine_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        # hyperparameters for windowing
        self.window_length = window_length  # in seconds
        self.step_length = step_length  # in seconds

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # init file path generator and get subject list
        self.file_path_gen = file_path_gen.FIlePathGen()
        self.subject_list = self.file_path_gen.get_subject_list()

    def extract_landmarks(self, image):
        results = self.face_mesh.process(image)
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        return None

