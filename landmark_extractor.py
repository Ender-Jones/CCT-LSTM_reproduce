import mediapipe as mp
import cv2
import json


class LandmarkExtractor:
    def __init__(self,
                 window_length=60,
                 step_length=5,
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

    @staticmethod
    def _cast_landmarks_to_json(landmark_result_obj):
        """use list comprehension to convert landmarks to JSON serializable format """
        if landmark_result_obj and landmark_result_obj.multi_face_landmarks:
            return [
                {'x': lm.x, 'y': lm.y, 'z': lm.z}
                for lm in landmark_result_obj.multi_face_landmarks[0].landmark
            ]
        # if no landmarks are detected, return none
        return None

    def _process_single_frame(self, frame_bgr):
        """process a single frame to extract landmarks """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # to boost performance, disable writing to the output
        frame_rgb.flags.writeable = False

        # process the frame using MediaPipe Face Mesh
        results = self.face_mesh.process(frame_rgb)

        return self._cast_landmarks_to_json(results)

    def _process_window(self, capture, start_frame, end_frame, video_fps):
        """process all frames in a sliding window"""
        current_window_landmarks = {}

        # position the capture to the start frame
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # loop through frames in the current window
        for current_frame_index in range(start_frame, end_frame):
            success, frame_img = capture.read()
            if not success:
                print(f"Warning: Could not read frame {current_frame_index} from video."
                      f"Around time {current_frame_index / video_fps:.2f} seconds.")
                break

            landmarks = self._process_single_frame(frame_img)
            # report error if no landmarks are detected
            if landmarks is None:
                print(f"Warning: No landmarks detected in frame {current_frame_index}. "
                      f"Around time {current_frame_index / video_fps:.2f} seconds.")
                landmarks = []  # use empty list to indicate no landmarks

            # record the landmarks for the current frame even if they are empty
            current_window_landmarks[current_frame_index] = landmarks

        return current_window_landmarks

    @staticmethod
    def _save_landmarks_to_subject_folder(landmarks_data, video_path):
        """
        save landmarks data to a JSON file
        into subject folder/landmarks/video_name.json
        """
        if not landmarks_data:
            print(f"No landmarks to save for video: {video_path.name}")
            return

        # save the landmarks data to a JSON file in landmarks folder under the subject folder
        # check if the landmarks folder exists, if not create it
        landmarks_folder_path = video_path.parent / 'landmarks'
        landmarks_folder_path.mkdir(parents=True, exist_ok=True)

        # create the JSON file path
        json_file_path = landmarks_folder_path / f"{video_path.stem}_landmarks.json"
        # write the landmarks data to the JSON file
        with open(json_file_path, 'w') as json_file:
            json.dump(landmarks_data, json_file, indent=4)

        print(f"Landmarks saved to {json_file_path}")

    def extract_landmarks_from_video(self, video_path):
        """
        The main orchestrator function
        """
        capture = cv2.VideoCapture(str(video_path))  # use str() to ensure compatibility with cv2
        if not capture.isOpened():
            raise ValueError(f"Error: Could not open video file: {video_path}")

        print(f"Processing video: {video_path.name}")

        # get video properties and check for validity
        video_fps = capture.get(cv2.CAP_PROP_FPS)
        video_total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps == 0 or video_total_frames == 0:
            capture.release()
            raise ValueError("Error: Video file might be corrupted or FPS/frame count is zero.")

        # convert time to frames
        sliding_window_in_frames = int(self.window_length * video_fps)
        step_in_frames = int(self.step_length * video_fps)

        all_windows_landmarks = []  # for storing landmarks for all windows
        # outer loop: slide the window across the video
        for window_count, start_frame in enumerate(range(0, video_total_frames, step_in_frames)):
            end_frame = start_frame + sliding_window_in_frames

            # Ensure the end frame does not exceed total frames
            if end_frame > video_total_frames:
                end_frame = video_total_frames

            # process the current window
            landmarks_in_window = self._process_window(capture, start_frame, end_frame, video_fps)

            # collect landmarks for the current window
            if landmarks_in_window:
                all_windows_landmarks.append({
                    'video_name': video_path.name,
                    'window_id': window_count,
                    'start_frame': start_frame,
                    'end_frame': end_frame - 1,
                    'landmarks': landmarks_in_window
                })

        capture.release()
        return all_windows_landmarks

    # def extract_landmarks_from_video(self, video_path):
    #     capture = cv2.VideoCapture(video_path)
    #     if not capture.isOpened():
    #         raise ValueError(f"Error: Could not open video file: {video_path}")
    #     else:
    #         print(f"Processing video: {video_path.name}")
    #
    #     # Get video properties
    #     video_fps = capture.get(cv2.CAP_PROP_FPS)
    #     video_total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    #
    #     # Check if video properties are valid
    #     if video_fps == 0 or video_total_frames == 0:
    #         raise ValueError("Error: Video file might be corrupted or FPS/frame count is zero.")
    #
    #     # convert time to frames
    #     sliding_window_in_frames = int(self.window_length * video_fps)
    #     step_in_frames = int(self.step_length * video_fps)
    #
    #     # data pool to store landmarks for each window and all windows
    #     all_windows_landmarks = []
    #     current_window_landmarks = {}
    #     window_count = 0
    #
    #     # outer loop: slide the window across the video
    #     for start_frame in range(0, video_total_frames, step_in_frames):
    #         end_frame = start_frame + sliding_window_in_frames
    #
    #         # Ensure the end frame does not exceed total frames
    #         if end_frame > video_total_frames:
    #             end_frame = video_total_frames
    #
    #         # Set the capture to the start frame
    #         capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    #
    #         # Read frames in the current window
    #         for current_frame_index in range(start_frame, end_frame):
    #             success, frame_img = capture.read()
    #             if not success:
    #                 print(f"Error: Could not read frame {current_frame_index} from video.")
    #                 break
    #
    #             # Convert the frame to RGB (MediaPipe uses RGB format)
    #             frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
    #             # boost performance by disabling writing to the output
    #             frame_img.flags.writeable = False
    #
    #             # Process the frame to extract landmarks
    #             results = self.face_mesh.process(frame_img)
    #
    #             # extract landmarks if available
    #             landmarks_for_this_frame = []
    #             if results.multi_face_landmarks:
    #                 landmarks_for_this_frame = [
    #                     {'x': lm.x, 'y': lm.y, 'z': lm.z}
    #                     for lm in results.multi_face_landmarks[0].landmark
    #                 ]
    #             else:
    #                 print(
    #                     f"Warning: No landmarks detected in frame {current_frame_index}."
    #                     f"Around time {current_frame_index / video_fps:.2f} seconds."
    #                 )
    #
    #             # Store the landmarks for the current frame
    #             current_window_landmarks[current_frame_index] = landmarks_for_this_frame
    #
    #         # Save the landmarks for the current window
    #         if current_window_landmarks:
    #             all_windows_landmarks.append(
    #                 {
    #                     'video_name': video_path.name,
    #                     'window_id': window_count,
    #                     'start_frame': start_frame,
    #                     'end_frame': end_frame - 1,  # used range so end_frame is exclusive
    #                     'landmarks': current_window_landmarks
    #                 }
    #             )
    #         window_count += 1
    #         current_window_landmarks = {}  # reset for the next window
    #
    #     return all_windows_landmarks

    def close(self):
        """
        Closes the MediaPipe Face Mesh instance to free up resources.
        """
        print("Closing Face Mesh instance.")
        self.face_mesh.close()
