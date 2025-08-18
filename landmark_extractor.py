import mediapipe as mp
import file_path_gen
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

        # init file path generator and get subject list
        self.file_path_gen = file_path_gen.FIlePathGen()
        self.subject_list = self.file_path_gen.get_subject_list()

    def _process_single_frame(self, frame_img):
        """
        处理单帧图像并提取人脸关键点。
        这是一个 "工人" 函数。
        """
        landmarks_for_this_frame = []

        # 转换颜色空间并设置标志以提高性能
        frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        # MediaPipe处理
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            landmarks_for_this_frame = [
                {'x': lm.x, 'y': lm.y, 'z': lm.z}
                for lm in results.multi_face_landmarks[0].landmark
            ]

        return landmarks_for_this_frame

    def _process_window(self, capture, start_frame, end_frame):
        """
        处理单个时间窗口内的所有帧。
        这是一个 "工头" 函数。
        """
        current_window_landmarks = {}

        # 定位到窗口的起始帧
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # 遍历窗口内的每一帧
        for current_frame_index in range(start_frame, end_frame):
            success, frame_img = capture.read()
            if not success:
                print(f"Warning: Could not read frame {current_frame_index} from video.")
                break

            # 调用 "工人" 函数处理单帧
            landmarks = self._process_single_frame(frame_img)

            # 记录该帧的结果（即使是空列表）
            current_window_landmarks[current_frame_index] = landmarks

        return current_window_landmarks

    def extract_landmarks_from_video(self, video_path):
        """
        从整个视频中提取所有滑动窗口的关键点数据。
        这是 "总管" 函数，也是对外暴露的主要接口。
        """
        capture = cv2.VideoCapture(str(video_path))  # use str() to ensure compatibility with cv2
        if not capture.isOpened():
            raise ValueError(f"Error: Could not open video file: {video_path}")

        print(f"Processing video: {video_path.name}")

        # get video properties
        video_fps = capture.get(cv2.CAP_PROP_FPS)
        video_total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps == 0 or video_total_frames == 0:
            capture.release()
            raise ValueError("Error: Video file might be corrupted or FPS/frame count is zero.")

        # 计算窗口和步进的帧数
        sliding_window_in_frames = int(self.window_length * video_fps)
        step_in_frames = int(self.step_length * video_fps)

        all_windows_landmarks = []

        # 主循环：滑动窗口
        for window_count, start_frame in enumerate(range(0, video_total_frames, step_in_frames)):
            end_frame = start_frame + sliding_window_in_frames
            if end_frame > video_total_frames:
                end_frame = video_total_frames

            # 调用 "工头" 函数处理一个窗口
            landmarks_in_window = self._process_window(capture, start_frame, end_frame)

            # 收集结果
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

    # def extract_landmarks(self, image):
    #     results = self.face_mesh.process(image)
    #     if results.multi_face_landmarks:
    #         return results.multi_face_landmarks[0]
    #     return None
