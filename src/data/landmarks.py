"""MediaPipe Face Landmarker adapter; accepts only a supplied official model asset."""

from pathlib import Path

import numpy as np

from src.models.rules import eye_aspect_ratio, mouth_aspect_ratio, pitch_degrees


class FaceFeatures:
    def __init__(self, model_path):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        if not Path(model_path).is_file():
            raise FileNotFoundError("Supply the official face_landmarker.task model asset")
        self.mp = mp
        self.detector = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                output_facial_transformation_matrixes=True,
            )
        )

    def extract(self, rgb, timestamp_ms):
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks or not result.facial_transformation_matrixes:
            return {"ear": None, "mar": None, "pitch": None}
        h, w = rgb.shape[:2]
        points = np.array([(p.x * w, p.y * h) for p in result.face_landmarks[0]])
        ear = (
            eye_aspect_ratio(points[[33, 160, 158, 133, 153, 144]])
            + eye_aspect_ratio(points[[362, 385, 387, 263, 373, 380]])
        ) / 2
        mar = mouth_aspect_ratio(points[[78, 82, 13, 312, 308, 317, 14, 87]])
        return {
            "ear": ear,
            "mar": mar,
            "pitch": pitch_degrees(result.facial_transformation_matrixes[0]),
        }

    def close(self):
        self.detector.close()
