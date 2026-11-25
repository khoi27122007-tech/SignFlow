
"""MediaPipe Hands landmark extraction.

Requires MediaPipe 0.10.21 because this project uses the
legacy Solutions API: mediapipe.solutions.hands.
"""

from __future__ import annotations

from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


class HandLandmarkExtractor:
    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        if not hasattr(mp, "solutions"):
            version = getattr(mp, "__version__", "unknown")
            raise RuntimeError(
                f"MediaPipe {version} không còn API mp.solutions.\n"
                "Hãy cài MediaPipe 0.10.21:\n"
                "python -m pip uninstall mediapipe -y\n"
                "python -m pip install mediapipe==0.10.21"
            )

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1,
        )

    @staticmethod
    def normalize_landmarks(hand_landmarks) -> np.ndarray:
        """
        Convert 21 landmarks into 63 normalized float32 features.

        - Wrist (landmark 0) is the origin.
        - Scale is based on wrist-to-middle-MCP distance.
        - The same normalization is used during training and inference.
        """
        points = np.array(
            [
                [point.x, point.y, point.z]
                for point in hand_landmarks.landmark
            ],
            dtype=np.float32,
        )

        if points.shape != (21, 3):
            raise ValueError(
                f"Expected 21 landmarks, received {points.shape}"
            )

        # Move wrist to origin.
        points -= points[0].copy()

        # Scale using wrist -> middle finger MCP.
        scale = float(np.linalg.norm(points[9, :2]))

        if scale < 1e-6:
            scale = float(
                np.max(np.linalg.norm(points[:, :2], axis=1))
            )

        if scale < 1e-6:
            scale = 1.0

        points /= scale

        return points.reshape(63).astype(np.float32)

    def extract(
        self,
        image_bgr: np.ndarray,
    ) -> tuple[Optional[np.ndarray], Optional[object]]:
        """
        Returns:
            features: shape (63,), or None if no hand is detected.
            result: MediaPipe result object.
        """
        if image_bgr is None:
            return None, None

        if not isinstance(image_bgr, np.ndarray):
            raise TypeError("image_bgr must be a numpy.ndarray")

        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Expected a 3-channel BGR image")

        if image_bgr.size == 0:
            return None, None

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        image_rgb.flags.writeable = False

        result = self.hands.process(image_rgb)

        if not result.multi_hand_landmarks:
            return None, result

        # Use the first detected hand.
        hand = result.multi_hand_landmarks[0]

        features = self.normalize_landmarks(hand)

        return features, result

    def draw(
        self,
        image_bgr: np.ndarray,
        result,
    ) -> np.ndarray:
        if result is None:
            return image_bgr

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    image_bgr,
                    hand,
                    self.mp_hands.HAND_CONNECTIONS,
                )

        return image_bgr

    def close(self):
        if getattr(self, "hands", None) is not None:
            self.hands.close()
            self.hands = None
