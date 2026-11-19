"""MediaPipe extraction and normalization for 21 hand landmarks."""
from __future__ import annotations

from typing import Optional
import numpy as np
import cv2
import mediapipe as mp


class HandLandmarkExtractor:
    def __init__(self, static_image_mode: bool = True, max_num_hands: int = 1,
                 min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils

    @staticmethod
    def normalize_landmarks(hand_landmarks) -> np.ndarray:
        """Return 63 normalized values. Wrist is origin; palm width sets scale."""
        points = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark], dtype=np.float32)
        if points.shape != (21, 3):
            raise ValueError(f"Expected 21 landmarks, got {points.shape}")
        points -= points[0]  # translate wrist to origin
        # Scale by distance from wrist to middle-finger MCP (landmark 9).
        scale = float(np.linalg.norm(points[9, :2]))
        if scale < 1e-6:
            scale = float(np.max(np.linalg.norm(points[:, :2], axis=1)))
        if scale < 1e-6:
            scale = 1.0
        points /= scale
        return points.reshape(-1).astype(np.float32)

    def extract(self, image_bgr: np.ndarray) -> tuple[Optional[np.ndarray], Optional[object]]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        result = self.hands.process(image_rgb)
        if not result.multi_hand_landmarks:
            return None, result
        return self.normalize_landmarks(result.multi_hand_landmarks[0]), result

    def draw(self, image_bgr: np.ndarray, result) -> np.ndarray:
        if result and result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(image_bgr, hand, self.mp_hands.HAND_CONNECTIONS)
        return image_bgr

    def close(self):
        self.hands.close()
