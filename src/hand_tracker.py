from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np
import mediapipe as mp


@dataclass
class HandResult:
    detected: bool
    landmarks: Optional[np.ndarray] = None   # (21, 3), normalized to wrist and hand scale
    bbox: Optional[tuple[int, int, int, int]] = None
    handedness: str = "Unknown"
    confidence: float = 0.0


class HandTracker:
    """MediaPipe hand detector with 21 landmarks and a padded image ROI."""

    def __init__(self, max_hands=1, detection_confidence=0.65,
                 tracking_confidence=0.60, model_complexity=1):
        self.mp_hands = mp.solutions.hands
        self.drawer = mp.solutions.drawing_utils
        self.connections = self.mp_hands.HAND_CONNECTIONS
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    @staticmethod
    def _normalize(points: np.ndarray) -> np.ndarray:
        points = points.astype(np.float32).copy()
        points -= points[0:1, :]  # wrist as origin
        scale = float(np.max(np.linalg.norm(points[:, :2], axis=1)))
        if scale > 1e-6:
            points /= scale
        return points

    def process(self, frame_bgr, draw=True, padding=0.22) -> HandResult:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            return HandResult(False)

        hand = result.multi_hand_landmarks[0]
        raw = np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=np.float32)
        xs, ys = raw[:, 0], raw[:, 1]
        x1, x2 = int(xs.min() * w), int(xs.max() * w)
        y1, y2 = int(ys.min() * h), int(ys.max() * h)
        side = max(x2 - x1, y2 - y1, 1)
        pad = int(side * padding)
        x1, y1 = max(0, x1-pad), max(0, y1-pad)
        x2, y2 = min(w, x2+pad), min(h, y2+pad)

        handedness, score = "Unknown", 0.0
        if result.multi_handedness:
            cls = result.multi_handedness[0].classification[0]
            handedness, score = cls.label, float(cls.score)

        if draw:
            self.drawer.draw_landmarks(frame_bgr, hand, self.connections)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (60, 220, 90), 2)

        return HandResult(
            detected=True,
            landmarks=self._normalize(raw),
            bbox=(x1, y1, x2, y2),
            handedness=handedness,
            confidence=score,
        )

    def close(self):
        self.hands.close()
