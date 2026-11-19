"""Real-time webcam inference. Classifies every frame, including while the hand moves."""
from __future__ import annotations
import argparse
from collections import deque, Counter
from pathlib import Path
import cv2

try:
    from .config import DEFAULT_MODEL_PATH
    from .landmarks import HandLandmarkExtractor
    from .predictor import GesturePredictor
except ImportError:
    from config import DEFAULT_MODEL_PATH
    from landmarks import HandLandmarkExtractor
    from predictor import GesturePredictor


def run(model_path: Path, camera: int, threshold: float, smooth_window: int, mirror: bool):
    predictor = GesturePredictor(model_path, confidence_threshold=threshold)
    extractor = HandLandmarkExtractor(static_image_mode=False, max_num_hands=1,
                                      min_detection_confidence=0.55, min_tracking_confidence=0.55)
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        extractor.close()
        raise RuntimeError(f"Cannot open webcam index {camera}")

    history = deque(maxlen=max(1, smooth_window))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            features, result = extractor.extract(frame)
            extractor.draw(frame, result)

            if features is not None:
                label, confidence = predictor.predict(features)
                history.append(label if label != 'Unknown' else None)
                valid = [x for x in history if x is not None]
                stable_label = Counter(valid).most_common(1)[0][0] if valid else 'Unknown'
                text = f"{stable_label}  {confidence:.1%}"
                color = (40, 220, 40) if stable_label != 'Unknown' else (0, 180, 255)
                cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            else:
                history.clear()
                cv2.putText(frame, 'No hand detected', (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(frame, 'Q: quit', (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('Hand Gesture Recognition | 21 Landmarks', frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.60)
    parser.add_argument('--smooth-window', type=int, default=5)
    parser.add_argument('--no-mirror', action='store_true')
    args = parser.parse_args()
    run(args.model, args.camera, args.threshold, args.smooth_window, not args.no_mirror)
