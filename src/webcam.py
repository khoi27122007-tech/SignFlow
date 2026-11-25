
"""Real-time webcam inference for SignFlow."""

from __future__ import annotations

import argparse

from collections import Counter, deque
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


def run(
    model_path: Path,
    camera: int = 0,
    threshold: float = 0.65,
    smooth_window: int = 5,
    mirror: bool = True,
):

    if smooth_window < 1:
        raise ValueError("smooth_window must be >= 1")

    predictor = GesturePredictor(
        model_path=model_path,
        confidence_threshold=threshold,
    )

    extractor = HandLandmarkExtractor(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
    )

    cap = cv2.VideoCapture(camera)

    if not cap.isOpened():
        extractor.close()

        raise RuntimeError(
            f"Cannot open webcam index {camera}"
        )

    history = deque(
        maxlen=smooth_window
    )

    try:

        while True:

            success, frame = cap.read()

            if not success:
                print("Cannot read webcam frame.")
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            features, result = extractor.extract(frame)

            extractor.draw(
                frame,
                result,
            )

            if features is None:

                # Never predict without a detected hand.
                history.clear()

                display_text = "NO HAND DETECTED"

                color = (0, 0, 255)

                cv2.putText(
                    frame,
                    display_text,
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            else:

                label, confidence = predictor.predict(
                    features
                )

                history.append(
                    (label, confidence)
                )

                valid_predictions = [
                    item for item in history
                    if item[0] != "Unknown"
                ]

                if valid_predictions:

                    votes = Counter(
                        item[0]
                        for item in valid_predictions
                    )

                    stable_label, vote_count = votes.most_common(1)[0]

                    required_votes = (
                        1
                        if smooth_window == 1
                        else max(2, smooth_window // 2 + 1)
                    )

                    if vote_count >= required_votes:

                        stable_scores = [
                            score
                            for current_label, score in valid_predictions
                            if current_label == stable_label
                        ]

                        stable_confidence = (
                            sum(stable_scores)
                            / len(stable_scores)
                        )

                        display_text = (
                            f"{stable_label} "
                            f"{stable_confidence:.1%}"
                        )

                        color = (40, 220, 40)

                    else:

                        display_text = "STABILIZING..."

                        color = (0, 200, 255)

                else:

                    display_text = "UNKNOWN"

                    color = (0, 165, 255)

                cv2.putText(
                    frame,
                    display_text,
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                "Q / ESC: QUIT",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "SignFlow - Landmark Recognition",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

    finally:

        cap.release()

        cv2.destroyAllWindows()

        extractor.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="SignFlow real-time landmark recognition."
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
    )

    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--no-mirror",
        action="store_true",
    )

    args = parser.parse_args()

    run(
        model_path=args.model,
        camera=args.camera,
        threshold=args.threshold,
        smooth_window=args.smooth_window,
        mirror=not args.no_mirror,
    )
