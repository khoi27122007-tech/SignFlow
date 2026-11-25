"""Build a landmark dataset for ASL alphabet + del + space.

The `nothing` folder is intentionally excluded: it represents absence of a
hand, not a hand gesture. Webcam handles that state before calling the model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    from .config import IMAGE_EXTENSIONS, DEFAULT_CACHE_PATH, FEATURE_SIZE
    from .landmarks import HandLandmarkExtractor
except ImportError:
    from config import IMAGE_EXTENSIONS, DEFAULT_CACHE_PATH, FEATURE_SIZE
    from landmarks import HandLandmarkExtractor

# Stable order shared by dataset creation, training reports and inference.
EXPECTED_CLASSES = [chr(code) for code in range(ord("A"), ord("Z") + 1)] + [
    "del", "space"
]
EXCLUDED_CLASSES = {"nothing"}


def build(
    dataset_dir: Path,
    output: Path,
    max_num_hands: int = 1,
    min_detection_confidence: float = 0.5,
    min_samples_per_class: int = 2,
    max_images_per_class: int | None = None,
):
    dataset_dir = Path(dataset_dir)
    output = Path(output)

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir.resolve()}")
    if min_samples_per_class < 2:
        raise ValueError("min_samples_per_class must be >= 2 for train/validation split.")
    if max_images_per_class is not None and max_images_per_class < 1:
        raise ValueError("max_images_per_class must be >= 1")

    folders = {p.name: p for p in dataset_dir.iterdir() if p.is_dir()}
    class_names = [name for name in EXPECTED_CLASSES if name in folders]
    missing = [name for name in EXPECTED_CLASSES if name not in folders]
    unexpected = sorted(set(folders) - set(EXPECTED_CLASSES) - EXCLUDED_CLASSES)

    if missing:
        raise RuntimeError(f"Dataset is missing class folders: {missing}")
    if unexpected:
        print("Ignoring folders not used by this landmark classifier:", unexpected)

    print("=" * 70)
    print("SIGNFLOW - LANDMARK DATASET BUILDER")
    print("=" * 70)
    print("Dataset:", dataset_dir.resolve())
    print("Output:", output.resolve())
    print("Gesture classes:", len(class_names), class_names)
    if "nothing" in folders:
        print("Excluded `nothing`: no-hand state is handled by webcam, not trained.")

    extractor = HandLandmarkExtractor(
        static_image_mode=True,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=0.5,
    )

    all_features: list[np.ndarray] = []
    all_labels: list[int] = []
    reports = []

    try:
        for class_index, class_name in enumerate(class_names):
            class_dir = folders[class_name]
            image_files = sorted(
                p for p in class_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if max_images_per_class is not None:
                image_files = image_files[:max_images_per_class]

            class_features = []
            unreadable = 0
            no_hand = 0

            for image_path in tqdm(image_files, desc=f"Class {class_name}", unit="image"):
                image = cv2.imread(str(image_path))
                if image is None:
                    unreadable += 1
                    continue
                try:
                    features, _ = extractor.extract(image)
                except Exception as exc:
                    print(f"\nWarning: {image_path}: {exc}")
                    no_hand += 1
                    continue
                if features is None:
                    no_hand += 1
                    continue
                class_features.append(features)

            detected = len(class_features)
            total = len(image_files)
            rate = detected / total * 100 if total else 0.0
            reports.append((class_name, total, detected, unreadable, no_hand, rate))
            print(
                f"\n[{class_name}] total={total}, detected={detected}, "
                f"unreadable={unreadable}, no_hand_or_error={no_hand}, rate={rate:.2f}%"
            )

            # Crucially, use the index in the filtered canonical class list.
            all_features.extend(class_features)
            all_labels.extend([class_index] * detected)
    finally:
        extractor.close()

    print("\n" + "=" * 70)
    print(f"{'CLASS':<12}{'TOTAL':>9}{'DETECTED':>12}{'RATE':>10}")
    for name, total, detected, unreadable, no_hand, rate in reports:
        print(f"{name:<12}{total:>9}{detected:>12}{rate:>9.2f}%")

    insufficient = [
        (name, detected) for name, _, detected, *_ in reports
        if detected < min_samples_per_class
    ]
    if insufficient:
        raise RuntimeError(
            "Not enough detected hand samples in these classes: "
            f"{insufficient}. No dataset was saved. Check those class images "
            "and MediaPipe detection; do not include `nothing` as a gesture class."
        )

    X = np.asarray(all_features, dtype=np.float32)
    y = np.asarray(all_labels, dtype=np.int64)

    if X.ndim != 2 or X.shape[1] != FEATURE_SIZE:
        raise RuntimeError(f"Expected X shape (N, {FEATURE_SIZE}), got {X.shape}")
    if not np.isfinite(X).all():
        raise RuntimeError("Extracted dataset contains NaN or infinite values.")
    if set(np.unique(y).tolist()) != set(range(len(class_names))):
        raise RuntimeError("Class indices are incomplete or incorrectly mapped.")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        X=X,
        y=y,
        class_names=np.asarray(class_names, dtype=str),
    )

    print("\nDATASET BUILD COMPLETED")
    print("Samples:", len(X))
    print("Feature shape:", X.shape)
    print("Classes:", len(class_names))
    print("Saved:", output.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build 63-feature hand landmark dataset.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--max-num-hands", type=int, default=1)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-samples-per-class", type=int, default=2)
    parser.add_argument("--max-images-per-class", type=int, default=None)
    args = parser.parse_args()

    build(
        dataset_dir=args.dataset,
        output=args.output,
        max_num_hands=args.max_num_hands,
        min_detection_confidence=args.min_detection_confidence,
        min_samples_per_class=args.min_samples_per_class,
        max_images_per_class=args.max_images_per_class,
    )
