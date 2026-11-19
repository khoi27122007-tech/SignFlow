"""Convert an image-folder dataset into normalized 21-landmark samples.

Expected layout:
    dataset_root/
      A/ image1.jpg image2.jpg ...
      B/ ...
      C/ ...
Each immediate subfolder is treated as one class label.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

try:
    from .config import IMAGE_EXTENSIONS, DEFAULT_CACHE_PATH
    from .landmarks import HandLandmarkExtractor
except ImportError:
    from config import IMAGE_EXTENSIONS, DEFAULT_CACHE_PATH
    from landmarks import HandLandmarkExtractor


def build(dataset_dir: Path, output: Path, max_num_hands: int = 1):
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")
    class_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
    if not class_dirs:
        raise RuntimeError("Dataset must contain one subfolder per class, e.g. dataset/A/*.jpg")

    extractor = HandLandmarkExtractor(static_image_mode=True, max_num_hands=max_num_hands)
    features, labels, class_names, skipped = [], [], [], []
    try:
        for class_dir in class_dirs:
            files = sorted(p for p in class_dir.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
            if not files:
                continue
            class_index = len(class_names)
            class_features = []
            for image_path in tqdm(files, desc=f"Extract {class_dir.name}"):
                image = cv2.imread(str(image_path))
                if image is None:
                    skipped.append(str(image_path))
                    continue
                feature, _ = extractor.extract(image)
                if feature is None:
                    skipped.append(str(image_path))
                    continue
                class_features.append(feature)
            if class_features:
                class_names.append(class_dir.name)
                features.extend(class_features)
                labels.extend([class_index] * len(class_features))
    finally:
        extractor.close()

    if not features or len(class_names) < 2:
        raise RuntimeError("Not enough usable samples. Need at least two class folders containing images with visible hands.")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, X=np.asarray(features, dtype=np.float32),
                        y=np.asarray(labels, dtype=np.int64),
                        class_names=np.asarray(class_names, dtype=str))
    print(f"Saved {len(features)} samples, {len(class_names)} classes -> {output}")
    print(f"Skipped {len(skipped)} images (unreadable or no hand detected).")
    if skipped:
        print("Tip: ensure hands are visible and reasonably large in the images.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=Path, help='Root folder with one subfolder per class')
    parser.add_argument('--output', type=Path, default=DEFAULT_CACHE_PATH)
    args = parser.parse_args()
    build(args.dataset, args.output)
