# Hand Gesture Recognition using 21 landmarks

This `src` pipeline is independent of the existing notebook. It does not modify notebook files or reuse a CNN image checkpoint. It extracts 21 MediaPipe hand landmarks from each labeled image, normalizes them, and trains a classifier whose output classes are discovered from the dataset folders.

## Dataset layout

```text
dataset/
  A/
    001.jpg
    002.jpg
  B/
    001.jpg
  C/
    ...
```

Each immediate child folder is one label. Keep your original dataset where it is; pass its path to the command.

## Install

From the project root:

```bash
pip install -r src/requirements.txt
```

## 1. Extract 21 points from all dataset images

```bash
python -m src.build_dataset --dataset "path/to/dataset" --output "data/landmarks.npz"
```

Images where a hand cannot be detected are skipped. Make sure each class has enough clear hand images and balanced examples.

## 2. Train

```bash
python -m src.train --data "data/landmarks.npz" --output "models/landmark_model.pth" --epochs 60
```

The number and names of classes are read automatically from the dataset folders.

## 3. Run webcam

```bash
python -m src.webcam --model "models/landmark_model.pth"
```

Press `Q` or `Esc` to exit. Adjust `--threshold` if predictions are too permissive, for example `--threshold 0.75`.

## Important limitation: static letters vs dynamic gestures

This pipeline predicts the hand pose on every webcam frame. Therefore it can keep recognizing a static letter while the hand is moving between positions. It is not yet a temporal sequence classifier. If a letter is defined by a movement path (rather than a pose), add labeled video clips or landmark sequences and train a temporal model (for example an LSTM/GRU/1D temporal CNN). Still images cannot teach the direction, speed, or order of a movement.

## Notebook compatibility

The notebook is intentionally untouched. This source pipeline trains a separate landmark model (`landmark_model.pth`); it is not compatible with a CNN checkpoint trained from raw images in the notebook. The notebook can remain as-is, but its model and this landmark model are separate pipelines.
