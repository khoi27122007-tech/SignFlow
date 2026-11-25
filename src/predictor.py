
"""Load and run inference using a trained landmark classifier."""

from pathlib import Path

import numpy as np
import torch

try:
    from .model import LandmarkClassifier
    from .config import DEFAULT_MODEL_PATH
except ImportError:
    from model import LandmarkClassifier
    from config import DEFAULT_MODEL_PATH


class GesturePredictor:

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        confidence_threshold: float = 0.65,
    ):

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path.resolve()}"
            )

        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0 and 1"
            )

        self.confidence_threshold = confidence_threshold

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False,
        )

        required_keys = {
            "model_state_dict",
            "class_names",
            "input_size",
            "num_classes",
        }

        missing = required_keys - set(checkpoint.keys())

        if missing:
            raise ValueError(
                f"Invalid checkpoint. Missing keys: {missing}"
            )

        self.class_names = [
            str(name)
            for name in checkpoint["class_names"]
        ]

        self.input_size = int(checkpoint["input_size"])

        self.num_classes = int(checkpoint["num_classes"])

        if len(self.class_names) != self.num_classes:
            raise ValueError(
                "Checkpoint class_names length does not match "
                "num_classes."
            )

        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError(
                "Checkpoint contains duplicate class names."
            )

        expected_classes = [chr(code) for code in range(ord("A"), ord("Z") + 1)] + ["del", "space"]
        if self.class_names != expected_classes:
            missing_classes = [name for name in expected_classes if name not in self.class_names]
            extra_classes = [name for name in self.class_names if name not in expected_classes]
            raise ValueError(
                "This checkpoint is not the complete SignFlow 28-class landmark model. "
                f"Missing classes: {missing_classes}; unexpected classes: {extra_classes}. "
                "Rebuild landmarks.npz and retrain landmark_model.pth before starting webcam."
            )

        self.model = LandmarkClassifier(
            input_size=self.input_size,
            num_classes=self.num_classes,
        ).to(self.device)

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        self.model.eval()

        print("=" * 55)
        print("SIGNFLOW LANDMARK PREDICTOR")
        print("=" * 55)
        print("Model:", self.model_path.resolve())
        print("Device:", self.device)
        print("Number of classes:", self.num_classes)
        print("Classes:", self.class_names)
        print("Confidence threshold:", self.confidence_threshold)

    @torch.inference_mode()
    def predict(self, features: np.ndarray):

        features = np.asarray(
            features,
            dtype=np.float32,
        ).reshape(-1)

        if features.shape[0] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} features, "
                f"received {features.shape[0]}"
            )

        if not np.isfinite(features).all():
            return "Unknown", 0.0

        x = torch.as_tensor(
            features,
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, -1)

        logits = self.model(x)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

        confidence, index = torch.max(
            probabilities,
            dim=0,
        )

        score = float(confidence.item())

        predicted_index = int(index.item())

        if score < self.confidence_threshold:
            return "Unknown", score

        label = self.class_names[predicted_index]

        return label, score
