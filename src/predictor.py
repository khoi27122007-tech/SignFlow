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
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, confidence_threshold: float = 0.60):
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}. Train it first.")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.class_names = checkpoint['class_names']
        self.model = LandmarkClassifier(checkpoint['input_size'], checkpoint['num_classes']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.confidence_threshold = confidence_threshold

    @torch.inference_mode()
    def predict(self, features: np.ndarray):
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device).reshape(1, -1)
        probabilities = torch.softmax(self.model(x), dim=1)[0]
        confidence, index = torch.max(probabilities, dim=0)
        score = float(confidence.item())
        label = self.class_names[int(index.item())] if score >= self.confidence_threshold else 'Unknown'
        return label, score
