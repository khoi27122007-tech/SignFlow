from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "landmark_model.pth"
DEFAULT_CACHE_PATH = PROJECT_ROOT / "data" / "landmarks.npz"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LANDMARK_COUNT = 21
FEATURE_SIZE = LANDMARK_COUNT * 3
