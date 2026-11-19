from __future__ import annotations
import numpy as np


def fuse_probabilities(cnn_probs=None, landmark_probs=None,
                       cnn_weight=0.45, landmark_weight=0.55):
    """
    Weighted probability fusion. Both arrays must use identical class order.
    If one branch is unavailable, the other branch is returned unchanged.
    """
    if cnn_probs is None and landmark_probs is None:
        return None
    if cnn_probs is None:
        return np.asarray(landmark_probs, dtype=np.float32)
    if landmark_probs is None:
        return np.asarray(cnn_probs, dtype=np.float32)

    a = np.asarray(cnn_probs, dtype=np.float32)
    b = np.asarray(landmark_probs, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"Probability shape mismatch: CNN {a.shape}, landmark {b.shape}")
    total = max(cnn_weight + landmark_weight, 1e-8)
    out = (cnn_weight * a + landmark_weight * b) / total
    return out / max(float(out.sum()), 1e-8)
