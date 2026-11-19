from collections import deque, Counter
from dataclasses import dataclass


@dataclass
class StablePrediction:
    label: str
    confidence: float
    stable: bool


class TemporalSmoother:
    """Require repeated confident predictions before emitting a stable label."""
    def __init__(self, window=9, min_votes=6, min_confidence=0.78,
                 switch_votes=5):
        self.window = deque(maxlen=window)
        self.min_votes = min_votes
        self.min_confidence = min_confidence
        self.switch_votes = switch_votes
        self.current = None

    def reset(self):
        self.window.clear()
        self.current = None

    def update(self, label, confidence):
        if label is None or confidence < self.min_confidence:
            self.window.append(None)
            return StablePrediction("UNKNOWN", float(confidence or 0), False)

        self.window.append((str(label), float(confidence)))
        valid = [x for x in self.window if x is not None]
        if not valid:
            return StablePrediction("UNKNOWN", 0.0, False)

        counts = Counter(x[0] for x in valid)
        candidate, votes = counts.most_common(1)[0]
        avg_conf = sum(c for l, c in valid if l == candidate) / votes

        required = self.min_votes
        if self.current is not None and candidate != self.current:
            required = self.switch_votes

        if votes >= required:
            self.current = candidate
            return StablePrediction(candidate, avg_conf, True)

        return StablePrediction("...", avg_conf, False)
