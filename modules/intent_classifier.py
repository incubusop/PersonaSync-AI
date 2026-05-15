"""Offline intent classifier - <50MB, <200ms CPU."""
import os
import time
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

CONFIDENCE_THRESHOLD = 0.35  # Below this -> "unknown"


class IntentClassifier:
    def __init__(self):
        self.clf = joblib.load(os.path.join(MODEL_DIR, "classifier.pkl"))
        self.vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
        self.labels = list(self.clf.classes_)

    def predict(self, text):
        start = time.time()
        vec = self.vectorizer.transform([text])
        probs = self.clf.predict_proba(vec)[0]
        best_idx = probs.argmax()
        best_label = self.labels[best_idx]
        confidence = float(probs[best_idx])

        # Fallback to "unknown" if confidence is too low
        if confidence < CONFIDENCE_THRESHOLD and best_label != "unknown":
            final_label = "unknown"
        else:
            final_label = best_label

        latency_ms = (time.time() - start) * 1000
        return {
            "intent": final_label,
            "confidence": round(confidence, 3),
            "latency_ms": round(latency_ms, 2),
            "all_scores": {lbl: round(float(p), 3) for lbl, p in zip(self.labels, probs)},
        }
