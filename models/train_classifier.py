"""Train the offline intent classifier.

Run once: python models/train_classifier.py
Produces: classifier.pkl, vectorizer.pkl
"""
import json
import os
import time
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.json")
MODEL_DIR = os.path.join(BASE_DIR, "models")


def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [d["text"] for d in data]
    labels = [d["label"] for d in data]
    return texts, labels


def train():
    texts, labels = load_data()
    print(f"[INFO] Loaded {len(texts)} samples across {len(set(labels))} classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, C=2.0, class_weight="balanced")
    clf.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Test accuracy: {acc:.3f}")
    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(y_test, y_pred))

    # Benchmark latency
    sample = "Remind me to call mom tomorrow"
    start = time.time()
    for _ in range(100):
        _ = clf.predict(vectorizer.transform([sample]))
    avg_ms = (time.time() - start) / 100 * 1000
    print(f"[BENCHMARK] Avg inference latency: {avg_ms:.2f} ms/message (CPU)")

    # Save
    joblib.dump(clf, os.path.join(MODEL_DIR, "classifier.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))

    clf_size = os.path.getsize(os.path.join(MODEL_DIR, "classifier.pkl")) / 1024
    vec_size = os.path.getsize(os.path.join(MODEL_DIR, "vectorizer.pkl")) / 1024
    print(f"[SIZE] classifier.pkl: {clf_size:.1f} KB | vectorizer.pkl: {vec_size:.1f} KB")
    print(f"[SIZE] Total: {(clf_size + vec_size)/1024:.2f} MB (limit: 50 MB)")


if __name__ == "__main__":
    train()
