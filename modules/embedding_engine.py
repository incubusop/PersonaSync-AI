"""Embedding generation using sentence-transformers all-MiniLM-L6-v2."""
import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def embed(texts):
    """texts: str or list[str] -> np.ndarray (n, 384)."""
    if isinstance(texts, str):
        texts = [texts]
    model = get_model()
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype("float32")
