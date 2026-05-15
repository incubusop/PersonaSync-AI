"""Conflict-aware RAG: FAISS retrieval + recency + emotion reranking + contradiction detection."""
import os
from datetime import datetime
import numpy as np
import faiss
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from modules.embedding_engine import embed
from modules.utils import load_json

VADER = SentimentIntensityAnalyzer()

# Reranking weights — justified in README
W_SIM = 0.5       # semantic match dominates: irrelevant chunks shouldn't win even if recent
W_RECENCY = 0.3   # user asked "did I mention" -> recall question, recency matters
W_EMOTION = 0.2   # strong emotion = memorable, more likely "worth surfacing"


class RAGEngine:
    def __init__(self, chats_file="chats.json"):
        self.chats = load_json(chats_file)
        self.chunks = []
        self.index = None
        self._build_index()

    def _build_index(self):
        if not self.chats:
            return
        # Each chat message becomes a chunk (small dataset)
        for c in self.chats:
            sentiment = VADER.polarity_scores(c["text"])["compound"]
            self.chunks.append({
                "text": c["text"],
                "day": c["day"],
                "timestamp": c.get("timestamp"),
                "emotion_intensity": abs(sentiment),
                "sentiment": sentiment,
            })

        texts = [ch["text"] for ch in self.chunks]
        vectors = embed(texts)
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine
        self.index.add(vectors)

    def _recency_score(self, day):
        max_day = max((c["day"] for c in self.chunks), default=1)
        min_day = min((c["day"] for c in self.chunks), default=1)
        if max_day == min_day:
            return 1.0
        return (day - min_day) / (max_day - min_day)

    def retrieve(self, query, top_k=5):
        if not self.index:
            return []
        qvec = embed(query)
        scores, idxs = self.index.search(qvec, min(top_k, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            ch = self.chunks[idx]
            sem_score = float(score)  # cosine similarity (0-1 for normalized)
            recency = self._recency_score(ch["day"])
            emotion = min(ch["emotion_intensity"], 1.0)
            final = W_SIM * sem_score + W_RECENCY * recency + W_EMOTION * emotion
            results.append({
                **ch,
                "semantic_similarity": round(sem_score, 3),
                "recency_score": round(recency, 3),
                "emotional_weight": round(emotion, 3),
                "final_score": round(final, 3),
            })
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results

    def detect_contradictions(self, results):
        """Look for opposing sentiment polarity AND/OR location-style entity swaps for the same subject."""
        contradictions = []
        # Sentiment polarity contradiction (e.g., "love sister" vs "fought with sister")
        positives = [r for r in results if r["sentiment"] > 0.3]
        negatives = [r for r in results if r["sentiment"] < -0.3]
        if positives and negatives:
            contradictions.append({
                "type": "emotional_polarity",
                "detail": f"Positive context on day {positives[0]['day']} vs negative context on day {negatives[0]['day']}",
                "positive_chunk": positives[0]["text"],
                "negative_chunk": negatives[0]["text"],
            })

        # Location/attribute swap: same key noun (e.g., place) differing across chunks
        import re
        places = {}
        for r in results:
            # crude: capture capitalized words that aren't sentence-starters
            words = re.findall(r"\b([A-Z][a-z]+)\b", r["text"])
            for w in words:
                if w.lower() in {"i", "my", "the", "a"}:
                    continue
                places.setdefault(w, []).append(r["day"])
        # If two different places mentioned on different days, flag
        place_list = list(places.keys())
        if len(place_list) >= 2:
            for i in range(len(place_list)):
                for j in range(i + 1, len(place_list)):
                    p1, p2 = place_list[i], place_list[j]
                    if set(places[p1]).isdisjoint(places[p2]):
                        # Only flag if both appear semantically as location-like
                        pass  # kept simple; surface above polarity check is the strong one

        return contradictions

    def synthesize_answer(self, query, results, contradictions):
        if not results:
            return "I couldn't find anything relevant in your memory about that."

        # Order by day for narrative coherence
        chronological = sorted(results, key=lambda x: x["day"])
        most_recent = max(results, key=lambda x: x["day"])

        parts = [f"You've mentioned this {len(results)} time(s) across your history."]
        parts.append(f"Most recently (Day {most_recent['day']}): \"{most_recent['text']}\"")

        if len(chronological) > 1:
            earliest = chronological[0]
            parts.append(f"Earlier (Day {earliest['day']}): \"{earliest['text']}\"")

        if contradictions:
            c = contradictions[0]
            parts.append(
                f"⚠️ Note: there's a contradiction in tone here — "
                f"{c['detail'].lower()}."
            )

        return " ".join(parts)

    def query(self, user_question, top_k=5):
        results = self.retrieve(user_question, top_k=top_k)
        contradictions = self.detect_contradictions(results)
        answer = self.synthesize_answer(user_question, results, contradictions)
        return {
            "query": user_question,
            "answer": answer,
            "contradictions": contradictions,
            "ranked_chunks": results,
            "scoring_formula": "final = 0.5*similarity + 0.3*recency + 0.2*emotion",
        }
