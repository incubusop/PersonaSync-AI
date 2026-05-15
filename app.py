"""KaStack AI Memory System - Flask backend."""
import os
import time
from flask import Flask, render_template, request, jsonify

from modules.intent_classifier import IntentClassifier
from modules.persona_engine import build_persona_timeline
from modules.rag_engine import RAGEngine
from modules.utils import load_json

app = Flask(__name__)

# Lazy globals — loaded on first request
_intent_clf = None
_rag = None


def get_intent_classifier():
    global _intent_clf
    if _intent_clf is None:
        _intent_clf = IntentClassifier()
    return _intent_clf


def get_rag():
    global _rag
    if _rag is None:
        _rag = RAGEngine()
    return _rag


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/intent", methods=["POST"])
def api_intent():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "empty text"}), 400
    clf = get_intent_classifier()
    result = clf.predict(text)
    return jsonify(result)


@app.route("/api/persona", methods=["GET"])
def api_persona():
    chats = load_json("chats.json")
    timeline = build_persona_timeline(chats)
    return jsonify({"timeline": timeline, "total_days": len(timeline)})


@app.route("/api/rag", methods=["POST"])
def api_rag():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "empty query"}), 400
    rag = get_rag()
    start = time.time()
    result = rag.query(query)
    result["total_latency_ms"] = round((time.time() - start) * 1000, 2)
    return jsonify(result)


@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
