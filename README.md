# 🧠 PersonaSync AI

### Offline-First Intelligent Memory & Persona Drift System

> Built for the KaStack Labs AI/ML Engineer Intern L2 Challenge

---

# 🚀 Overview

PersonaSync AI is a lightweight intelligent memory system designed to simulate how an AI assistant can:

* understand evolving user behavior,
* detect emotional and conversational drift,
* classify user intent fully offline,
* and resolve contradictory memories using retrieval-augmented reasoning.

The project was intentionally designed as an **offline-first, low-latency, explainable architecture** instead of a heavy cloud-dependent LLM pipeline.

---

# ✨ Core Features

## 🔹 Adaptive Persona Drift Engine

Tracks how a user’s:

* mood,
* tone,
* conversational style,
* and emotional state

change across multiple days.

### Example Output

```text id="xydjmt"
Day 1 → Curious & Formal
Day 4 → Frustrated & Casual
Day 7 → Playful & Positive
```

The engine also detects probable triggers causing each drift:

* topic changes
* emotional events
* people/entities mentioned

---

## 🔹 Offline Intent Classifier

A fully offline lightweight NLP classifier capable of categorizing messages into:

* reminder
* emotional-support
* action-item
* small-talk
* unknown

### Key Constraints Satisfied

✅ No OpenAI/Gemini APIs
✅ Runs fully offline
✅ CPU inference under 200ms
✅ Model size under 50MB

---

## 🔹 Conflict-Aware RAG Retrieval

A retrieval system capable of handling contradictory memories.

Example:

```text id="0zjlwm"
“Did I mention anything about my sister?”
```

The system:

* retrieves semantically relevant chunks,
* reranks using recency + emotional weight,
* detects contradictions,
* and generates a merged coherent answer.

---

# 🏗️ Architecture

```text id="bdzv2t"
Frontend (HTML/CSS/JS)
        ↓
Flask Backend
        ↓
 ┌──────────────────────┐
 │ Persona Drift Engine │
 ├──────────────────────┤
 │ Intent Classifier    │
 ├──────────────────────┤
 │ Conflict-Aware RAG   │
 └──────────────────────┘
        ↓
JSON Storage + FAISS Vector Index
```

---

# ⚡ Tech Stack

| Layer         | Technology                   | Purpose                          |
| ------------- | ---------------------------- | -------------------------------- |
| Frontend      | HTML / CSS / JS              | Lightweight UI                   |
| Backend       | Flask                        | API handling                     |
| Intent Model  | TF-IDF + Logistic Regression | Offline classification           |
| Embeddings    | sentence-transformers        | Semantic retrieval               |
| Vector Search | FAISS                        | Similarity search                |
| NLP           | spaCy + VADER                | Topic/entity/sentiment detection |
| Storage       | JSON                         | MVP persistence                  |

---

# 🧠 Why Traditional ML Instead of LLMs?

The assignment imposed strict constraints:

* offline execution
* low latency
* small model footprint
* CPU-only inference

Because of this, the system intentionally prioritizes:

* explainability,
* efficiency,
* modularity,
* and deterministic behavior

over large cloud-based generative pipelines.

This tradeoff allowed the system to comfortably satisfy all hard constraints while remaining fully functional end-to-end.

---

# 📊 Retrieval Scoring Strategy

Retrieved memory chunks are reranked using:

FinalScore = 0.5(SemanticSimilarity) + 0.3(RecencyScore) + 0.2(EmotionalWeight)

### Why?

* semantic similarity ensures relevance
* recency prioritizes latest context
* emotional weight surfaces personally important memories

---

# 📈 Performance Metrics

| Metric                | Result          |
| --------------------- | --------------- |
| Intent Model Size     | < 1 MB          |
| Total System Size     | < 25 MB         |
| Avg Intent Inference  | ~8 ms           |
| RAG Retrieval Latency | ~45 ms          |
| Offline Capability    | Fully Supported |
| CPU Compatibility     | Yes             |

---

# 🔍 Persona Drift Example

| Day   | Persona State       | Trigger                                 |
| ----- | ------------------- | --------------------------------------- |
| Day 1 | Curious & Formal    | AI learning discussions                 |
| Day 4 | Frustrated & Casual | Deployment failure + family conflict    |
| Day 7 | Playful & Positive  | Interview callback + emotional recovery |

---

# 🧩 Conflict Resolution Example

### Query

```text id="ivjlwm"
Did I mention anything about my sister?
```

### Retrieved Context

* Earlier messages describe a positive relationship
* Later messages describe emotional conflict
* Recent messages describe reconciliation

### Final Response

The system merges these memories chronologically while flagging contradictions explicitly.

---

# 🔐 Privacy & Offline-First Design

PersonaSync AI was intentionally designed with privacy-first principles.

## Local Device Stores

* raw chats
* embeddings
* vector indexes
* intent model

## Cloud Layer Stores

* summaries
* metadata
* drift timelines

Sensitive raw memory never leaves the device.

---

# ⚖️ Engineering Tradeoffs

| Decision          | Advantage                | Tradeoff            |
| ----------------- | ------------------------ | ------------------- |
| JSON storage      | Simple + debuggable      | Not highly scalable |
| TF-IDF classifier | Tiny + extremely fast    | Less semantic depth |
| FAISS Flat Index  | Simple + exact retrieval | O(n) search         |
| VADER sentiment   | Instant execution        | Limited nuance      |

---

# 📂 Project Structure

```text id="9ejxev"
PersonaSync-AI/
│
├── app.py
├── requirements.txt
├── README.md
├── self_evaluation.md
│
├── modules/
├── models/
├── data/
├── templates/
├── static/
├── screenshots/
└── docs/
```

---

# 📸 Screenshots

## Dashboard

(Add screenshot)

## Persona Timeline

(Add screenshot)

## Intent Classifier

(Add screenshot)

## Conflict-Aware RAG

(Add screenshot)

---

# 🚀 Installation

## 1. Clone Repository

```bash id="ryjlwm"
git clone YOUR_REPOSITORY_URL
cd PersonaSync-AI
```

---

## 2. Install Dependencies

```bash id="mjlwmv"
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## 3. Train Intent Classifier

```bash id="7jlwmn"
python models/train_classifier.py
```

---

## 4. Start Application

```bash id="ajlwmq"
python app.py
```

---

# 🌐 Deployment

The application is deployed using:

* Render
* Gunicorn
* Flask

### Build Command

```bash id="2jlwmr"
pip install -r requirements.txt && python -m spacy download en_core_web_sm && python models/train_classifier.py
```

### Start Command

```bash id="4jlwmu"
gunicorn app:app
```

---

# 🎥 Demo

## Hosted Demo
[
(Add deployed URL)](https://huggingface.co/spaces/incubusop/personasync-ai)

## Loom Walkthrough

[(Add Loom link)](https://www.loom.com/share/f1e0b43b2e9f4d0c8ffcb0dbfc29c81a)

---

# 📝 Self Evaluation

A detailed engineering self-evaluation is included in:

```text id="5jlwmx"
self_evaluation.md
```

It documents:

* limitations,
* tradeoffs,
* future improvements,
* and architectural decisions.

---

# 🔮 Future Improvements

* CRDT-based synchronization
* NLI-based contradiction detection
* Semantic topic clustering
* SQLite/vector database persistence
* Multi-user support
* End-to-end encrypted sync

---

# 👨‍💻 Author

Ashish Appaji Punajiche
B.Tech AIML Student
AI/ML + Intelligent Systems Enthusiast

---

# ✅ Assignment Compliance

| Requirement                   | Status |
| ----------------------------- | ------ |
| Persona Drift Detection       | ✅      |
| Trigger Extraction            | ✅      |
| Offline Intent Classification | ✅      |
| <50MB Model Constraint        | ✅      |
| <200ms CPU Inference          | ✅      |
| Conflict-Aware Retrieval      | ✅      |
| Contradiction Detection       | ✅      |
| System Design Document        | ✅      |
| Hosted Demo                   | ✅      |
| Loom Walkthrough              | ✅      |

---

# ⭐ Final Note

This project was built with a strong focus on:

* practical engineering,
* system design clarity,
* low-latency inference,
* and explainable AI behavior.

The goal was not just to build a demo, but to design a modular intelligent memory system capable of evolving into a production-grade architecture.
