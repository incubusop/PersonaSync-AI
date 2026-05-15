"""Adaptive Persona Engine: daily mood/tone aggregation + drift detection + trigger extraction."""
import re
from collections import defaultdict, Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    import spacy
    NLP = spacy.load("en_core_web_sm")
    HAS_SPACY = True
except Exception:
    NLP = None
    HAS_SPACY = False

VADER = SentimentIntensityAnalyzer()

CASUAL_MARKERS = {
    "haha", "lol", "lmao", "ugh", "omg", "bro", "yeah", "yep", "nah",
    "gonna", "wanna", "kinda", "dunno", "u", "ur", "thx", "ya", "okay",
}
FORMAL_MARKERS = {
    "however", "moreover", "therefore", "regarding", "additionally",
    "furthermore", "could", "would", "shall", "indeed",
}

NEGATIVE_TRIGGERS = {"fail", "failed", "broke", "argument", "fight", "fought",
                     "issue", "problem", "crashed", "missed", "rejected", "loss"}
POSITIVE_TRIGGERS = {"got", "achieved", "callback", "selected", "won", "made up",
                     "celebrate", "promotion", "offer", "passed"}


def detect_tone(text):
    t = text.lower()
    casual = sum(1 for w in CASUAL_MARKERS if w in t.split())
    formal = sum(1 for w in FORMAL_MARKERS if w in t.split())
    contractions = len(re.findall(r"\b\w+'(s|t|re|ve|ll|d|m)\b", t))
    if casual + contractions > formal:
        return "casual"
    if formal > casual:
        return "formal"
    return "neutral"


def detect_mood(text):
    scores = VADER.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.5:
        return "playful" if any(w in text.lower() for w in ["haha", "lol", "yay", "!"]) else "happy"
    if compound >= 0.15:
        return "curious" if "?" in text else "positive"
    if compound <= -0.5:
        return "frustrated"
    if compound <= -0.15:
        return "down"
    return "neutral"


def extract_entities_and_topics(text):
    entities = []
    topics = []
    if HAS_SPACY:
        doc = NLP(text)
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "ORG", "GPE", "EVENT", "PRODUCT"}:
                entities.append(ent.text)
        # noun chunks as topics
        topics = [chunk.text.lower() for chunk in doc.noun_chunks
                  if len(chunk.text.split()) <= 3 and chunk.text.lower() not in {"i", "you", "we"}]
    return entities, topics


def aggregate_day(messages):
    """Aggregate a list of message dicts for one day -> persona state."""
    if not messages:
        return None

    moods = [detect_mood(m["text"]) for m in messages]
    tones = [detect_tone(m["text"]) for m in messages]

    sentiment_scores = [VADER.polarity_scores(m["text"])["compound"] for m in messages]
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)

    all_entities = []
    all_topics = []
    for m in messages:
        ents, tops = extract_entities_and_topics(m["text"])
        all_entities.extend(ents)
        all_topics.extend(tops)

    dominant_mood = Counter(moods).most_common(1)[0][0]
    dominant_tone = Counter(tones).most_common(1)[0][0]

    return {
        "day": messages[0]["day"],
        "timestamp": messages[0].get("timestamp"),
        "mood": dominant_mood,
        "tone": dominant_tone,
        "sentiment_score": round(avg_sentiment, 3),
        "entities": list(set(all_entities)),
        "topics": list(set(all_topics))[:5],
        "message_count": len(messages),
    }


def find_trigger(prev_state, curr_state, curr_messages):
    """Identify what likely caused the drift between prev and curr day."""
    if not prev_state:
        return None

    # New entities introduced today
    new_entities = set(curr_state["entities"]) - set(prev_state["entities"])
    new_topics = set(curr_state["topics"]) - set(prev_state["topics"])

    # Look for trigger keywords in messages of the drift day
    trigger_phrases = []
    for m in curr_messages:
        text_lower = m["text"].lower()
        for w in NEGATIVE_TRIGGERS | POSITIVE_TRIGGERS:
            if w in text_lower:
                # Capture surrounding context (3 words on each side)
                match = re.search(rf"(\b\w+\b\s+){{0,3}}{w}(\s+\b\w+\b){{0,3}}", text_lower)
                if match:
                    trigger_phrases.append(match.group().strip())

    trigger = {
        "new_people_or_places": list(new_entities)[:3] if new_entities else [],
        "new_topics": list(new_topics)[:3] if new_topics else [],
        "key_phrases": trigger_phrases[:3] if trigger_phrases else [],
    }

    # Build a short summary string
    parts = []
    if trigger_phrases:
        parts.append(f"mentioned '{trigger_phrases[0]}'")
    if new_entities:
        parts.append(f"new entity: {list(new_entities)[0]}")
    trigger["summary"] = "; ".join(parts) if parts else "no clear trigger identified"

    return trigger


def detect_drift(prev_state, curr_state):
    """Decide if there was meaningful drift between two days."""
    if not prev_state:
        return False
    sentiment_delta = abs(curr_state["sentiment_score"] - prev_state["sentiment_score"])
    mood_changed = curr_state["mood"] != prev_state["mood"]
    tone_changed = curr_state["tone"] != prev_state["tone"]
    return sentiment_delta > 0.3 or mood_changed or tone_changed


def build_persona_timeline(chats):
    """Main entry: takes list of chat messages, returns persona timeline with drift triggers."""
    # group by day
    by_day = defaultdict(list)
    for c in chats:
        by_day[c["day"]].append(c)

    sorted_days = sorted(by_day.keys())
    timeline = []
    prev_state = None

    for day in sorted_days:
        msgs = by_day[day]
        state = aggregate_day(msgs)
        if not state:
            continue

        drift = detect_drift(prev_state, state)
        trigger = find_trigger(prev_state, state, msgs) if drift else None

        entry = {
            **state,
            "drift_detected": drift,
            "trigger": trigger,
            "label": f"Day {day} → {state['mood'].capitalize()} & {state['tone'].capitalize()}",
        }
        timeline.append(entry)
        prev_state = state

    return timeline
