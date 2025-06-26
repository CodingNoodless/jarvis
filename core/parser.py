from sentence_transformers import SentenceTransformer, util
from core.intents import INTENT_REGISTRY
import re

{
  "intent": "intent_name",
  "entities": { "key": "value" },
  "confidence": 0.9,
  "raw": "user's original text"
}


model = SentenceTransformer("all-MiniLM-L6-v2")

def parse(text):
    user_vec = model.encode(text, convert_to_tensor=True)
    best_intent = None
    best_score = 0.0

    for intent, data in INTENT_REGISTRY.items():
        for phrase in data["examples"]:
            example_vec = model.encode(phrase, convert_to_tensor=True)
            sim = util.pytorch_cos_sim(user_vec, example_vec).item()
            if sim > best_score:
                best_score = sim
                best_intent = intent
    print(f"Best intent: {best_intent} with score: {best_score}")

    if best_score > 0.55:
        return {
            "intent": best_intent,
            "entities": extract_entities(text, best_intent),
            "confidence": best_score,
            "raw": text
        }
    else:
        return {
            "intent": "unknown",
            "entities": {},
            "confidence": best_score,
            "raw": text
        }


def extract_entities(text, intent):
    entities = {}
    if intent == "get_weather":
        match = re.search(r"in ([A-Za-z ]+)", text)
        if match:
            entities["location"] = match.group(1).strip()
    return entities
