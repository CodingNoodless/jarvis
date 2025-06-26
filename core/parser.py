def parse(text):
    text = text.lower()
    if "time" in text:
        return {"intent": "get_time", "entities": {}}
    elif "weather" in text:
        return {"intent": "get_weather", "entities": {"location": "San Jose"}}
    else:
        return {"intent": "unknown", "text": text}
