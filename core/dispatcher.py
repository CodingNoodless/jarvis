from skills import time, weather, fallback

def dispatch(intent_obj):
    intent = intent_obj["intent"]
    
    if intent == "get_time":
        return time.run()
    elif intent == "get_weather":
        return weather.run(intent_obj["entities"])
    else:
        return fallback.run(intent_obj["text"])
