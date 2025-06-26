def run(intent, context, memory):
    location = intent["entities"].get("location", "your area")
    return f"The weather in {location} is sunny and 75°F."  # Placeholder
