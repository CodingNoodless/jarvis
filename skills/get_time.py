from datetime import datetime

def run(intent, context, memory):
    now = datetime.now().strftime("%I:%M %p")
    return f"The current time is {now}."