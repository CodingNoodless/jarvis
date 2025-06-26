from datetime import datetime

def run():
    now = datetime.now().strftime("%I:%M %p")
    return f"The current time is {now}."
