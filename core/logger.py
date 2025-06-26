import datetime
import os

def log(text):
    date = datetime.date.today().isoformat()
    filename = os.path.join("logs", f"{date}.log")
    with open(filename, "a") as f:
        f.write(f"{datetime.datetime.now()} - {text}\n")
