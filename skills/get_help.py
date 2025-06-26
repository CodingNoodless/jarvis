import os

def run(intent, context, memory):
    files = [f[:-3] for f in os.listdir("skills") if f.endswith(".py")]
    return "Available skills: " + ", ".join(files)
