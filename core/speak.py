import pyttsx3

engine = pyttsx3.init()
engine.setProperty("rate", 175)

def speak(text):
    print("Jarvis (speaking):", text)
    engine.say(text)
    engine.runAndWait()
