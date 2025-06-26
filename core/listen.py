import sounddevice as sd
import queue
import vosk
import json

q = queue.Queue()

model = vosk.Model("models/vosk_model")
samplerate = 16000

def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))

def listen():
    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        rec = vosk.KaldiRecognizer(model, samplerate)
        print("Listening... (say something)")
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                return result.get("text", "")
