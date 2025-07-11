import sounddevice as sd
import numpy as np
import requests
import warnings
from faster_whisper import WhisperModel
import pvporcupine
import pyaudio
import struct
import soundfile as sf
from io import BytesIO
from threading import Thread, Event
from TTS.api import TTS as CoquiTTS
import webrtcvad
import collections
import queue
import time
import json

warnings.filterwarnings("ignore", category=UserWarning)
SAMPLERATE = 16000

print("🔊 Loading Whisper model...")
whisper_model = WhisperModel("tiny", compute_type="int8")

print("🔊 Loading British TTS model...")
# Use VCTK model for authentic British accents
vits_tts = CoquiTTS(model_name="tts_models/en/vctk/vits", gpu=True)
print("⏳ Warming up TTS...")
# Use p230 - British male, fastest synthesis speed with British accent
_ = vits_tts.tts("Hello, Jarvis is ready.", speaker="p230")
VITS_SR = 22050

print("🎧 Initializing hotword detector...")
porcupine = pvporcupine.create(access_key="BMthPl0em5470FOwow1fy6/z50cW1UKMarJqcTV7Pvb32BZVxkR1Tw==", keywords=["jarvis"])
pa = pyaudio.PyAudio()
audio_stream = pa.open(rate=porcupine.sample_rate, channels=1, format=pyaudio.paInt16,
                       input=True, frames_per_buffer=porcupine.frame_length)

audio_queue = queue.Queue()
transcription_queue = queue.Queue()

# This queue holds text chunks ready for TTS synthesis
tts_text_queue = queue.Queue()

# This queue holds synthesized audio chunks ready for playback
tts_audio_queue = queue.Queue()

stop_event = Event()

def record_audio_vad(max_duration=4, samplerate=SAMPLERATE):
    # Optimized VAD for faster speech detection and shorter recording times
    vad = webrtcvad.Vad(1)  # Less aggressive VAD mode
    frame_duration = 30
    frame_size = int(samplerate * frame_duration / 1000)
    ring = collections.deque(maxlen=6)  # Reduced ring buffer size
    q = queue.Queue()
    recording = []
    triggered = False
    silence_count = 0
    min_speech_frames = 3  # Minimum frames to consider speech started

    def callback(indata, frames, time_info, status):
        q.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32',
                        blocksize=frame_size, callback=callback):
        start = time.time()
        while True:
            if time.time() - start > max_duration:
                break
            try:
                chunk = q.get(timeout=0.1)
            except queue.Empty:
                continue
            pcm = (chunk * 32767).astype(np.int16).tobytes()
            speech = vad.is_speech(pcm, samplerate)
            
            if not triggered:
                ring.append((chunk, speech))
                speech_frames = sum(1 for c, s in ring if s)
                # Start recording with fewer speech frames detected
                if speech_frames >= min_speech_frames:
                    triggered = True
                    for c, _ in ring:
                        recording.append(c)
                    ring.clear()
                    silence_count = 0
            else:
                recording.append(chunk)
                if speech:
                    silence_count = 0
                else:
                    silence_count += 1
                    # Stop recording after shorter silence period
                    if silence_count >= 4:  # Reduced from 9 to 4 frames
                        break
    
    if not recording:
        return np.zeros(frame_size, dtype=np.float32)
    return np.concatenate(recording)

def transcription_worker():
    while not stop_event.is_set():
        try:
            audio_np = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        start = time.time()
        buf = BytesIO()
        sf.write(buf, audio_np, SAMPLERATE, format='WAV')
        buf.seek(0)
        segments, _ = whisper_model.transcribe(buf)
        text = " ".join(seg.text for seg in segments)
        print(f"⏱️ Transcription took {time.time() - start:.2f}s")
        print("You said:", text)
        transcription_queue.put(text)
        audio_queue.task_done()

def llm_worker():
    """
    This worker streams LLM output, buffers partial chunks,
    and pushes well-formed text chunks to tts_text_queue.
    """
    sentence_end_chars = {'.', '!', '?'}
    buffer = ""
    chunk_sequence = 0

    def find_smart_chunk_boundary(text, min_length=25):
        """Find intelligent chunk boundaries that preserve word integrity"""
        if len(text) <= min_length:
            # Only return if it's a complete sentence
            return text.strip() if any(text.strip().endswith(c) for c in sentence_end_chars) else None
        
        # Priority 1: Look for sentence endings
        for i, char in enumerate(text):
            if char in sentence_end_chars and i >= min_length:
                return text[:i+1].strip()
        
        # Priority 2: Look for comma boundaries (natural pause points)
        for i, char in enumerate(text):
            if char == ',' and i >= min_length:
                return text[:i+1].strip()
        
        # Priority 3: Find word boundary after min_length
        if len(text) > min_length + 10:
            # Look for the last space after min_length to avoid splitting words
            chunk_end = text.rfind(' ', min_length, min_length + 20)
            if chunk_end > min_length:
                return text[:chunk_end].strip()
        
        return None

    while not stop_event.is_set():
        try:
            prompt = transcription_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if prompt.lower() in ["exit", "quit", "shutdown"]:
            print("Shutdown command detected, stopping.")
            stop_event.set()
            transcription_queue.task_done()
            break

        print("⏱️ LLM call started")
        start = time.time()
        buffer = ""
        chunk_sequence = 0
        
        try:
            res = requests.post(
                "http://localhost:11434/api/generate",
                json={
                "model": "phi3:mini",
                "prompt": f"Answer briefly in 1-2 sentences: {prompt}",
                "stream": True,
                "options": {
                    "num_predict": 25,  # Further reduced for faster response
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
                },
                stream=True,
                timeout=60
            )
            
            for line in res.iter_lines():
                if stop_event.is_set():
                    break
                if line:
                    decoded = line.decode('utf-8')
                    try:
                        data = json.loads(decoded)
                        chunk = data.get("response", "")
                        if chunk:
                            buffer += chunk

                            # Check for smart chunk boundary
                            chunked_text = find_smart_chunk_boundary(buffer)
                            if chunked_text:
                                # Add sequence number for ordered playback
                                tts_text_queue.put((chunk_sequence, chunked_text))
                                buffer = buffer[len(chunked_text):].strip()
                                chunk_sequence += 1
                                
                    except json.JSONDecodeError:
                        continue
            
            # Flush remaining buffer at end
            if buffer.strip():
                tts_text_queue.put((chunk_sequence, buffer.strip()))

            print(f"⏱️ LLM call finished in {time.time() - start:.2f}s")

        except Exception as e:
            print("LLM request error:", e)
            tts_text_queue.put((0, "Sorry, I encountered an error."))

        transcription_queue.task_done()

def tts_synthesis_worker():
    """
    Synthesizes audio from text chunks and pushes to playback queue.
    Simplified approach for immediate playback.
    """
    while not stop_event.is_set():
        try:
            text_data = tts_text_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        try:
            # Unpack sequence and text
            sequence, text_chunk = text_data
            
            start = time.time()
            # Use p230 - British male, fastest synthesis with clear accent
            wav = vits_tts.tts(text_chunk, speaker="p230")
            synthesis_time = time.time() - start
            
            print(f"⏱️ TTS synthesis took {synthesis_time:.2f}s for chunk #{sequence}: {text_chunk[:30]!r}")
            
            # Immediately queue for playback with sequence info
            tts_audio_queue.put((sequence, wav))
                
        except Exception as e:
            print("💥 TTS Error:", e)
            
        tts_text_queue.task_done()

def tts_playback_worker():
    """
    Plays synthesized audio chunks with proper completion and seamless flow.
    """
    playback_buffer = {}
    next_expected_sequence = 0
    is_playing = False
    
    while not stop_event.is_set():
        try:
            audio_data = tts_audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        try:
            # Unpack sequence and audio
            sequence, audio_chunk = audio_data
            
            # Reset sequence for new query (if sequence 0 appears and we're not at the beginning)
            if sequence == 0 and next_expected_sequence != 0:
                # Wait for any current playback to finish
                if is_playing:
                    sd.wait()
                    is_playing = False
                
                # New query started, reset counter and clear buffer
                next_expected_sequence = 0
                playback_buffer.clear()
            
            # Store in buffer with sequence number
            playback_buffer[sequence] = audio_chunk
            
            # Play chunks in sequence order
            while next_expected_sequence in playback_buffer:
                chunk_to_play = playback_buffer[next_expected_sequence]
                
                print(f"🔊 Playing chunk #{next_expected_sequence}")
                
                # Calculate expected duration for this chunk
                chunk_duration = len(chunk_to_play) / VITS_SR
                
                # Play with minimal gap for seamless flow
                sd.play(chunk_to_play, VITS_SR, blocking=True)
                
                # Small pause to prevent audio artifacts
                time.sleep(0.05)  # 50ms gap between chunks
                
                del playback_buffer[next_expected_sequence]
                next_expected_sequence += 1
                
        except Exception as e:
            print("💥 Audio Playback Error:", e)
            
        tts_audio_queue.task_done()

def main():
    # Single TTS worker for proper sequencing
    threads = [
        Thread(target=transcription_worker, daemon=True),
        Thread(target=llm_worker, daemon=True),
        Thread(target=tts_synthesis_worker, daemon=True),
        Thread(target=tts_playback_worker, daemon=True),
    ]

    for t in threads:
        t.start()

    print("✅ Ready. Say 'Jarvis' to activate.")

    try:
        while not stop_event.is_set():
            hotword_start = time.time()
            pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
            samples = struct.unpack('h' * porcupine.frame_length, pcm)
            if porcupine.process(samples) >= 0:
                hotword_end = time.time()
                print(f"⏱️ Hotword detection took {hotword_end - hotword_start:.2f}s")
                print("Hotword detected!")

                record_start = time.time()
                audio = record_audio_vad()
                record_end = time.time()
                print(f"⏱️ Recording took {record_end - record_start:.2f}s")

                audio_queue.put(audio)

                # Small delay to avoid busy loop
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nExiting on user interrupt.")

    finally:
        print("Stopping threads...")
        stop_event.set()
        audio_queue.join()
        transcription_queue.join()
        tts_text_queue.join()
        tts_audio_queue.join()

        audio_stream.stop_stream()
        audio_stream.close()
        pa.terminate()
        porcupine.delete()

if __name__ == "__main__":
    main()
