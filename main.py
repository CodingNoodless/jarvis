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

def record_audio_vad(max_duration=30, samplerate=SAMPLERATE):
    # Smart VAD for detecting complete speech, including long prompts
    vad = webrtcvad.Vad(2)  # More aggressive VAD for better speech detection
    frame_duration = 30
    frame_size = int(samplerate * frame_duration / 1000)
    ring = collections.deque(maxlen=10)  # Larger ring buffer for better speech detection
    q = queue.Queue()
    recording = []
    triggered = False
    silence_count = 0
    min_speech_frames = 4  # Minimum frames to consider speech started
    # Dynamic silence threshold - longer silence required after longer speech
    base_silence_threshold = 8  # Base frames of silence to stop recording
    speech_duration_frames = 0  # Track how long we've been recording speech

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
                speech_duration_frames += 1
                if speech:
                    silence_count = 0
                else:
                    silence_count += 1
                    # Dynamic silence threshold based on speech duration
                    # Longer speeches require more silence to confirm end
                    if speech_duration_frames > 100:  # Long speech (>3 seconds)
                        silence_threshold = base_silence_threshold + 6  # 14 frames (~0.4s)
                    elif speech_duration_frames > 50:  # Medium speech (>1.5 seconds)
                        silence_threshold = base_silence_threshold + 3  # 11 frames (~0.33s)
                    else:  # Short speech
                        silence_threshold = base_silence_threshold  # 8 frames (~0.24s)
                    
                    if silence_count >= silence_threshold:
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

    def find_smart_chunk_boundary(text, min_length=10, chunk_sequence=0):
        """Ensure complete words and low latency for first chunk"""
        text = text.strip()
        
        # For first chunk, be more aggressive to reduce latency
        if chunk_sequence == 0:
            min_length = 4  # Ultra-short first chunk for immediate response
        elif chunk_sequence == 1:
            min_length = 8  # Second chunk slightly longer
        else:
            min_length = max(min_length, 12)  # Subsequent chunks longer for better quality
        
        # Don't chunk if text is too short
        if len(text) < min_length:
            return None
        
        # Priority 1: Look for sentence endings (natural pause points)
        for i, char in enumerate(text):
            if char in sentence_end_chars and i >= min_length:
                # Make sure we're at a word boundary
                if i + 1 >= len(text) or text[i + 1] == ' ':
                    return text[:i+1].strip()
        
        # Priority 2: Look for comma boundaries (natural pause points)
        for i, char in enumerate(text):
            if char == ',' and i >= min_length:
                # Make sure we're at a word boundary
                if i + 1 >= len(text) or text[i + 1] == ' ':
                    return text[:i+1].strip()
        
        # Priority 3: Find word boundary - NEVER break words
        # Look for spaces after min_length
        for i in range(min_length, len(text)):
            if text[i] == ' ':
                # Ensure we have complete words by checking boundaries
                chunk = text[:i].strip()
                if chunk and ' ' in chunk:  # Ensure we have at least one complete word
                    return chunk
        
        # If no good boundary found, don't chunk yet
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
                "prompt": f"You are Jarvis, a helpful AI assistant. Provide a clear, complete answer to: {prompt}",
                "stream": True,
                "options": {
                    "num_predict": 150,  # Allow longer responses to prevent cut-off
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "stop": ["\n\n", "User:", "Human:"]  # Natural stopping points
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
                            chunked_text = find_smart_chunk_boundary(buffer, chunk_sequence=chunk_sequence)
                            if chunked_text:
                                # Add sequence number for ordered playback
                                tts_text_queue.put((chunk_sequence, chunked_text))
                                # Remove exactly what we sent, but preserve word boundaries
                                # Find the exact end position of the chunked text in buffer
                                chunk_end_pos = buffer.find(chunked_text) + len(chunked_text)
                                # Skip any trailing spaces/punctuation but preserve next word
                                while chunk_end_pos < len(buffer) and buffer[chunk_end_pos] in ' .,!?':
                                    chunk_end_pos += 1
                                buffer = buffer[chunk_end_pos:]
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
            
            print(f"⏱️ TTS synthesis took {synthesis_time:.2f}s for chunk #{sequence}: {text_chunk!r}")
            
            # Immediately queue for playback with sequence info
            tts_audio_queue.put((sequence, wav))
            
            # Force immediate processing by yielding to other threads
            time.sleep(0.001)
                
        except Exception as e:
            print("💥 TTS Error:", e)
            
        tts_text_queue.task_done()

def tts_playback_worker():
    """
    Plays synthesized audio chunks with true seamless streaming playback.
    Prevents duplicate playback and ensures proper sequencing.
    """
    playback_buffer = {}
    next_expected_sequence = 0
    played_sequences = set()  # Track played sequences to prevent duplicates
    current_query_id = None
    
    while not stop_event.is_set():
        # Check if we have audio data ready
        try:
            audio_data = tts_audio_queue.get(timeout=0.001)  # Ultra-frequent checks
            sequence, audio_chunk = audio_data
            
            # Reset sequence for new query (if sequence 0 appears and we're not at the beginning)
            if sequence == 0 and next_expected_sequence != 0:
                # Stop current playback immediately
                sd.stop()
                # New query started, reset everything
                next_expected_sequence = 0
                playback_buffer.clear()
                played_sequences.clear()
                current_query_id = time.time()  # Use timestamp as query ID
            
            # Avoid duplicate chunks - skip if already played
            if sequence in played_sequences:
                print(f"⚠️ Skipping duplicate chunk #{sequence}")
                tts_audio_queue.task_done()
                continue
            
            # Store in buffer with sequence number
            playback_buffer[sequence] = audio_chunk
            tts_audio_queue.task_done()
            
        except queue.Empty:
            pass
        
        # Play all available chunks in sequence immediately
        while next_expected_sequence in playback_buffer:
            # Double-check we haven't played this sequence
            if next_expected_sequence in played_sequences:
                del playback_buffer[next_expected_sequence]
                next_expected_sequence += 1
                continue
                
            chunk_to_play = playback_buffer[next_expected_sequence]
            
            print(f"🔊 Playing chunk #{next_expected_sequence}")
            
            # Mark as played BEFORE playing to prevent race conditions
            played_sequences.add(next_expected_sequence)
            
            # NON-BLOCKING playback for immediate response
            sd.play(chunk_to_play, VITS_SR, blocking=False)
            
            # Calculate playback duration to ensure proper timing
            playback_duration = len(chunk_to_play) / VITS_SR
            
            # Wait for this chunk to finish before starting next
            # Use precise timing to eliminate gaps completely  
            time.sleep(max(0.001, playback_duration - 0.01))
            
            del playback_buffer[next_expected_sequence]
            next_expected_sequence += 1
        
        # Minimal sleep only when no chunks are available
        time.sleep(0.001)  # 1ms check interval

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
