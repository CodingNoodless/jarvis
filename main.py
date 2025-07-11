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
from threading import Thread, Event, Lock
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
vits_tts = CoquiTTS(model_name="tts_models/en/vctk/vits", gpu=True)
print("⏳ Warming up TTS...")
_ = vits_tts.tts("Hello, Jarvis is ready.", speaker="p230")
VITS_SR = 22050

print("🎧 Initializing hotword detector...")
porcupine = pvporcupine.create(access_key="BMthPl0em5470FOwow1fy6/z50cW1UKMarJqcTV7Pvb32BZVxkR1Tw==", keywords=["jarvis"])
pa = pyaudio.PyAudio()
audio_stream = pa.open(rate=porcupine.sample_rate, channels=1, format=pyaudio.paInt16,
                       input=True, frames_per_buffer=porcupine.frame_length)

audio_queue = queue.Queue()
transcription_queue = queue.Queue()
audio_chunk_queue = queue.Queue()
tts_synthesis_queue = queue.PriorityQueue()
stop_event = Event()

# Audio playback state - REVAMPED SYSTEM
audio_stream_lock = Lock()
current_audio_stream = None
audio_buffer = np.array([], dtype=np.float32)
buffer_ready = Event()
is_playing = False
prebuffer_threshold = 0.5  # 500ms of audio before starting playback

# Multiple TTS instances for parallel synthesis
NUM_TTS_WORKERS = 3
tts_instances = []

print("🔊 Loading multiple TTS instances for parallel synthesis...")
for i in range(NUM_TTS_WORKERS):
    print(f"🔊 Loading TTS instance {i+1}/{NUM_TTS_WORKERS}...")
    tts_instance = CoquiTTS(model_name="tts_models/en/vctk/vits", gpu=True)
    _ = tts_instance.tts("Warmup", speaker="p230")
    tts_instances.append(tts_instance)
print(f"✅ {NUM_TTS_WORKERS} TTS instances loaded")

def record_audio_vad(max_duration=30, samplerate=SAMPLERATE):
    vad = webrtcvad.Vad(2)
    frame_duration = 30
    frame_size = int(samplerate * frame_duration / 1000)
    ring = collections.deque(maxlen=10)
    q = queue.Queue()
    recording = []
    triggered = False
    silence_count = 0
    min_speech_frames = 4
    base_silence_threshold = 8
    speech_duration_frames = 0

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
                    if speech_duration_frames > 100:
                        silence_threshold = base_silence_threshold + 6
                    elif speech_duration_frames > 50:
                        silence_threshold = base_silence_threshold + 3
                    else:
                        silence_threshold = base_silence_threshold
                    
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
    # Expanded punctuation marks for better chunking
    chunk_end_chars = {'.', '!', '?', ',', ';', ':', '—', '–', '-'}
    buffer = ""
    chunk_sequence = 0

    def find_punctuation_chunk(text, chunk_sequence=0):
        """Find the next complete chunk ending at punctuation"""
        text = text.strip()
        
        # Don't send empty chunks
        if not text:
            return None
        
        # For the first chunk, allow smaller chunks to start playback quickly
        min_length = 10 if chunk_sequence == 0 else 15
        
        # If text is shorter than minimum and we have some content, return it
        if len(text) < min_length and chunk_sequence > 0:
            return None
        
        # Look for punctuation marks
        for i, char in enumerate(text):
            if char in chunk_end_chars and i >= min_length:
                # Make sure we include the punctuation and any following space
                end_pos = i + 1
                while end_pos < len(text) and text[end_pos] in ' \t':
                    end_pos += 1
                return text[:end_pos].strip()
        
        # If we have a very long text without punctuation, break at a space
        if len(text) > 100:
            for i in range(80, len(text)):
                if text[i] == ' ':
                    return text[:i].strip()
        
        # If no punctuation found and not too long, return None to wait for more
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
        
        # Clear audio buffer for new response
        with audio_stream_lock:
            global audio_buffer
            audio_buffer = np.array([], dtype=np.float32)
        
        try:
            res = requests.post(
                "http://localhost:11434/api/generate",
                json={
                "model": "phi3:mini",
                "prompt": f"You are Jarvis, a helpful AI assistant. Provide a clear, complete answer to: {prompt}",
                "stream": True,
                "options": {
                    "num_predict": 150,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "stop": ["\n\n", "User:", "Human:"]
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

                            # Try to find a complete chunk ending with punctuation
                            chunked_text = find_punctuation_chunk(buffer, chunk_sequence=chunk_sequence)
                            if chunked_text:
                                print(f"📝 Sending chunk {chunk_sequence}: {chunked_text!r}")
                                tts_synthesis_queue.put((0, chunk_sequence, chunked_text))
                                
                                # Remove the chunked text from buffer
                                buffer = buffer[len(chunked_text):].lstrip()
                                chunk_sequence += 1
                                
                    except json.JSONDecodeError:
                        continue
            
            # Send any remaining text as the final chunk
            if buffer.strip():
                print(f"📝 Sending final chunk {chunk_sequence}: {buffer.strip()!r}")
                tts_synthesis_queue.put((0, chunk_sequence, buffer.strip()))

            print(f"⏱️ LLM call finished in {time.time() - start:.2f}s")

        except Exception as e:
            print("LLM request error:", e)
            tts_synthesis_queue.put((0, 0, "Sorry, I encountered an error."))

        transcription_queue.task_done()

def tts_synthesis_worker(worker_id):
    tts_instance = tts_instances[worker_id]
    
    while not stop_event.is_set():
        try:
            priority, sequence, text_chunk = tts_synthesis_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        try:
            start = time.time()
            wav = tts_instance.tts(text_chunk, speaker="p230", speed=1.2)
            synthesis_time = time.time() - start
            
            print(f"⏱️ Worker {worker_id} TTS synthesis took {synthesis_time:.2f}s for chunk {sequence}: {text_chunk!r}")
            
            if isinstance(wav, list):
                wav = np.array(wav, dtype=np.float32)
            
            max_val = np.max(np.abs(wav))
            if max_val > 0:
                wav = wav / max_val * 0.95
            
            # Apply fade-out only at the very end of the response
            if sequence > 1000:  # Special marker for final chunk
                if len(wav) > 100:
                    wav[-50:] *= np.linspace(1, 0, 50)
            
            audio_chunk_queue.put((sequence, wav))
            
        except Exception as e:
            print(f"💥 Worker {worker_id} TTS Error:", e)
            
        tts_synthesis_queue.task_done()

def audio_callback(outdata, frames, time_info, status):
    """Optimized audio callback with continuous playback"""
    global audio_buffer
    outdata[:, 0] = 0  # Initialize with silence
    
    with audio_stream_lock:
        available = len(audio_buffer)
        if available == 0:
            return
        
        # Calculate how much we can copy
        to_copy = min(frames, available)
        outdata[:to_copy, 0] = audio_buffer[:to_copy]
        
        # Remove the copied portion
        audio_buffer = audio_buffer[to_copy:]
        
        # If we didn't fill the buffer, silence will remain
        # which is acceptable for end of response

def continuous_audio_feeder():
    """Advanced audio feeder with pre-buffering and seamless concatenation"""
    global audio_buffer
    
    # We'll collect chunks until we have enough to start playback
    pending_chunks = []
    current_sequence = 0
    response_start_time = 0
    first_chunk_received = False
    
    while not stop_event.is_set():
        try:
            sequence, audio_chunk = audio_chunk_queue.get(timeout=0.001)
            
            # If this is the first chunk of a new response
            if not first_chunk_received:
                first_chunk_received = True
                response_start_time = time.time()
                pending_chunks = []
                current_sequence = sequence
            
            # Track sequence to handle out-of-order chunks
            if sequence == current_sequence:
                pending_chunks.append(audio_chunk)
                current_sequence += 1
                
                # Check if we have the next chunks already
                while True:
                    next_seq = current_sequence
                    found = False
                    # Check if we have the next chunk in queue without blocking
                    for i in range(audio_chunk_queue.qsize()):
                        try:
                            seq, chunk = audio_chunk_queue.get_nowait()
                            if seq == next_seq:
                                pending_chunks.append(chunk)
                                current_sequence += 1
                                found = True
                            else:
                                # Not in sequence, put back
                                audio_chunk_queue.put((seq, chunk))
                        except queue.Empty:
                            break
                    if not found:
                        break
                
                # Concatenate all pending chunks
                if pending_chunks:
                    # Apply crossfade between chunks
                    concatenated = pending_chunks[0]
                    for i in range(1, len(pending_chunks)):
                        prev = concatenated
                        next_chunk = pending_chunks[i]
                        
                        # Crossfade the last 10ms of previous with first 10ms of next
                        fade_len = min(220, len(prev), min(220, len(next_chunk)))  # 10ms at 22.05kHz
                        
                        if fade_len > 0:
                            # Fade out previous
                            fade_out = np.linspace(1.0, 0.0, fade_len)
                            prev[-fade_len:] *= fade_out
                            
                            # Fade in next
                            fade_in = np.linspace(0.0, 1.0, fade_len)
                            next_chunk[:fade_len] *= fade_in
                            
                            # Overlap-add
                            concatenated = np.concatenate([
                                concatenated[:-fade_len],
                                concatenated[-fade_len:] + next_chunk[:fade_len],
                                next_chunk[fade_len:]
                            ])
                        else:
                            concatenated = np.concatenate([concatenated, next_chunk])
                    
                    # Add to global buffer
                    with audio_stream_lock:
                        audio_buffer = np.concatenate([audio_buffer, concatenated])
                    
                    print(f"🔊 Added {len(pending_chunks)} chunks to buffer (total: {len(audio_buffer)/VITS_SR:.2f}s)")
                    pending_chunks = []
            
            audio_chunk_queue.task_done()
            
        except queue.Empty:
            # If we have pending chunks but no new ones, flush them
            if pending_chunks:
                concatenated = np.concatenate(pending_chunks)
                with audio_stream_lock:
                    audio_buffer = np.concatenate([audio_buffer, concatenated])
                print(f"🔊 Flushed {len(pending_chunks)} chunks to buffer (total: {len(audio_buffer)/VITS_SR:.2f}s)")
                pending_chunks = []
                first_chunk_received = False
            pass

def main():
    global current_audio_stream
    
    # Start continuous audio stream
    current_audio_stream = sd.OutputStream(
        samplerate=VITS_SR,
        channels=1,
        dtype='float32',
        callback=audio_callback,
        blocksize=2048,  # Larger blocksize for stability
        latency='low'
    )
    current_audio_stream.start()
    
    threads = [
        Thread(target=transcription_worker, daemon=True),
        Thread(target=llm_worker, daemon=True),
        Thread(target=continuous_audio_feeder, daemon=True),
    ]
    
    # Add TTS synthesis workers
    for i in range(NUM_TTS_WORKERS):
        threads.append(Thread(target=tts_synthesis_worker, args=(i,), daemon=True))

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
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nExiting on user interrupt.")

    finally:
        print("Stopping threads...")
        stop_event.set()
        
        if current_audio_stream:
            current_audio_stream.stop()
            current_audio_stream.close()
        
        audio_queue.join()
        transcription_queue.join()
        tts_synthesis_queue.join()
        audio_chunk_queue.join()

        audio_stream.stop_stream()
        audio_stream.close()
        pa.terminate()
        porcupine.delete()

if __name__ == "__main__":
    main()