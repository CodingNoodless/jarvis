import sounddevice as sd
import numpy as np
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
import os
import importlib.util
import sys
from pathlib import Path
from typing import Optional
import traceback
import re
from langchain_core.tools import tool, StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamingStdOutCallbackHandler

warnings.filterwarnings("ignore", category=UserWarning)
SAMPLERATE = 16000

# Skills system configuration
SKILLS_DIR = "./skillss/"

class SkillManager:
    """Manages dynamic loading and reloading of skills"""
    
    def __init__(self, skills_dir: str = SKILLS_DIR):
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(exist_ok=True)
        self.loaded_skills = {}
        self.last_modified = {}
        self.lock = Lock()
        print(f"📁 Skills directory: {self.skills_dir.absolute()}")
        
    def _load_skill_from_file(self, skill_file: Path):
        """Load a skill from a Python file"""
        try:
            module_name = skill_file.stem
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            spec = importlib.util.spec_from_file_location(module_name, skill_file)
            if not spec or not spec.loader:
                print(f"❌ Could not create spec for {skill_file.name}")
                return None
                
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            if not hasattr(module, 'skill'):
                print(f"⚠️ No 'skill' attribute in {skill_file.name}")
                return None
                
            skill = module.skill
            if isinstance(skill, type) and issubclass(skill, BaseModel) and not isinstance(skill, BaseModel):
                # Convert Pydantic model to a tool
                tool = StructuredTool.from_pydantic(skill)
                print(f"✅ Loaded Pydantic tool: {tool.name}")
                return tool
            elif callable(skill) and hasattr(skill, '__tool__'):
                # Function decorated with @tool
                print(f"✅ Loaded function tool: {skill.name}")
                return skill
            elif isinstance(skill, StructuredTool):
                # Directly a StructuredTool
                print(f"✅ Loaded StructuredTool: {skill.name}")
                return skill
            else:
                print(f"⚠️ Unsupported skill type in {skill_file.name}: {type(skill)}")
                return None
            
        except Exception as e:
            print(f"❌ Error loading skill {skill_file.name}: {e}")
            traceback.print_exc()
            return None
    
    def load_skills(self):
        """Load all skills from the skills directory"""
        with self.lock:
            current_skills = []
            if not self.skills_dir.exists():
                print(f"⚠️ Skills directory does not exist: {self.skills_dir}")
                return current_skills
            
            skill_files = list(self.skills_dir.glob("*.py"))
            if not skill_files:
                print("⚠️ No skill files found in skills directory")
                return current_skills
            
            print(f"🔍 Found {len(skill_files)} skill files")
            
            for skill_file in skill_files:
                if skill_file.name.startswith("_"):
                    continue
                    
                skill_name = skill_file.stem
                
                try:
                    file_mtime = skill_file.stat().st_mtime
                except OSError as e:
                    print(f"❌ Cannot access file {skill_file.name}: {e}")
                    continue
                
                needs_reload = (
                    skill_name not in self.loaded_skills or 
                    skill_name not in self.last_modified or
                    file_mtime > self.last_modified.get(skill_name, 0)
                )
                
                if needs_reload:
                    print(f"🔄 Loading skill: {skill_name}")
                    skill = self._load_skill_from_file(skill_file)
                    
                    if skill:
                        self.loaded_skills[skill_name] = skill
                        self.last_modified[skill_name] = file_mtime
                        print(f"✅ Loaded skill: {skill.name} - {skill.description}")
                    else:
                        if skill_name in self.loaded_skills:
                            del self.loaded_skills[skill_name]
                            print(f"🗑️ Removed failed skill: {skill_name}")
                        if skill_name in self.last_modified:
                            del self.last_modified[skill_name]
                
                if skill_name in self.loaded_skills:
                    current_skills.append(self.loaded_skills[skill_name])
            
            print(f"📊 Total loaded skills: {len(current_skills)}")
            return current_skills
    
    def get_skill_info(self) -> str:
        """Get information about loaded skills"""
        with self.lock:
            if not self.loaded_skills:
                return "No skills loaded."
            
            info = "Loaded skills:\n"
            for skill in self.loaded_skills.values():
                info += f"- {skill.name}: {skill.description}\n"
            return info.strip()

class JarvisAgent:
    def __init__(self):
        self.skill_manager = SkillManager()
        
        # Initialize LLM with better configuration
        self.llm = OllamaLLM(
            model="gemma:2b",
            base_url="http://localhost:11434",
            temperature=0.7,
            timeout=30,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        # Initialize memory with updated implementation
        self.history = ChatMessageHistory()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=self.history,
            return_messages=True,
            input_key="input",
            output_key="output"
        )
        
        self.agent_executor = None
        self.last_skill_count = 0
        self._setup_agent()

    def _test_llm_connection(self) -> bool:
        """Test if LLM is accessible"""
        try:
            print("🔍 Testing LLM connection...")
            response = self.llm.invoke("Hello")
            print(f"✅ LLM connection successful: {response[:50]}...")
            return True
        except Exception as e:
            print(f"❌ LLM connection failed: {e}")
            return False

    def _setup_agent(self):
        """Set up the agent with loaded skills"""
        skills = self.skill_manager.load_skills()
        self.last_skill_count = len(skills)
        
        if not skills:
            print("⚠️ No skills loaded, agent will not be initialized")
            self.agent_executor = None
            return
        
        print(f"🔧 Setting up agent with {len(skills)} skills")
        
        # Bind tools to the LLM
        try:
            llm_with_tools = self.llm.bind_tools(tools=skills)
        except Exception as e:
            print(f"❌ Failed to bind tools to LLM: {e}")
            traceback.print_exc()
            self.agent_executor = None
            return
        
        # Define the prompt template with explicit tool usage instruction
        prompt = PromptTemplate.from_template("""
You are Jarvis, an AI assistant dedicated to serving Rohit Pulle. Answer the following question as best you can. You have access to the following tools:

{tools}

If the question involves time or location, use the get_current_time tool with the specified location.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")
        
        try:
            agent = create_react_agent(self.llm, skills, prompt)
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=skills,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True
            )
            print("✅ Agent initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            traceback.print_exc()
            self.agent_executor = None

    def _should_reload_skills(self) -> bool:
        """Check if skills need to be reloaded"""
        current_skills = self.skill_manager.load_skills()
        return len(current_skills) != self.last_skill_count

    def reload_skills(self):
        """Reload skills and reinitialize agent"""
        print("🔄 Reloading skills and reinitializing agent...")
        self._setup_agent()
    
    def get_skill_info(self) -> str:
        """Get information about loaded skills"""
        return self.skill_manager.get_skill_info()
    
    def process_query(self, query: str) -> str:
        """Process a query using the agent or fallback to basic LLM"""
        if not query or not query.strip():
            return "I didn't receive a valid query. Please try again."
        
        query = query.strip()
        
        try:
            if self._should_reload_skills():
                print("🔄 Skills changed, reloading agent...")
                self.reload_skills()
            
            if self.agent_executor:
                print(f"🤖 Processing query with agent: {query}")
                response = self.agent_executor.invoke({"input": query})
                if isinstance(response, dict):
                    result = response.get("output", response.get("result", str(response)))
                else:
                    result = str(response)
                
                return result.strip() if result and result.strip() else "I processed your request but didn't generate a response."
            else:
                print("🔄 No agent available, using basic LLM...")
                return self._basic_llm_response(query)
                
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            traceback.print_exc()
            return "I apologize, I encountered an error processing your request."
    
    def _basic_llm_response(self, query: str) -> str:
        """Fallback to basic LLM with tool usage for time queries"""
        try:
            # Check if the query is about time and location
            time_pattern = re.compile(r"(?:what(?:'s| is)\s*(?:the)?\s*(?:current)?\s*time\s*(?:in)?\s*([a-zA-Z\s]+))", re.IGNORECASE)
            match = time_pattern.search(query)
            if match:
                location = match.group(1).strip()
                # Find the get_current_time tool
                for skill in self.skill_manager.loaded_skills.values():
                    if skill.name == "get_current_time":
                        return skill.invoke({"location": location})
            
            # Fallback to basic LLM if no time query
            prompt = f"""You are Jarvis, an AI assistant dedicated to serving Rohit Pulle. 
Provide a helpful, clear, and concise answer to the following question:

Question: {query}

Answer:"""
            
            response = self.llm.invoke(prompt)
            return response.strip() if response and response.strip() else "I'm sorry, I couldn't generate a response to your question."
                
        except Exception as e:
            print(f"❌ Basic LLM failed: {e}")
            traceback.print_exc()
            return "I apologize, I'm having trouble processing your request right now."

    def get_status(self) -> str:
        """Get current agent status"""
        skills = self.skill_manager.load_skills()
        status = f"Agent Status:\n"
        status += f"- Skills loaded: {len(skills)}\n"
        status += f"- Agent active: {'Yes' if self.agent_executor else 'No'}\n"
        status += f"- LLM model: gemma:2b\n"
        
        if skills:
            status += "\nAvailable skills:\n"
            for skill in skills:
                status += f"  - {skill.name}: {skill.description}\n"
        
        return status

# Initialize the enhanced Jarvis agent
jarvis_agent = None

print("🔊 Loading Whisper model...")
whisper_model = WhisperModel("tiny", compute_type="int8")

print("🔊 Loading British TTS model...")
vits_tts = CoquiTTS(model_name="tts_models/en/vctk/vits", gpu=False)
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
response_complete_event = Event()

# Audio playback state
audio_stream_lock = Lock()
current_audio_stream = None
audio_buffer = np.array([], dtype=np.float32)
is_playing = False
playback_active = False
prebuffer_threshold = 0.3

# Multiple TTS instances for parallel synthesis
NUM_TTS_WORKERS = 3
tts_instances = []

print("🔊 Loading multiple TTS instances for parallel synthesis...")
for i in range(NUM_TTS_WORKERS):
    print(f"🔊 Loading TTS instance {i+1}/{NUM_TTS_WORKERS}...")
    tts_instance = CoquiTTS(model_name="tts_models/en/vctk/vits", gpu=False)
    _ = tts_instance.tts("Warmup", speaker="p230")
    tts_instances.append(tts_instance)
print(f"✅ {NUM_TTS_WORKERS} TTS instances loaded")

# Initialize Jarvis agent
print("🤖 Initializing Jarvis agent with skills...")
try:
    jarvis_agent = JarvisAgent()
    print("✅ Jarvis agent initialized")
    print(jarvis_agent.get_status())
except Exception as e:
    print(f"❌ Failed to initialize Jarvis agent: {e}")
    traceback.print_exc()
    print("🔄 Continuing with basic functionality...")

def record_audio_vad(max_duration=30, samplerate=SAMPLERATE):
    vad = webrtcvad.Vad(1)
    frame_duration = 30
    frame_size = int(samplerate * frame_duration / 1000)
    ring = collections.deque(maxlen=15)
    q = queue.Queue()
    recording = []
    triggered = False
    silence_count = 0
    min_speech_frames = 3
    base_silence_threshold = 15
    speech_duration_frames = 0
    total_frames = 0
    last_speech_frame = 0
    
    energy_threshold = 0.002
    energy_history = collections.deque(maxlen=20)

    def callback(indata, frames, time_info, status):
        q.put(indata[:, 0].copy())

    def calculate_energy(chunk):
        return np.sqrt(np.mean(chunk**2))

    def is_speech_energy(chunk, threshold):
        energy = calculate_energy(chunk)
        energy_history.append(energy)
        
        if len(energy_history) < 10:
            return False
            
        avg_energy = np.mean(energy_history)
        return energy > threshold and energy > avg_energy * 1.5

    print("🎙️ Recording... (speak now)")
    
    with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32',
                        blocksize=frame_size, callback=callback):
        start = time.time()
        while True:
            if time.time() - start > max_duration:
                print("⏰ Recording stopped: maximum duration reached")
                break
                
            try:
                chunk = q.get(timeout=0.1)
            except queue.Empty:
                continue
            
            total_frames += 1
            pcm = (chunk * 32767).astype(np.int16).tobytes()
            
            speech_vad = vad.is_speech(pcm, samplerate)
            speech_energy = is_speech_energy(chunk, energy_threshold)
            speech = speech_vad or speech_energy
            
            if not triggered:
                ring.append((chunk, speech))
                speech_frames = sum(1 for c, s in ring if s)
                
                if speech_frames >= min_speech_frames:
                    triggered = True
                    print("🎯 Speech detected, recording started")
                    for c, _ in ring:
                        recording.append(c)
                    ring.clear()
                    silence_count = 0
                    last_speech_frame = total_frames
            else:
                recording.append(chunk)
                speech_duration_frames += 1
                
                if speech:
                    silence_count = 0
                    last_speech_frame = total_frames
                    if speech_duration_frames % 30 == 0:
                        print(f"🗣️ Recording... ({speech_duration_frames * frame_duration / 1000:.1f}s)")
                else:
                    silence_count += 1
                    
                    if speech_duration_frames > 200:
                        silence_threshold = base_silence_threshold + 12
                    elif speech_duration_frames > 100:
                        silence_threshold = base_silence_threshold + 8
                    elif speech_duration_frames > 50:
                        silence_threshold = base_silence_threshold + 4
                    else:
                        silence_threshold = base_silence_threshold
                    
                    frames_since_speech = total_frames - last_speech_frame
                    if frames_since_speech < 20:
                        silence_threshold += 5
                    
                    min_recording_duration_frames = 30
                    if speech_duration_frames < min_recording_duration_frames:
                        silence_threshold = max(silence_threshold, 20)
                    
                    if silence_count >= silence_threshold:
                        print(f"🔇 Silence detected ({silence_count * frame_duration}ms), stopping recording")
                        break
    
    if not recording:
        print("⚠️ No speech detected, returning empty audio")
        return np.zeros(frame_size, dtype=np.float32)
    
    final_audio = np.concatenate(recording)
    duration = len(final_audio) / samplerate
    print(f"✅ Recording complete: {duration:.2f}s of audio captured")
    return final_audio

def transcription_worker():
    while not stop_event.is_set():
        try:
            audio_np = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        try:
            start = time.time()
            buf = BytesIO()
            sf.write(buf, audio_np, SAMPLERATE, format='WAV')
            buf.seek(0)
            segments, _ = whisper_model.transcribe(buf)
            text = " ".join(seg.text for seg in segments).strip()
            print(f"⏱️ Transcription took {time.time() - start:.2f}s")
            
            if text:
                print("You said:", text)
                transcription_queue.put(text)
            else:
                print("⚠️ No transcription result")
                
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            traceback.print_exc()
            
        audio_queue.task_done()

def split_into_chunks(text, min_len=20):
    """Split response into sentence-like chunks."""
    text = text.strip()
    if not text:
        return []

    sentence_enders = re.compile(r'([.!?])')
    parts = sentence_enders.split(text)
    chunks = []

    current = ""
    for part in parts:
        if part in ".!?":
            current += part
            if len(current.strip()) >= min_len:
                chunks.append(current.strip())
                current = ""
        else:
            current += part

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]

def llm_worker():
    while not stop_event.is_set():
        try:
            prompt = transcription_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        print("⏱️ LLM call started")
        start = time.time()

        global playback_active
        response_complete_event.clear()
        with audio_stream_lock:
            global audio_buffer
            audio_buffer = np.array([], dtype=np.float32)
            playback_active = True

        try:
            if jarvis_agent:
                response = jarvis_agent.process_query(prompt)
            else:
                response = "Sorry, I have no brain right now."

            if not response or not response.strip():
                response = "I'm sorry, I couldn't generate a response."

            chunks = split_into_chunks(response)
            print(f"💬 Split into {len(chunks)} chunks")

            for i, chunk in enumerate(chunks):
                is_final = i == len(chunks) - 1
                print(f"📤 Sending chunk {i}: {chunk!r} (final={is_final})")
                tts_synthesis_queue.put((0, i, chunk, is_final))

        except Exception as e:
            print("💥 LLM request error:", e)
            traceback.print_exc()
            tts_synthesis_queue.put((0, 0, "Sorry, I encountered an error processing your request.", True))

        transcription_queue.task_done()
        print(f"⏱️ LLM call finished in {time.time() - start:.2f}s")

def tts_synthesis_worker(worker_id):
    tts_instance = tts_instances[worker_id]
    
    while not stop_event.is_set():
        try:
            priority, sequence, text_chunk, is_final = tts_synthesis_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        try:
            if text_chunk:
                start = time.time()
                wav = tts_instance.tts(text_chunk, speaker="p230", speed=1.1)
                synthesis_time = time.time() - start
                
                print(f"⏱️ Worker {worker_id} TTS synthesis took {synthesis_time:.2f}s for chunk {sequence}: {text_chunk!r}")
                
                if isinstance(wav, list):
                    wav = np.array(wav, dtype=np.float32)
                
                max_val = np.max(np.abs(wav))
                if max_val > 0:
                    wav = wav / max_val * 0.95
                
                if is_final and len(wav) > 100:
                    wav[-50:] *= np.linspace(1, 0, 50)
                
                audio_chunk_queue.put((sequence, wav, is_final))
            else:
                audio_chunk_queue.put((sequence, np.array([], dtype=np.float32), is_final))
            
        except Exception as e:
            print(f"💥 Worker {worker_id} TTS Error:", e)
            audio_chunk_queue.put((sequence, np.array([], dtype=np.float32), is_final))
            
        tts_synthesis_queue.task_done()

def audio_callback(outdata, frames, time_info, status):
    global audio_buffer, playback_active
    outdata[:, 0] = 0
    
    with audio_stream_lock:
        available = len(audio_buffer)
        if available == 0:
            return
        
        to_copy = min(frames, available)
        outdata[:to_copy, 0] = audio_buffer[:to_copy]
        audio_buffer = audio_buffer[to_copy:]

def continuous_audio_feeder():
    global audio_buffer, playback_active
    
    while not stop_event.is_set():
        chunk_store = {}
        current_sequence = 0
        prebuffer_complete = False
        response_finished = False
        final_chunk_received = False

        while not playback_active and not stop_event.is_set():
            time.sleep(0.01)
        
        if stop_event.is_set():
            break
            
        print("🎵 Starting new response processing...")

        while not response_finished and not stop_event.is_set():
            try:
                sequence, audio_chunk, is_final = audio_chunk_queue.get(timeout=2.0)
                
                chunk_store[sequence] = (audio_chunk, is_final)
                
                if is_final:
                    final_chunk_received = True
                    print(f"🏁 Final chunk received: {sequence}")
                
                while current_sequence in chunk_store:
                    chunk_audio, chunk_is_final = chunk_store.pop(current_sequence)
                    
                    if len(chunk_audio) > 0:
                        with audio_stream_lock:
                            audio_buffer = np.concatenate([audio_buffer, chunk_audio])
                        
                        print(f"🔊 Added chunk {current_sequence} to buffer (size: {len(audio_buffer)} samples)")
                        
                        if not prebuffer_complete:
                            buffer_duration_sec = len(audio_buffer) / VITS_SR
                            if buffer_duration_sec >= prebuffer_threshold:
                                print(f"▶️ Prebuffer complete ({buffer_duration_sec:.2f}s). Audio ready for playback.")
                                prebuffer_complete = True
                    
                    current_sequence += 1
                    
                    if chunk_is_final:
                        print(f"🏁 Final chunk processed: {current_sequence - 1}")
                        remaining_wait = 0
                        while remaining_wait < 10 and not stop_event.is_set():
                            if chunk_store:
                                remaining_sequences = sorted(chunk_store.keys())
                                for seq in remaining_sequences:
                                    chunk_audio, _ = chunk_store.pop(seq)
                                    if len(chunk_audio) > 0:
                                        with audio_stream_lock:
                                            audio_buffer = np.concatenate([audio_buffer, chunk_audio])
                                        print(f"🔊 Added remaining chunk {seq} to buffer")
                                break
                            time.sleep(0.1)
                            remaining_wait += 1
                        
                        response_finished = True
                        break
                
                audio_chunk_queue.task_done()
                
            except queue.Empty:
                if final_chunk_received:
                    buffer_empty_count = 0
                    while buffer_empty_count < 20 and not stop_event.is_set():
                        with audio_stream_lock:
                            buffer_size = len(audio_buffer)
                        
                        if buffer_size == 0:
                            buffer_empty_count += 1
                        else:
                            buffer_empty_count = 0
                        
                        time.sleep(0.1)
                    
                    print("🏁 Response playback complete.")
                    response_finished = True
                    playback_active = False
                    break
                
                continue
        
        remaining_chunks = 0
        try:
            while True:
                audio_chunk_queue.get_nowait()
                audio_chunk_queue.task_done()
                remaining_chunks += 1
        except queue.Empty:
            pass
        
        if remaining_chunks > 0:
            print(f"🧹 Cleaned up {remaining_chunks} remaining chunks")

def main():
    global current_audio_stream

    threads = [
        Thread(target=transcription_worker, daemon=True),
        Thread(target=llm_worker, daemon=True),
        Thread(target=continuous_audio_feeder, daemon=True),
    ]

    for i in range(NUM_TTS_WORKERS):
        threads.append(Thread(target=tts_synthesis_worker, args=(i,), daemon=True))

    for t in threads:
        t.start()

    current_audio_stream = sd.OutputStream(
        samplerate=VITS_SR,
        channels=1,
        dtype='float32',
        callback=audio_callback,
        blocksize=2048,
        latency='low'
    )
    current_audio_stream.start()
    print("▶️ Audio playback system ready.")

    print("✅ Ready. Say 'Jarvis' to activate.")
    if jarvis_agent:
        print(f"🤖 {jarvis_agent.get_skill_info()}")

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

        for _ in range(50):
            if (audio_queue.empty() and transcription_queue.empty() and 
                tts_synthesis_queue.empty() and audio_chunk_queue.empty()):
                break
            time.sleep(0.1)

        audio_stream.stop_stream()
        audio_stream.close()
        pa.terminate()
        porcupine.delete()

if __name__ == "__main__":
    main()