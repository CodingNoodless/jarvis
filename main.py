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
import os
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import inspect
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import AgentAction, AgentFinish
import traceback
import re

warnings.filterwarnings("ignore", category=UserWarning)
SAMPLERATE = 16000

# Skills system configuration
SKILLS_DIR = "./skillss/"

class SkillManager:
    """Manages dynamic loading and reloading of skills"""
    
    def __init__(self, skills_dir: str = SKILLS_DIR):
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(exist_ok=True)
        self.loaded_skills: Dict[str, BaseTool] = {}
        self.skill_modules: Dict[str, Any] = {}
        self.last_modified: Dict[str, float] = {}
        self.lock = Lock()
        
        print(f"📁 Skills directory: {self.skills_dir.absolute()}")
        
    def _load_skill_from_file(self, skill_file: Path) -> Optional[BaseTool]:
        """Load a skill from a Python file"""
        try:
            spec = importlib.util.spec_from_file_location(
                skill_file.stem, skill_file
            )
            if spec is None or spec.loader is None:
                return None
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for the exported skill
            if hasattr(module, 'skill'):
                skill = module.skill
                if isinstance(skill, BaseTool):
                    return skill
                    
            print(f"⚠️ No valid skill found in {skill_file.name}")
            return None
            
        except Exception as e:
            print(f"❌ Error loading skill {skill_file.name}: {e}")
            return None
    
    def load_skills(self) -> List[BaseTool]:
        """Load all skills from the skills directory"""
        with self.lock:
            current_skills = []
            
            if not any(self.skills_dir.glob("*.py")):
                print("⚠️ No skill files found in skills directory")
                return current_skills
            
            for skill_file in self.skills_dir.glob("*.py"):
                if skill_file.name.startswith("_"):
                    continue
                    
                skill_name = skill_file.stem
                file_mtime = skill_file.stat().st_mtime
                
                # Check if we need to reload this skill
                if (skill_name not in self.loaded_skills or 
                    skill_name not in self.last_modified or
                    file_mtime > self.last_modified[skill_name]):
                    
                    print(f"🔄 Loading skill: {skill_name}")
                    skill = self._load_skill_from_file(skill_file)
                    
                    if skill:
                        self.loaded_skills[skill_name] = skill
                        self.last_modified[skill_name] = file_mtime
                        print(f"✅ Loaded skill: {skill.name}")
                    else:
                        # Remove failed skill
                        if skill_name in self.loaded_skills:
                            del self.loaded_skills[skill_name]
                        if skill_name in self.last_modified:
                            del self.last_modified[skill_name]
                
                # Add to current skills if loaded successfully
                if skill_name in self.loaded_skills:
                    current_skills.append(self.loaded_skills[skill_name])
            
            return current_skills
    
    def get_skill_info(self) -> str:
        """Get information about loaded skills"""
        with self.lock:
            if not self.loaded_skills:
                return "No skills loaded."
            
            info = "Loaded skills:\n"
            for skill in self.loaded_skills.values():
                info += f"- {skill.name}: {skill.description}\n"
            return info

class JarvisAgent:
    """Enhanced Jarvis with LangChain agent capabilities"""
    
    def __init__(self):
        self.skill_manager = SkillManager()
        self.llm = Ollama(
            model="llama3.2:3b-instruct-q4_K_M",
            base_url="http://localhost:11434"
        )
        self.memory = ConversationBufferWindowMemory(
            k=10, 
            return_messages=True,
            memory_key="chat_history",
            input_key="input"
        )
        self.agent_executor = None
        self._setup_agent()
    
    def _setup_agent(self):
        """Setup the LangChain agent with current skills"""
        try:
            skills = self.skill_manager.load_skills()
            
            if not skills:
                print("⚠️ No skills loaded, agent will work with basic capabilities only")
                return
            
            # Create a more focused prompt template
            template = """You are Jarvis, an AI assistant dedicated to serving Rohit Pulle. 
You have access to tools that can help you provide accurate information.
Always answer in the minimum number of steps. Do not repeat or rephrase the question.

Available tools:
{tools}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: I need to think about what information is needed
Action: [tool name]
Action Input: [input for the tool]
Observation: [result from the tool]
Thought: I now know the final answer
Final Answer: [the final answer to the question]

IMPORTANT RULES:
1. You MUST use the exact format above
2. Action Input must be on its own line
3. Only use tool names from the available tools: {tool_names}
4. After getting an Observation, provide a Final Answer
5. Do not repeat the same action multiple times
6. When using a tool, follow the instructions in the description and do not deviate, do not provide lengthy explanations or extra text.
7. Do NOT write code, pseudo-code, or describe steps. Just carry out the actions using the tools provided. Only use the Action and Action Input format as shown.
8. Never explain your reasoning or output, just use the tools are shown.
9. After you have used a tool and received an Observation that answers the user's question, you MUST immediately output:
Thought: I now know the final answer
Final Answer: [the answer]
and then STOP.
10. Do NOT ask or answer any new questions unless the user asks them.
11. Do NOT repeat the same Action or Action Input more than once per user question.
12. You must answer in as few steps as possible. If a tool gives you the answer, immediately output the Final Answer and stop.
13. If none of the available tools are relevant to the question, answer directly using your own knowledge. Do NOT use any tool in that case.
Question: {input}
Thought: {agent_scratchpad}"""

            prompt = PromptTemplate(
                template=template,
                input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
            )
            
            # Create agent executor with better error handling
            self.agent_executor = initialize_agent(
                tools=skills,
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                max_iterations=1,
                max_execution_time=30,
                early_stopping_method="generate",
                handle_parsing_errors=True,
                memory=self.memory,
                agent_kwargs={
                    "prefix": "You are Jarvis, an AI assistant dedicated to serving Rohit Pulle. Use the available tools to help answer questions accurately."
                }
            )
            
            print(f"🤖 Agent initialized with {len(skills)} skills")
            
        except Exception as e:
            print(f"❌ Error setting up agent: {e}")
            traceback.print_exc()
    
    def reload_skills(self):
        """Reload skills and reinitialize agent"""
        print("🔄 Reloading skills...")
        self._setup_agent()
    
    def process_query(self, query: str) -> str:
        """Process a query using the agent or fallback to basic LLM"""
        try:
            # Check if we need to reload skills
            current_skills = self.skill_manager.load_skills()
            if len(current_skills) != len(self.skill_manager.loaded_skills):
                self.reload_skills()
            
            # If we have an agent, use it
            if self.agent_executor:
                try:
                    # Clean the query to avoid parsing issues
                    cleaned_query = query.strip()
                    
                    print(f"🤖 Processing query: {cleaned_query}")
                    response = self.agent_executor.invoke({"input": cleaned_query})
                    
                    # Extract the output properly
                    if isinstance(response, dict):
                        if "output" in response:
                            return response["output"]
                        elif "result" in response:
                            return response["result"]
                        else:
                            return str(response)
                    else:
                        return str(response)
                        
                except Exception as e:
                    print(f"⚠️ Agent execution failed: {e}")
                    traceback.print_exc()
                    # Fallback to basic LLM
                    return self._basic_llm_response(query)
            else:
                return self._basic_llm_response(query)
                
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            return "I apologize, I encountered an error processing your request."
    
    def _basic_llm_response(self, query: str) -> str:
        """Fallback to basic LLM without skills"""
        try:
            response = self.llm.invoke(f"You are Jarvis, an AI dedicated to serving Rohit Pulle. Provide a clear, complete and concise answer to: {query}")
            return response
        except Exception as e:
            print(f"❌ Basic LLM failed: {e}")
            return "I apologize, I'm having trouble processing your request right now."

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
except Exception as e:
    print(f"❌ Failed to initialize Jarvis agent: {e}")
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
    """Modified LLM worker to handle agent responses better"""
    chunk_end_chars = {'.', '!', '?', ',', ';', ':', '—', '–', '-'}
    
    def find_punctuation_chunk(text, chunk_sequence=0):
        text = text.strip()
        
        if not text:
            return None
        
        min_length = 10 if chunk_sequence == 0 else 15
        
        if len(text) < min_length and chunk_sequence > 0:
            return None
        
        for i, char in enumerate(text):
            if char in chunk_end_chars and i >= min_length:
                end_pos = i + 1
                while end_pos < len(text) and text[end_pos] in ' \t':
                    end_pos += 1
                return text[:end_pos].strip()
        
        if len(text) > 100:
            for i in range(80, len(text)):
                if text[i] == ' ':
                    return text[:i].strip()
        
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
        chunk_sequence = 0
        
        global playback_active
        response_complete_event.clear()
        with audio_stream_lock:
            global audio_buffer
            audio_buffer = np.array([], dtype=np.float32)
            playback_active = True
        
        try:
            # Use the enhanced Jarvis agent if available
            if jarvis_agent:
                # Use streaming invoke
                response_chunks = jarvis_agent.agent_executor.stream({"input": prompt})
                full_output = ""
                final_answer_found = False
                for chunk in response_chunks:
                    text = chunk.get("output") or chunk.get("result") or str(chunk)
                    if not text:
                        continue
                    full_output += text + " "
                    # Check for Final Answer
                    if "Final Answer:" in full_output:
                        # Extract only the last Final Answer (handles repeated or noisy output)
                        parts = full_output.rsplit("Final Answer:", 1)
                        answer = parts[-1].strip()
                        # Optionally, remove anything after a new "Question:" or agent log
                        answer = answer.split("Question:")[0].strip()
                        # Remove any trailing curly braces or agent logs
                        answer = answer.split("}")[-1].strip() if "}" in answer else answer
                        if answer:
                            print(f"📝 Streaming final answer: {answer!r}")
                            tts_synthesis_queue.put((0, 0, answer, True))
                            final_answer_found = True
                            break  # Stop processing further chunks
                if not final_answer_found:
                    # Normalize whitespace and newlines for robust matching
                    normalized = full_output.replace('\n', ' ').replace('\r', ' ')
                    # Try to extract the last Final Answer
                    fa_match = re.findall(r"Final Answer:\s*(.*?)(?=(?:Question:|Thought:|Action:|Action Input:|Observation:|$))", normalized, re.DOTALL)
                    if fa_match:
                        answer = fa_match[-1].strip()
                    else:
                        # Try to extract the last Observation
                        obs_match = re.findall(r"Observation:\s*(.*?)(?=(?:Question:|Thought:|Action:|Action Input:|Final Answer:|$))", normalized, re.DOTALL)
                        if obs_match:
                            answer = obs_match[-1].strip()
                        else:
                            # Try to extract the last human sentence
                            cleaned = re.sub(r"\{.*?\}", "", normalized)
                            cleaned = re.sub(r"(Question:|Thought:|Action:|Action Input:).*", "", cleaned)
                            cleaned = re.sub(r",?\s*response_metadata=.*", "", cleaned)
                            sentences = re.findall(r"([A-Z][^\.!?]*[\.!?])", cleaned)
                            if sentences:
                                answer = sentences[-1].strip()
                            else:
                                answer = cleaned.strip()
                    # If answer is still empty or looks like junk, use a default message
                    if not answer or "response_metadata" in answer or answer.startswith(","):
                        answer = "Sorry, I could not extract a valid answer."
                    print(f"📝 Fallback streaming answer: {answer!r}")
                    tts_synthesis_queue.put((0, 0, answer, True))
            else:
                # Fallback to original streaming approach
                # [Original streaming code remains the same]
                pass

            print(f"⏱️ LLM call finished in {time.time() - start:.2f}s")

        except Exception as e:
            print("LLM request error:", e)
            traceback.print_exc()
            tts_synthesis_queue.put((0, 0, "Sorry, I encountered an error.", True))

        transcription_queue.task_done()

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
        print(f"🤖 {jarvis_agent.skill_manager.get_skill_info()}")

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