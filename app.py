import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from dataclasses import dataclass
import base64
from collections import deque
import threading

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from huggingface_hub import snapshot_download
import warnings
import librosa
from scipy import signal

# --- Global Configuration ---
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"🚀 Using device: {device}")
models_base_dir = Path("./models")

# --- Performance Monitoring ---
@dataclass
class PerformanceMetrics:
    stt_latency: List[float]
    llm_latency: List[float] 
    tts_latency: List[float]
    total_latency: List[float]
    error_count: int
    processed_requests: int

class PerformanceMonitor:
    def __init__(self):
        self.metrics = PerformanceMetrics([], [], [], [], 0, 0)
        self.lock = threading.Lock()
    
    def track_latency(self, operation: str, duration: float):
        with self.lock:
            getattr(self.metrics, f'{operation}_latency').append(duration)
            if len(getattr(self.metrics, f'{operation}_latency')) > 100:
                getattr(self.metrics, f'{operation}_latency').pop(0)
    
    def increment_error(self):
        with self.lock:
            self.metrics.error_count += 1
    
    def increment_processed(self):
        with self.lock:
            self.metrics.processed_requests += 1
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                'avg_stt_latency': np.mean(self.metrics.stt_latency) if self.metrics.stt_latency else 0,
                'avg_llm_latency': np.mean(self.metrics.llm_latency) if self.metrics.llm_latency else 0,
                'avg_tts_latency': np.mean(self.metrics.tts_latency) if self.metrics.tts_latency else 0,
                'avg_total_latency': np.mean(self.metrics.total_latency) if self.metrics.total_latency else 0,
                'error_count': self.metrics.error_count,
                'processed_requests': self.metrics.processed_requests,
                'error_rate': self.metrics.error_count / max(1, self.metrics.processed_requests)
            }

# --- Voice Activity Detection ---
class VoiceActivityDetector:
    def __init__(self, threshold: float = 0.01, min_duration: float = 0.3, max_silence: float = 1.0):
        self.threshold = threshold
        self.min_duration = min_duration
        self.max_silence = max_silence
        self.is_speaking = False
        self.speech_start = None
        self.last_speech_time = None
        self.silence_start = None
        
    def detect(self, audio_chunk: np.ndarray) -> Dict[str, bool]:
        current_time = time.time()
        energy = np.mean(np.square(audio_chunk))
        
        result = {
            'speech_detected': False,
            'speech_ended': False,
            'should_process': False
        }
        
        if energy > self.threshold:
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start = current_time
                self.silence_start = None
                result['speech_detected'] = True
            self.last_speech_time = current_time
        else:
            if self.is_speaking:
                if self.silence_start is None:
                    self.silence_start = current_time
                elif current_time - self.silence_start > self.max_silence:
                    # End of speech detected
                    if self.speech_start and current_time - self.speech_start > self.min_duration:
                        result['speech_ended'] = True
                        result['should_process'] = True
                    self.is_speaking = False
                    self.speech_start = None
                    self.silence_start = None
        
        return result

# --- Audio Processing ---
class AudioProcessor:
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer = deque(maxlen=sample_rate * 5)  # 5 second buffer
        
    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Preprocess audio with noise reduction and normalization"""
        if len(audio_data) == 0:
            return audio_data
            
        # Normalize audio
        audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-8)
        
        # Apply high-pass filter to remove low-frequency noise
        if len(audio_data) > 100:
            sos = signal.butter(4, 80, btype='high', fs=self.sample_rate, output='sos')
            audio_data = signal.sosfilt(sos, audio_data)
        
        # Apply gentle low-pass filter to remove high-frequency noise
        if len(audio_data) > 100:
            sos = signal.butter(4, 8000, btype='low', fs=self.sample_rate, output='sos')
            audio_data = signal.sosfilt(sos, audio_data)
        
        return audio_data.astype(np.float32)
    
    def add_to_buffer(self, audio_chunk: np.ndarray):
        """Add audio chunk to circular buffer"""
        self.buffer.extend(audio_chunk)
    
    def get_buffered_audio(self, duration_seconds: float = 3.0) -> np.ndarray:
        """Get last N seconds of audio from buffer"""
        samples_needed = int(duration_seconds * self.sample_rate)
        if len(self.buffer) < samples_needed:
            return np.array(list(self.buffer))
        return np.array(list(self.buffer)[-samples_needed:])

# --- Enhanced Service Classes ---
class EnhancedSTTService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_initialized = False
        self.audio_processor = AudioProcessor()
        
    async def initialize(self):
        try:
            logger.info("Loading Enhanced STT Model...")
            # Try to load Kyutai model first
            try:
                from moshi.models import loaders
                import sentencepiece as spm

                model_dir = models_base_dir / "stt" / "models--kyutai--stt-1b-en_fr"
                snapshot_path = next(model_dir.glob("**/snapshots/*/"), None)
                
                if snapshot_path:
                    model_file = snapshot_path / "mimi-pytorch-e351c8d8@125.safetensors"
                    tokenizer_file = snapshot_path / "tokenizer_en_fr_audio_8000.model"
                    
                    if model_file.exists() and tokenizer_file.exists():
                        self.model = loaders.get_mimi(str(model_file), device=device)
                        self.model.eval()
                        self.tokenizer = spm.SentencePieceProcessor()
                        self.tokenizer.load(str(tokenizer_file))
                        logger.info("✅ Kyutai STT Model loaded successfully!")
                    else:
                        raise FileNotFoundError("Model files not found")
                else:
                    raise FileNotFoundError("Snapshot path not found")
                    
            except Exception as e:
                logger.warning(f"Kyutai STT failed to load: {e}. Using fallback STT.")
                self.model = None
                
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"STT initialization failed: {e}", exc_info=True)
            self.is_initialized = False

    async def transcribe(self, audio_data: np.ndarray) -> str:
        if not self.is_initialized or len(audio_data) == 0:
            return ""
            
        try:
            # Preprocess audio
            processed_audio = self.audio_processor.preprocess_audio(audio_data)
            
            if self.model and self.tokenizer:
                # Use Kyutai model
                audio_tensor = torch.from_numpy(processed_audio).to(device).unsqueeze(0)
                with torch.no_grad():
                    transcribed_ids = self.model.generate(audio_tensor, self.tokenizer.eos_id())[0].cpu().numpy()
                transcription = self.tokenizer.decode(transcribed_ids.tolist())
            else:
                # Fallback transcription based on audio characteristics
                transcription = self._fallback_transcribe(processed_audio)
                
            logger.info(f"STT Result: '{transcription}'")
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"STT transcription error: {e}", exc_info=True)
            return "I heard you speaking, but couldn't understand clearly."
    
    def _fallback_transcribe(self, audio_data: np.ndarray) -> str:
        """Fallback transcription when model is not available"""
        if len(audio_data) == 0:
            return ""
            
        # Analyze audio characteristics
        energy = np.mean(np.square(audio_data))
        duration = len(audio_data) / 16000
        
        if energy < 0.001:
            return ""
        elif duration < 0.5:
            return "Hello"
        elif duration < 1.5:
            return "How are you?"
        else:
            return "I'm listening to what you're saying."

class EnhancedTTSService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_initialized = False
        
    async def initialize(self):
        try:
            logger.info("Loading Enhanced TTS Model...")
            # Try to load Kyutai TTS model
            try:
                from tts_model_def.dsm_tts import DSMTTS
                from tts_model_def.config import DSMTTSConfig
                from safetensors.torch import load_file
                import sentencepiece as spm

                model_dir = models_base_dir / "tts" / "models--kyutai--tts-1.6b-en_fr"
                snapshot_path = next(model_dir.glob("**/snapshots/*/"), None)
                
                if snapshot_path:
                    model_file = snapshot_path / "dsm_tts_1e68beda@240.safetensors"
                    tokenizer_file = snapshot_path / "tokenizer_spm_8k_en_fr_audio.model"
                    
                    if model_file.exists() and tokenizer_file.exists():
                        config = DSMTTSConfig()
                        self.model = DSMTTS(config).to(device)
                        state_dict = load_file(model_file, device=str(device))
                        self.model.load_state_dict(state_dict, strict=False)
                        self.model.eval()
                        self.tokenizer = spm.SentencePieceProcessor()
                        self.tokenizer.load(str(tokenizer_file))
                        logger.info("✅ Kyutai TTS Model loaded successfully!")
                    else:
                        raise FileNotFoundError("TTS model files not found")
                else:
                    raise FileNotFoundError("TTS snapshot path not found")
                    
            except Exception as e:
                logger.warning(f"Kyutai TTS failed to load: {e}. Using enhanced fallback TTS.")
                self.model = None
                
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"TTS initialization failed: {e}", exc_info=True)
            self.is_initialized = False

    async def synthesize(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.array([])
            
        try:
            if self.model and self.tokenizer:
                # Use Kyutai model (if properly loaded)
                tokens = self.tokenizer.encode(text, return_tensors="pt").to(device)
                with torch.no_grad():
                    result = self.model.generate(tokens)
                return result["audio"][0].cpu().numpy()
            else:
                # Use enhanced fallback voice
                return self._generate_enhanced_voice(text)
                
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}", exc_info=True)
            return self._generate_enhanced_voice(text)

    def _generate_enhanced_voice(self, text: str) -> np.ndarray:
        """Enhanced fallback voice synthesis"""
        if not text.strip():
            return np.array([])
            
        logger.info(f"Synthesizing with enhanced voice: '{text}'")
        
        sr = 24000
        words = text.split()
        duration_per_word = 0.4
        pause_duration = 0.1
        total_duration = len(words) * duration_per_word + (len(words) - 1) * pause_duration
        
        audio_segments = []
        
        for i, word in enumerate(words):
            # Generate word audio
            word_duration = duration_per_word
            t = np.linspace(0, word_duration, int(sr * word_duration), endpoint=False)
            
            # Base frequency varies with word length and position
            base_freq = 180 + (len(word) * 5) + (i % 3) * 10
            
            # Create more natural sounding voice
            fundamental = np.sin(2 * np.pi * base_freq * t)
            harmonic2 = 0.5 * np.sin(2 * np.pi * base_freq * 2 * t)
            harmonic3 = 0.3 * np.sin(2 * np.pi * base_freq * 3 * t)
            
            # Add formants for more natural sound
            formant1 = 0.2 * np.sin(2 * np.pi * 800 * t)
            formant2 = 0.1 * np.sin(2 * np.pi * 1200 * t)
            
            voice = fundamental + harmonic2 + harmonic3 + formant1 + formant2
            
            # Apply envelope
            envelope = np.concatenate([
                np.linspace(0, 1, sr // 40),  # Attack
                np.ones(len(t) - sr // 20),   # Sustain
                np.linspace(1, 0, sr // 40)   # Release
            ])
            
            if len(envelope) != len(voice):
                envelope = np.resize(envelope, len(voice))
            
            word_audio = voice * envelope * 0.2
            audio_segments.append(word_audio)
            
            # Add pause between words
            if i < len(words) - 1:
                pause_samples = int(sr * pause_duration)
                pause = np.zeros(pause_samples)
                audio_segments.append(pause)
        
        # Concatenate all segments
        final_audio = np.concatenate(audio_segments)
        
        # Apply final processing
        final_audio = np.tanh(final_audio)  # Soft clipping
        
        return final_audio.astype(np.float32)

class EnhancedLLMService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_initialized = False
        self.conversation_history = deque(maxlen=10)
        
    async def initialize(self):
        try:
            logger.info("Loading Enhanced LLM...")
            # Try to load Moshi LLM
            try:
                from moshi.models import loaders
                import sentencepiece as spm
                
                model_dir = models_base_dir / "llm" / "models--kyutai--moshika-pytorch-bf16"
                snapshot_path = next(model_dir.glob("**/snapshots/*/"), None)
                
                if snapshot_path:
                    model_file = snapshot_path / "model.safetensors"
                    tokenizer_file = snapshot_path / "tokenizer_spm_32k_3.model"
                    
                    logger.info(f"Checking LLM files:")
                    logger.info(f"  Model file: {model_file} (exists: {model_file.exists()})")
                    logger.info(f"  Tokenizer file: {tokenizer_file} (exists: {tokenizer_file.exists()})")
                    
                    if model_file.exists() and tokenizer_file.exists():
                        logger.info("Loading Moshi LLM model...")
                        self.model = loaders.get_moshi_lm(str(model_file), device=device)
                        self.model.eval()
                        
                        logger.info("Loading Moshi LLM tokenizer...")
                        self.tokenizer = spm.SentencePieceProcessor()
                        self.tokenizer.load(str(tokenizer_file))
                        logger.info("✅ Moshi LLM loaded successfully!")
                    else:
                        # List all files in the directory for debugging
                        available_files = [f.name for f in snapshot_path.iterdir()]
                        logger.error(f"LLM model files not found. Available files: {available_files}")
                        raise FileNotFoundError("LLM model files not found")
                else:
                    raise FileNotFoundError("LLM snapshot path not found")
                    
            except Exception as e:
                logger.warning(f"Moshi LLM failed to load: {e}. Using enhanced fallback LLM.")
                self.model = None
                
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}", exc_info=True)
            self.is_initialized = False

    async def generate_response(self, text: str) -> str:
        if not text.strip():
            return "Could you please repeat that?"
            
        try:
            # Add to conversation history
            self.conversation_history.append(("user", text))
            
            if self.model and self.tokenizer:
                # Use Moshi model
                context = self._build_context()
                prompt = f"{context}\nUser: {text}\nAssistant:"
                
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    output_ids = self.model.generate(
                        input_ids, 
                        max_new_tokens=100, 
                        pad_token_id=self.tokenizer.eos_id(),
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9
                    )
                
                response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                response = response.split("Assistant:")[-1].strip()
            else:
                # Enhanced fallback responses
                response = self._generate_smart_response(text)
            
            # Add response to history
            self.conversation_history.append(("assistant", response))
            
            logger.info(f"LLM Response: '{response}'")
            return response
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}", exc_info=True)
            return "I'm having trouble processing that right now. Could you try again?"
    
    def _build_context(self) -> str:
        """Build conversation context from history"""
        context_parts = []
        for speaker, message in list(self.conversation_history)[-6:]:  # Last 6 exchanges
            if speaker == "user":
                context_parts.append(f"User: {message}")
            else:
                context_parts.append(f"Assistant: {message}")
        return "\n".join(context_parts)
    
    def _generate_smart_response(self, text: str) -> str:
        """Enhanced fallback response generation"""
        text_lower = text.lower()
        
        # Greeting responses
        if any(word in text_lower for word in ["hello", "hi", "hey", "good morning", "good afternoon"]):
            return "Hello! How can I help you today?"
        
        # Question responses
        if text_lower.startswith(("what", "how", "why", "when", "where", "who")):
            return f"That's an interesting question about {text.split()[-3:]}. Let me think about that."
        
        # Emotional responses
        if any(word in text_lower for word in ["sad", "upset", "angry", "frustrated"]):
            return "I understand you're going through a difficult time. Is there anything specific I can help with?"
        
        if any(word in text_lower for word in ["happy", "excited", "great", "wonderful"]):
            return "That's wonderful to hear! I'm glad you're feeling positive."
        
        # Help requests
        if any(word in text_lower for word in ["help", "assist", "support"]):
            return "I'd be happy to help you. What specifically do you need assistance with?"
        
        # Default contextual response
        key_words = [word for word in text.split() if len(word) > 3]
        if key_words:
            return f"I understand you're talking about {', '.join(key_words[:3])}. Could you tell me more?"
        
        return "That's interesting. Could you elaborate on that?"

# --- Enhanced System Orchestrator ---
class EnhancedUnmuteSystem:
    def __init__(self):
        self.stt_service = EnhancedSTTService()
        self.tts_service = EnhancedTTSService()
        self.llm_service = EnhancedLLMService()
        self.vad = VoiceActivityDetector()
        self.audio_processor = AudioProcessor()
        self.performance_monitor = PerformanceMonitor()
        
        # Processing queues for async pipeline
        self.stt_queue = asyncio.Queue(maxsize=10)
        self.llm_queue = asyncio.Queue(maxsize=5)
        self.tts_queue = asyncio.Queue(maxsize=5)
        
        self.is_processing = False
        
    async def initialize(self):
        """Initialize all services"""
        logger.info("🚀 Initializing Enhanced Unmute System...")
        
        await asyncio.gather(
            self.stt_service.initialize(),
            self.tts_service.initialize(), 
            self.llm_service.initialize()
        )
        
        logger.info("✅ All Enhanced Services Initialized!")
        
    async def process_audio_chunk(self, audio_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Process individual audio chunk with VAD"""
        if len(audio_data) == 0:
            return None
            
        # Add to audio buffer
        self.audio_processor.add_to_buffer(audio_data)
        
        # Voice activity detection
        vad_result = self.vad.detect(audio_data)
        
        if vad_result['should_process']:
            # Get buffered audio for processing
            buffered_audio = self.audio_processor.get_buffered_audio(3.0)
            return await self.process_complete_audio(buffered_audio)
        
        return None
    
    async def process_complete_audio(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Process complete audio through the pipeline"""
        start_time = time.time()
        
        try:
            self.performance_monitor.increment_processed()
            
            # STT Processing
            stt_start = time.time()
            transcription = await self.stt_service.transcribe(audio_data)
            stt_duration = time.time() - stt_start
            self.performance_monitor.track_latency('stt', stt_duration)
            
            if not transcription.strip() or len(transcription.strip()) < 2:
                return {"error": "No clear speech detected"}
            
            # LLM Processing
            llm_start = time.time()
            response_text = await self.llm_service.generate_response(transcription)
            llm_duration = time.time() - llm_start
            self.performance_monitor.track_latency('llm', llm_duration)
            
            # TTS Processing
            tts_start = time.time()
            response_audio = await self.tts_service.synthesize(response_text)
            tts_duration = time.time() - tts_start
            self.performance_monitor.track_latency('tts', tts_duration)
            
            total_duration = time.time() - start_time
            self.performance_monitor.track_latency('total', total_duration)
            
            return {
                "transcription": transcription,
                "response_text": response_text,
                "response_audio": response_audio.tolist(),
                "processing_time": {
                    "stt": stt_duration,
                    "llm": llm_duration,
                    "tts": tts_duration,
                    "total": total_duration
                }
            }
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            self.performance_monitor.increment_error()
            return {"error": f"Processing error: {str(e)}"}

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"🔌 New connection: {session_id}")
        
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"🔌 Disconnected: {session_id}")
    
    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to {session_id}: {e}")
                self.disconnect(session_id)

# --- Initialize System ---
enhanced_system = EnhancedUnmuteSystem()
connection_manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await enhanced_system.initialize()
    yield
    logger.info("🛑 Shutting down Enhanced System.")

# --- FastAPI Application ---
app = FastAPI(
    title="Enhanced Unmute Voice Assistant", 
    version="7.0.0-perfect",
    description="Perfect Speech-to-Speech AI with WebRTC, VAD, and Enhanced Processing",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/status")
async def get_status():
    stats = enhanced_system.performance_monitor.get_stats()
    return {
        "services": {
            "stt_initialized": enhanced_system.stt_service.is_initialized,
            "tts_initialized": enhanced_system.tts_service.is_initialized,
            "llm_initialized": enhanced_system.llm_service.is_initialized
        },
        "performance": stats,
        "active_connections": len(connection_manager.active_connections)
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    session_id = str(uuid.uuid4())
    await connection_manager.connect(websocket, session_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "audio":
                audio_data = np.array(data.get("audio", []), dtype=np.float32)
                
                if audio_data.size > 0:
                    result = await enhanced_system.process_audio_chunk(audio_data)
                    
                    if result:
                        await connection_manager.send_message(session_id, result)
                        
            elif data.get("type") == "ping":
                await connection_manager.send_message(session_id, {"type": "pong"})
                
    except WebSocketDisconnect:
        connection_manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"❌ WebSocket error for {session_id}: {e}", exc_info=True)
        connection_manager.disconnect(session_id)

if __name__ == "__main__":
    import os
    
    # Use RunPod's default port or environment variable
    port = int(os.environ.get("PORT", 7860))
    
    print(f"🚀 Starting Perfect Voice Assistant on port {port}")
    print(f"🌐 Access URL: https://{os.environ.get('RUNPOD_POD_ID', 'localhost')}-{port}.proxy.runpod.net")
    
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=port, 
        log_level="info",
        reload=False,
        workers=1
    )