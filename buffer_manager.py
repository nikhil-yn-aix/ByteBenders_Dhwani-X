import numpy as np
import sounddevice as sd
import threading
import queue
from collections import deque
from typing import Optional, Callable
import logging
import os
import time

from config import Config


class BufferManager:

    def __init__(self, config: Config = None, callback: Optional[Callable] = None):
        self.config = config or Config.default()

        
        self.process_callback = callback

        
        self.audio_queue = queue.Queue(maxsize=self.config.buffer.queue_maxsize)
        self.error_queue = queue.Queue()  

        
        self.ring_buffer = deque(maxlen=self.config.audio.context_samples)
        self.speech_buffer = []
        self.in_speech = False

        
        self.total_chunks_received = 0
        self.total_chunks_processed = 0

        
        self.is_recording = False
        self.audio_stream = None
        self.worker_thread = None

        
        self.logger = logging.getLogger("BufferManager")
        if not self.logger.handlers:
            os.makedirs("logs", exist_ok=True)
            fh = logging.FileHandler("logs/buffer_manager.log", encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(fh)
            self.logger.setLevel(logging.INFO)

    
    
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Audio callback MUST NEVER crash or print."""
        try:
            if status:
                try:
                    self.error_queue.put_nowait(("audio_status", str(status)))
                except Exception:
                    pass

            chunk = indata.copy()

            
            try:
                self.audio_queue.put_nowait(chunk)
                self.total_chunks_received += 1
            except queue.Full:
                self.logger.warning("Audio queue full — dropping chunk")
                return

            
            cb = self.process_callback
            if cb is not None:
                try:
                    context = np.array(self.ring_buffer) if len(self.ring_buffer) else np.array([])
                    cb(chunk.flatten(), context)
                except Exception as cb_e:
                    try:
                        self.error_queue.put_nowait(("audio_callback", repr(cb_e)))
                    except Exception:
                        pass

        except Exception as fatal_e:
            try:
                self.error_queue.put_nowait(("audio_callback_FATAL", repr(fatal_e)))
            except Exception:
                pass

    
    
    
    def _processing_loop(self):
        """Consumes audio_queue chunks and updates ring buffer."""
        while self.is_recording:
            try:
                chunk = self.audio_queue.get(timeout=0.5)

                
                try:
                    self.ring_buffer.extend(chunk.flatten())
                except Exception:
                    try:
                        self.error_queue.put_nowait(("processing_ringbuffer", "failed to extend ring buffer"))
                    except Exception:
                        pass

                
                cb = self.process_callback
                if cb is not None:
                    try:
                        context = np.array(self.ring_buffer)
                        cb(chunk.flatten(), context)
                    except Exception as cb_e:
                        try:
                            self.error_queue.put_nowait(("processing_callback", repr(cb_e)))
                        except Exception:
                            pass

                self.total_chunks_processed += 1

            except queue.Empty:
                continue
            except Exception as e:
                try:
                    self.error_queue.put_nowait(("processing_loop", repr(e)))
                except Exception:
                    pass

    
    
    
    def start_recording(self, device=None):
        """Starts microphone + worker thread safely."""
        if self.is_recording:
            return

        self.is_recording = True

        
        try:
            self.audio_stream = sd.InputStream(
                samplerate=self.config.audio.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
                blocksize=int(self.config.audio.chunk_samples),
                device=device
            )
            self.audio_stream.start()
        except Exception as e:
            self.is_recording = False
            raise RuntimeError(f"Failed to start microphone stream: {e}")

        
        self.worker_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.worker_thread.start()

        self.logger.info(f"Recording started (device={device})")

    def stop_recording(self):
        """Stops audio stream + thread safely."""
        if not self.is_recording:
            return

        self.is_recording = False

        
        try:
            if self.audio_stream:
                self.audio_stream.stop()
                self.audio_stream.close()
        except Exception as e:
            try:
                self.error_queue.put_nowait(("stop_stream", repr(e)))
            except Exception:
                pass

        
        if self.worker_thread:
            try:
                self.worker_thread.join(timeout=1.0)
            except Exception:
                pass

        self.logger.info("Recording stopped")

    
    
    
    def update_speech_state(self, is_speech: bool):
        """Tracks start/end of speech segments."""
        prev = self.in_speech
        self.in_speech = is_speech

        if not prev and is_speech:
            self.speech_buffer = []
            return "speech_started"

        if prev and not is_speech:
            return "utterance_complete"

        return "no_change"

    def add_to_speech_buffer(self, chunk):
        self.speech_buffer.extend(chunk.tolist())

    def get_complete_utterance(self):
        """Returns full utterance audio array."""
        if len(self.speech_buffer) == 0:
            return None
        return np.array(self.speech_buffer, dtype=np.float32)

    
    
    
    def get_buffer_stats(self):
        return {
            "chunks_received": self.total_chunks_received,
            "chunks_processed": self.total_chunks_processed,
            "ring_buffer_length": len(self.ring_buffer),
            "speech_buffer_length": len(self.speech_buffer)
        }

    
    
    
    @staticmethod
    def get_available_devices():
        """Returns first few microphone devices."""
        try:
            devices = sd.query_devices()
            mics = []
            for i, d in enumerate(devices):
                if d["max_input_channels"] > 0:
                    mics.append({"id": i, "name": d["name"]})
            return mics
        except Exception:
            return []
