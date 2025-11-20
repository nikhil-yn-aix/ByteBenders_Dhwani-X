import numpy as np
import time
from pathlib import Path
from collections import deque
from typing import Optional
import sys
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from config import Config
from buffer_manager import BufferManager
from vad_processor import VADProcessor
from noise_classifier import NoiseClassifier
from transcriber import Transcriber


class LivePipeline:
    
    def __init__(self, config: Config = None):
        self.config = config or Config.default()
        
        print("="*70)
        print("INITIALIZING LIVE KANNADA SPEECH PIPELINE")
        print("="*70)
        
        try:
            print("\n[1/3] Loading VAD model...", end=" ", flush=True)
            self.vad = VADProcessor(config)
            print("✓")
        except Exception as e:
            print(f"✗")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        try:
            print("[2/3] Loading Noise Classifier...", end=" ", flush=True)
            self.classifier = NoiseClassifier(config)
            print("✓")
        except Exception as e:
            print(f"✗")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        self.transcriber = None
        self.transcriber_loaded = False
        
        print("\n[*] Initializing buffer manager...", end=" ", flush=True)
        self.buffer_manager = BufferManager(config, callback=self._process_chunk)
        print("✓")
        
        self.last_noise_update = 0
        self.noise_update_interval = 5.0
        self.current_noise_type = "unknown"
        self.current_noise_confidence = 0.0
        
        self.utterance_count = 0
        self.total_speech_time = 0.0
        self.session_start = None
        
        self.noise_history = deque(maxlen=100)
        
        self.is_running = False
        
        print("\n" + "="*70)
        print("✓ Core components loaded!")
        print("  Transcriber will load on first use")
        print("="*70 + "\n")
    
    def _load_transcriber_lazy(self):
        if not self.transcriber_loaded:
            print("\n[*] Loading Transcriber (first use)...", end=" ", flush=True)
            try:
                self.transcriber = Transcriber(self.config, language="kn")
                self.transcriber_loaded = True
                print("✓\n")
            except Exception as e:
                print(f"✗")
                print(f"Error loading transcriber: {e}")
                import traceback
                traceback.print_exc()
                self.transcriber_loaded = False
    
    def _process_chunk(self, chunk: np.ndarray, context: np.ndarray):
        
        try:
            current_time = time.time()
            
            if current_time - self.last_noise_update >= self.noise_update_interval:
                self._update_noise_classification(context)
                self.last_noise_update = current_time
            
            speech_prob = self._check_speech_activity(chunk)
            
            state = self.buffer_manager.update_speech_state(speech_prob > 0.5)
            
            if speech_prob > 0.5:
                self.buffer_manager.add_to_speech_buffer(chunk)
                print("🎤", end="", flush=True)
            
            if state == "utterance_complete":
                print()
                self._process_utterance()
                
        except Exception as e:
            print(f"\n✗ Chunk processing error: {e}")
    
    def _check_speech_activity(self, chunk: np.ndarray) -> float:
        
        try:
            timestamps = self.vad.process_audio(chunk, self.config.audio.sample_rate)
            
            if timestamps:
                total_speech = sum(ts['end_sec'] - ts['start_sec'] for ts in timestamps)
                chunk_duration = len(chunk) / self.config.audio.sample_rate
                speech_prob = total_speech / chunk_duration if chunk_duration > 0 else 0.0
            else:
                speech_prob = 0.0
            
            return speech_prob
        except Exception as e:
            print(f"\n✗ VAD error: {e}")
            return 0.0
    
    def _update_noise_classification(self, audio: np.ndarray):
        
        try:
            result = self.classifier.analyze_background_noise(
                audio,
                self.config.audio.sample_rate
            )
            
            self.current_noise_type = result['category']
            self.current_noise_confidence = result['top_prediction']['confidence']
            
            self.noise_history.append({
                'timestamp': time.time() - self.session_start,
                'type': self.current_noise_type,
                'confidence': self.current_noise_confidence
            })
            
            print(f"\n\n{'─'*70}")
            print(f"🔊 BACKGROUND NOISE DETECTED")
            print(f"{'─'*70}")
            print(f"  Category: {self.current_noise_type.upper()}")
            print(f"  Confidence: {self.current_noise_confidence:.3f}")
            print(f"  Details: {result['top_prediction']['class']}")
            print(f"{'─'*70}\n")
            
        except Exception as e:
            print(f"\n✗ Noise classification error: {e}\n")
    
    def _process_utterance(self):
        
        try:
            utterance = self.buffer_manager.get_complete_utterance()
            
            if utterance is None or len(utterance) < self.config.audio.sample_rate * 0.3:
                return
            
            self.utterance_count += 1
            duration = len(utterance) / self.config.audio.sample_rate
            self.total_speech_time += duration
            
            print(f"\n{'='*70}")
            print(f"💬 UTTERANCE #{self.utterance_count}")
            print(f"{'='*70}")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Ambient noise: {self.current_noise_type}")
            
            if not self.transcriber_loaded:
                self._load_transcriber_lazy()
            
            if not self.transcriber_loaded:
                print("  ⚠ Transcriber not available, skipping transcription")
                print(f"{'='*70}\n")
                return
            
            print(f"\n  Transcribing...", end=" ", flush=True)
            
            start_time = time.time()
            
            result = self.transcriber.transcribe(
                utterance,
                self.config.audio.sample_rate
            )
            
            transcribe_time = time.time() - start_time
            rtf = transcribe_time / duration
            
            print("✓\n")
            print(f"  📝 Text: {result['text']}")
            print(f"  ⏱  Time: {transcribe_time:.2f}s (RTF: {rtf:.3f}x)")
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"✗\n  Error: {e}\n")
            import traceback
            traceback.print_exc()
    
    def start(self, device=None, duration: Optional[float] = None):
        
        print("="*70)
        print("🎙️  STARTING LIVE RECORDING")
        print("="*70)
        
        try:
            devices = BufferManager.get_available_devices()
            print(f"\nAvailable microphones:")
            for dev in devices[:5]:
                marker = " ← SELECTED" if dev['id'] == device else ""
                print(f"  [{dev['id']}] {dev['name']}{marker}")
        except Exception as e:
            print(f"Warning: Could not list devices - {e}")
        
        print(f"\nConfiguration:")
        print(f"  Sample rate: {self.config.audio.sample_rate}Hz")
        print(f"  Chunk duration: {self.config.audio.chunk_duration}s")
        print(f"  Noise updates: Every {self.noise_update_interval}s")
        
        print("\n" + "="*70)
        print("🎤 LISTENING...")
        print("="*70)
        if duration:
            print(f"Recording for {duration}s")
        else:
            print("Press Ctrl+C to stop")
        print("Speak into your microphone now!")
        print("="*70 + "\n")
        
        self.is_running = True
        self.session_start = time.time()
        self.utterance_count = 0
        self.total_speech_time = 0.0
        
        try:
            self.buffer_manager.start_recording(device=device)
            
            if duration:
                time.sleep(duration)
            else:
                while self.is_running:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\n\n⚠️  Recording interrupted by user")
        except Exception as e:
            print(f"\n✗ Recording error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
    
    def stop(self):
        
        if not self.is_running:
            return
        
        self.is_running = False
        self.buffer_manager.stop_recording()
        
        session_duration = time.time() - self.session_start if self.session_start else 0
        
        print("\n" + "="*70)
        print("📊 SESSION SUMMARY")
        print("="*70)
        print(f"  Total duration: {session_duration:.1f}s")
        print(f"  Utterances: {self.utterance_count}")
        print(f"  Speech time: {self.total_speech_time:.1f}s")
        
        if session_duration > 0:
            print(f"  Speech ratio: {(self.total_speech_time/session_duration)*100:.1f}%")
        
        if self.noise_history:
            noise_types = {}
            for entry in self.noise_history:
                noise_types[entry['type']] = noise_types.get(entry['type'], 0) + 1
            
            print(f"\n  Background noise detected:")
            for noise_type, count in sorted(noise_types.items(), key=lambda x: x[1], reverse=True):
                print(f"    • {noise_type}: {count}x")
        
        stats = self.buffer_manager.get_buffer_stats()
        print(f"\n  Audio chunks:")
        print(f"    Received: {stats['chunks_received']}")
        print(f"    Processed: {stats['chunks_processed']}")
        
        print("="*70 + "\n")
    
    def save_session_log(self, output_path: Path = None):
        
        if output_path is None:
            output_path = Path("output/live_session.json")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        session_data = {
            'session_duration_sec': time.time() - self.session_start if self.session_start else 0,
            'utterances_captured': self.utterance_count,
            'total_speech_time_sec': self.total_speech_time,
            'noise_history': list(self.noise_history),
            'buffer_stats': self.buffer_manager.get_buffer_stats()
        }
        
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Session log: {output_path}")


if __name__ == "__main__":
    
    print("\n🚀 KANNADA LIVE SPEECH TRANSCRIPTION")
    print("="*70)
    
    try:
        pipeline = LivePipeline()
    except Exception as e:
        print(f"\n✗ Initialization failed: {e}")
        sys.exit(1)
    
    devices = BufferManager.get_available_devices()
    
    print("\n📱 Select microphone:")
    for i, dev in enumerate(devices[:5]):
        print(f"  [{dev['id']}] {dev['name']}")
    
    try:
        device_input = input("\nDevice ID (Enter for default): ").strip()
        device_id = int(device_input) if device_input else None
    except:
        device_id = None
    
    try:
        duration_input = input("Duration in seconds (Enter for continuous): ").strip()
        duration = float(duration_input) if duration_input else None
    except:
        duration = None
    
    try:
        pipeline.start(device=device_id, duration=duration)
    finally:
        try:
            pipeline.save_session_log()
        except:
            pass
    
    print("\n✓ Done!")