import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from config import Config
from audio_utils import AudioUtils


class VADProcessor:
    
    def __init__(self, config: Config = None):
        self.config = config or Config.default()
        self.model = None
        self.utils = None
        self._load_model()
    
    def _load_model(self):
        torch.hub.set_dir(str(self.config.paths.models_cache_dir))
        
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
            verbose=False,
            trust_repo=True
        )
        self.model = model
        self.utils = utils
    
    def process_audio(
        self,
        audio: np.ndarray,
        sr: int = 16000
    ) -> List[Dict]:
        if sr != 16000:
            audio = AudioUtils.resample_audio(audio, sr, 16000)
            sr = 16000
        
        audio_tensor = torch.from_numpy(audio).float()
        
        speech_timestamps = self.utils[0](
            audio_tensor,
            self.model,
            sampling_rate=sr,
            threshold=self.config.audio.vad_threshold,
            min_speech_duration_ms=int(self.config.audio.vad_min_speech_duration * 1000),
            min_silence_duration_ms=int(self.config.audio.vad_min_silence_duration * 1000)
        )
        
        timestamps = []
        for ts in speech_timestamps:
            timestamps.append({
                'start': int(ts['start']),
                'end': int(ts['end']),
                'start_sec': ts['start'] / sr,
                'end_sec': ts['end'] / sr
            })
        
        return timestamps
    
    def extract_speech_segments(
        self,
        audio: np.ndarray,
        timestamps: List[Dict]
    ) -> List[np.ndarray]:
        segments = []
        for ts in timestamps:
            segment = audio[ts['start']:ts['end']]
            segments.append(segment)
        return segments
    
    def extract_silence_segments(
        self,
        audio: np.ndarray,
        timestamps: List[Dict],
        sr: int = 16000
    ) -> List[np.ndarray]:
        silence_segments = []
        prev_end = 0
        
        for ts in timestamps:
            if ts['start'] > prev_end:
                silence = audio[prev_end:ts['start']]
                if len(silence) > sr * 0.1:
                    silence_segments.append(silence)
            prev_end = ts['end']
        
        if prev_end < len(audio):
            final_silence = audio[prev_end:]
            if len(final_silence) > sr * 0.1:
                silence_segments.append(final_silence)
        
        return silence_segments
    
    def get_speech_ratio(
        self,
        timestamps: List[Dict],
        total_duration: float
    ) -> float:
        speech_duration = sum(ts['end_sec'] - ts['start_sec'] for ts in timestamps)
        return speech_duration / total_duration if total_duration > 0 else 0.0

if __name__ == "__main__":
    print("Testing VADProcessor...")
    
    test_audio_path = Path("data/Nikhil_Indoor.mp3")
    if not test_audio_path.exists():
        print("✗ Test audio file not found: Nikhil_Indoor.mp3")
        exit(1)
    
    Path("test").mkdir(exist_ok=True)
    
    audio, sr = AudioUtils.load_audio(test_audio_path, sr=16000)
    print(f"✓ Loaded audio: {len(audio)/sr:.1f}s @ {sr}Hz")
    
    vad = VADProcessor()
    print("✓ VAD model loaded")
    
    timestamps = vad.process_audio(audio, sr)
    print(f"✓ Found {len(timestamps)} speech segments")
    
    for i, ts in enumerate(timestamps[:3]):
        print(f"  Segment {i+1}: {ts['start_sec']:.2f}s - {ts['end_sec']:.2f}s")
    
    speech_segments = vad.extract_speech_segments(audio, timestamps)
    print(f"✓ Extracted {len(speech_segments)} speech segments")
    
    speech_only = np.concatenate(speech_segments)
    AudioUtils.save_audio("test/vad_speech_only.wav", speech_only, sr)
    print("✓ Saved speech-only audio")
    
    silence_segments = vad.extract_silence_segments(audio, timestamps, sr)
    print(f"✓ Found {len(silence_segments)} silence segments")
    
    if silence_segments:
        silence_only = np.concatenate(silence_segments)
        AudioUtils.save_audio("test/vad_silence_only.wav", silence_only, sr)
        print("✓ Saved silence-only audio")
    
    speech_ratio = vad.get_speech_ratio(timestamps, len(audio) / sr)
    print(f"✓ Speech ratio: {speech_ratio:.2%}")
    
    print("\n✓ VADProcessor working correctly")