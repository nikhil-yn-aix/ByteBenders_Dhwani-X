import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Union
from config import Config
from audio_utils import AudioUtils


class DenoiserProcessor:
    
    def __init__(self, config: Config = None):
        self.config = config or Config.default()
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        model = torch.hub.load(
            repo_or_dir='facebookresearch/denoiser',
            model='dns64',
            force_reload=False
        )
        self.model = model.to(self.device)
        self.model.eval()
    
    def denoise(
        self,
        audio: np.ndarray,
        sr: int = 16000
    ) -> np.ndarray:
        
        if sr != 16000:
            audio = AudioUtils.resample_audio(audio, sr, 16000)
            sr = 16000
        
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            denoised = self.model(audio_tensor)
        
        denoised_audio = denoised.squeeze().cpu().numpy()
        
        return denoised_audio
    
    def denoise_with_context(
        self,
        audio_chunk: np.ndarray,
        context: np.ndarray,
        sr: int = 16000
    ) -> np.ndarray:
        combined = np.concatenate([context, audio_chunk])
        
        denoised_combined = self.denoise(combined, sr)
        
        context_len = len(context)
        denoised_chunk = denoised_combined[context_len:]
        
        return denoised_chunk


if __name__ == "__main__":
    print("Testing DenoiserProcessor...")
    
    test_audio_path = Path("Nikhil_Indoor.mp3")
    if not test_audio_path.exists():
        print("✗ Test audio file not found: Nikhil_Indoor.mp3")
        exit(1)
    
    Path("test").mkdir(exist_ok=True)
    
    audio, sr = AudioUtils.load_audio(test_audio_path, sr=16000)
    print(f"✓ Loaded audio: {len(audio)/sr:.1f}s @ {sr}Hz")
    
    original_rms = AudioUtils.calculate_rms(audio)
    print(f"  Original RMS: {original_rms:.4f}")
    
    denoiser = DenoiserProcessor()
    print("✓ Demucs model loaded")
    
    denoised = denoiser.denoise(audio, sr)
    print("✓ Denoising complete")
    
    denoised_rms = AudioUtils.calculate_rms(denoised)
    print(f"  Denoised RMS: {denoised_rms:.4f}")
    print(f"  RMS change: {((denoised_rms / original_rms) - 1) * 100:.1f}%")
    
    AudioUtils.save_audio("test/denoised_full.wav", denoised, sr)
    print("✓ Saved denoised audio")
    
    chunk_size = sr * 3
    chunk = audio[:chunk_size]
    context = np.zeros(sr)
    
    denoised_chunk = denoiser.denoise_with_context(chunk, context, sr)
    AudioUtils.save_audio("test/denoised_chunk.wav", denoised_chunk, sr)
    print("✓ Chunk denoising test complete")
    
    print("\n✓ DenoiserProcessor working correctly")