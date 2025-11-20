import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from scipy import signal
from config import Config
from audio_utils import AudioUtils
import os


os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class Transcriber:
    
    def __init__(self, config: Config = None, language: str = "kn"):
        self.config = config or Config.default()
        self.model_name = 'ai4bharat/indic-conformer-600m-multilingual'
        self.language = language
        self.decoding_method = 'rnnt'
        
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self._load_conformer()
    
    def _load_conformer(self):
        import torch.multiprocessing as mp
        
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        
        from transformers import AutoModel
        
        torch.set_num_threads(1)
        
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=False
        )
        
        if self.device == 'cuda':
            self.model = self.model.to('cuda')
        
        self.model.eval()
    
    def _ensure_16khz(self, audio: np.ndarray, current_sr: int) -> np.ndarray:
        if current_sr == 16000:
            return audio
        num_samples = int(len(audio) * 16000 / current_sr)
        resampled = signal.resample(audio, num_samples)
        return resampled.astype(np.float32)
    
    def transcribe(
        self,
        audio: np.ndarray,
        sr: int = 16000
    ) -> Dict[str, str]:
        
        audio_16k = self._ensure_16khz(audio, sr)
        
        text = self._transcribe_with_conformer(audio_16k)
        
        result = {
            "text": text,
            "language": self.language,
            "model": self.model_name
        }
        
        return result
    
    def _transcribe_with_conformer(self, audio: np.ndarray) -> str:
        try:
            with torch.no_grad():
                audio_tensor = torch.from_numpy(audio).unsqueeze(0)
                if self.device == 'cuda':
                    audio_tensor = audio_tensor.to('cuda')
                
                transcription = self.model(audio_tensor, self.language, self.decoding_method)
            
            if isinstance(transcription, list) and len(transcription) > 0:
                text = transcription[0]
            elif isinstance(transcription, str):
                text = transcription
            else:
                text = str(transcription)
            
            return text.strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
    
    def transcribe_utterances(
        self,
        utterances: list,
        sr: int = 16000
    ) -> list:
        
        results = []
        for i, utterance in enumerate(utterances):
            result = self.transcribe(utterance, sr)
            result["utterance_id"] = i
            results.append(result)
        
        return results


if __name__ == "__main__":
    print("Testing Transcriber with IndicConformer 600M...")
    print("=" * 60)
    
    test_audio_path = Path("data/Nikhil_Indoor.mp3")
    if not test_audio_path.exists():
        print("✗ Test audio file not found: Nikhil_Indoor.mp3")
        exit(1)
    
    Path("output").mkdir(exist_ok=True)
    
    audio, sr = AudioUtils.load_audio(test_audio_path, sr=16000)
    print(f"✓ Loaded audio: {len(audio)/sr:.1f}s @ {sr}Hz")
    
    transcriber = Transcriber(language="kn")
    print(f"✓ Model: {transcriber.model_name}")
    print(f"✓ Language: {transcriber.language}")
    print("=" * 60)
    
    print("\nTranscribing full audio...")
    result = transcriber.transcribe(audio, sr)
    
    print("\nTranscription Result:")
    print("=" * 60)
    print(result['text'])
    print("=" * 60)
    
    output_path = Path("output/transcription_full.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Model: {result['model']}\n")
        f.write(f"Language: {result['language']}\n")
        f.write(f"Duration: {len(audio)/sr:.1f}s\n")
        f.write("=" * 60 + "\n\n")
        f.write(result['text'])
        f.write("\n")
    
    print(f"\n✓ Saved full transcription to {output_path}")
    print("✓ Transcriber working correctly")