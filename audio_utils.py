import numpy as np
import librosa
import soundfile as sf
import torch
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, List
from pesq import pesq
from pystoi import stoi
from jiwer import wer, cer
import json
import time


class AudioUtils:
    
    @staticmethod
    def load_audio(
        path: Union[str, Path],
        sr: int = 16000,
        mono: bool = True,
        duration: Optional[float] = None,
        offset: float = 0.0
    ) -> Tuple[np.ndarray, int]:
        audio, sample_rate = librosa.load(
            path,
            sr=sr,
            mono=mono,
            duration=duration,
            offset=offset
        )
        return audio, sample_rate
    
    @staticmethod
    def save_audio(
        path: Union[str, Path],
        audio: np.ndarray,
        sr: int
    ) -> None:
        sf.write(path, audio, sr)
    
    @staticmethod
    def resample_audio(
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    
    @staticmethod
    def normalize_audio(
        audio: np.ndarray,
        target_level: float = 0.95
    ) -> np.ndarray:
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio * (target_level / max_val)
        return audio
    
    @staticmethod
    def numpy_to_torch(
        audio: np.ndarray,
        device: str = "cpu"
    ) -> torch.Tensor:
        tensor = torch.from_numpy(audio).float()
        if len(tensor.shape) == 1:
            tensor = tensor.unsqueeze(0)
        return tensor.to(device)
    
    @staticmethod
    def torch_to_numpy(audio: torch.Tensor) -> np.ndarray:
        return audio.squeeze().cpu().numpy()
    
    @staticmethod
    def calculate_rms(signal: np.ndarray) -> float:
        return float(np.sqrt(np.mean(signal ** 2)))
    
    @staticmethod
    def calculate_snr(
        signal: np.ndarray,
        noise: np.ndarray
    ) -> float:
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power == 0:
            return float('inf')
        return float(10 * np.log10(signal_power / noise_power))
    
    @staticmethod
    def calculate_snr_from_speech_and_full(
        speech_segments: List[np.ndarray],
        full_audio: np.ndarray,
        speech_timestamps: List[Dict]
    ) -> float:
        speech = np.concatenate(speech_segments)
        
        noise_segments = []
        prev_end = 0
        for ts in speech_timestamps:
            if ts['start'] > prev_end:
                noise_segments.append(full_audio[prev_end:ts['start']])
            prev_end = ts['end']
        
        if len(noise_segments) == 0:
            return float('inf')
        
        noise = np.concatenate(noise_segments)
        
        return AudioUtils.calculate_snr(speech, noise)
    
    @staticmethod
    def calculate_pesq(
        reference: np.ndarray,
        degraded: np.ndarray,
        sr: int = 16000
    ) -> float:
        if sr not in [8000, 16000]:
            reference = librosa.resample(reference, orig_sr=sr, target_sr=16000)
            degraded = librosa.resample(degraded, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        mode = 'wb' if sr == 16000 else 'nb'
        
        min_len = min(len(reference), len(degraded))
        reference = reference[:min_len]
        degraded = degraded[:min_len]
        
        return float(pesq(sr, reference, degraded, mode))
    
    @staticmethod
    def calculate_stoi(
        reference: np.ndarray,
        degraded: np.ndarray,
        sr: int = 16000
    ) -> float:
        min_len = min(len(reference), len(degraded))
        reference = reference[:min_len]
        degraded = degraded[:min_len]
        
        return float(stoi(reference, degraded, sr, extended=False))
    
    @staticmethod
    def calculate_wer(
        reference: str,
        hypothesis: str
    ) -> float:
        return float(wer(reference, hypothesis))
    
    @staticmethod
    def calculate_cer(
        reference: str,
        hypothesis: str
    ) -> float:
        return float(cer(reference, hypothesis))
    
    @staticmethod
    def calculate_rtf(
        processing_time: float,
        audio_duration: float
    ) -> float:
        if audio_duration == 0:
            return float('inf')
        return float(processing_time / audio_duration)
    
    @staticmethod
    def export_metrics_json(
        metrics: Dict,
        output_path: Union[str, Path]
    ) -> None:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def export_metrics_csv(
        metrics: Dict,
        output_path: Union[str, Path]
    ) -> None:
        import csv
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            for key, value in metrics.items():
                writer.writerow([key, value])
    
    @staticmethod
    def create_metrics_report(
        noise_type: str,
        snr_original: float,
        snr_cleaned: float,
        pesq_score: Optional[float],
        stoi_score: Optional[float],
        wer_score: Optional[float],
        cer_score: Optional[float],
        rtf: float,
        transcription: str
    ) -> Dict:
        return {
            "noise_type": noise_type,
            "snr_original_db": round(snr_original, 2),
            "snr_cleaned_db": round(snr_cleaned, 2),
            "snr_improvement_db": round(snr_cleaned - snr_original, 2),
            "pesq": round(pesq_score, 3) if pesq_score else None,
            "stoi": round(stoi_score, 3) if stoi_score else None,
            "wer": round(wer_score, 4) if wer_score else None,
            "cer": round(cer_score, 4) if cer_score else None,
            "rtf": round(rtf, 3),
            "transcription": transcription
        }


if __name__ == "__main__":
    print("Testing AudioUtils...")
    
    sr = 16000
    duration = 3.0
    audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.5
    
    test_path = Path("test_audio.wav")
    AudioUtils.save_audio(test_path, audio, sr)
    loaded_audio, loaded_sr = AudioUtils.load_audio(test_path, sr=sr)
    assert loaded_sr == sr
    assert len(loaded_audio) == len(audio)
    print("✓ Load/save audio works")
    
    resampled = AudioUtils.resample_audio(audio, sr, 8000)
    assert len(resampled) == int(len(audio) * 8000 / sr)
    print("✓ Resample audio works")
    
    normalized = AudioUtils.normalize_audio(audio, target_level=0.95)
    assert np.max(np.abs(normalized)) <= 0.95
    print("✓ Normalize audio works")
    
    tensor = AudioUtils.numpy_to_torch(audio)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape[0] == 1
    back_to_numpy = AudioUtils.torch_to_numpy(tensor)
    assert isinstance(back_to_numpy, np.ndarray)
    print("✓ Numpy/Torch conversion works")
    
    rms = AudioUtils.calculate_rms(audio)
    assert rms > 0
    print("✓ RMS calculation works")
    
    signal = np.random.randn(sr) * 0.5
    noise = np.random.randn(sr) * 0.1
    snr = AudioUtils.calculate_snr(signal, noise)
    assert snr > 0
    print("✓ SNR calculation works")
    
    ref_audio = np.random.randn(sr * 2).astype(np.float32) * 0.3
    deg_audio = ref_audio + np.random.randn(sr * 2).astype(np.float32) * 0.05
    
    pesq_score = AudioUtils.calculate_pesq(ref_audio, deg_audio, sr)
    assert -0.5 <= pesq_score <= 4.5
    print("✓ PESQ calculation works")
    
    stoi_score = AudioUtils.calculate_stoi(ref_audio, deg_audio, sr)
    assert 0 <= stoi_score <= 1
    print("✓ STOI calculation works")
    
    ref_text = "hello world this is a test"
    hyp_text = "hello world this is test"
    wer_score = AudioUtils.calculate_wer(ref_text, hyp_text)
    assert wer_score > 0
    print("✓ WER calculation works")
    
    cer_score = AudioUtils.calculate_cer(ref_text, hyp_text)
    assert cer_score > 0
    print("✓ CER calculation works")
    
    rtf = AudioUtils.calculate_rtf(1.5, 3.0)
    assert rtf == 0.5
    print("✓ RTF calculation works")
    
    metrics = AudioUtils.create_metrics_report(
        noise_type="traffic",
        snr_original=8.5,
        snr_cleaned=23.2,
        pesq_score=3.1,
        stoi_score=0.92,
        wer_score=0.12,
        cer_score=0.05,
        rtf=0.45,
        transcription="test transcription"
    )
    assert metrics["snr_improvement_db"] == 14.7
    print("✓ Metrics report creation works")
    
    json_path = Path("test_metrics.json")
    AudioUtils.export_metrics_json(metrics, json_path)
    assert json_path.exists()
    print("✓ JSON export works")
    
    csv_path = Path("test_metrics.csv")
    AudioUtils.export_metrics_csv(metrics, csv_path)
    assert csv_path.exists()
    print("✓ CSV export works")
    
    test_path.unlink()
    json_path.unlink()
    csv_path.unlink()
    
    print("\n✓ AudioUtils module working correctly")