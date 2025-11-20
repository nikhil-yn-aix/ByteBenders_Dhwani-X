import numpy as np
from pathlib import Path
from typing import Dict, Optional
import time
import json
from datetime import datetime
from config import Config
from audio_utils import AudioUtils
from vad_processor import VADProcessor
from noise_classifier import NoiseClassifier
from denoiser_preprocessor import DenoiserProcessor
from transcriber import Transcriber


class RecordedPipeline:
    
    def __init__(self, config: Config = None):
        self.config = config or Config.default()
        
        print("Initializing pipeline components...")
        self.vad = VADProcessor(config)
        self.classifier = NoiseClassifier(config)
        self.denoiser = DenoiserProcessor(config)
        self.transcriber = Transcriber(config, language="kn")
        print("✓ All components loaded\n")
    
    def process(
        self,
        audio_path: Path,
        output_dir: Optional[Path] = None,
        ground_truth: Optional[str] = None,
        save_intermediate: bool = False,
        yield_progress: bool = False
    ):
        
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.config.paths.output_dir / f"pipeline_{timestamp}"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print("\n" + "="*70)
        print("KANNADA SPEECH DENOISING & TRANSCRIPTION PIPELINE")
        print("="*70)
        print(f"Input: {audio_path.name}")
        print(f"Output: {output_dir}")
        print("="*70 + "\n")
        
        pipeline_start = time.time()
        
        print("[1/6] Loading audio...")
        audio, sr = AudioUtils.load_audio(audio_path, sr=16000)
        duration = len(audio) / sr
        print(f"✓ Loaded: {duration:.1f}s @ {sr}Hz")
        if yield_progress:
            yield {"step": 0, "status": "LOAD", "elapsed": time.time() - pipeline_start}
        
        if save_intermediate:
            AudioUtils.save_audio(output_dir / "01_original.wav", audio, sr)
        
        print("\n[2/6] Classifying background noise...")
        noise_result = self.classifier.analyze_background_noise(audio, sr)
        print(f"✓ Detected noise type: {noise_result['category'].upper()}")
        print(f"  Confidence: {noise_result['top_prediction']['confidence']:.3f}")
        print(f"  Top prediction: {noise_result['top_prediction']['class']}")
        if yield_progress:
            yield {"step": 1, "status": "NOISE", "elapsed": time.time() - pipeline_start}
        
        print("\n[3/6] Running Voice Activity Detection...")
        timestamps = self.vad.process_audio(audio, sr)
        print(f"✓ Found {len(timestamps)} speech segments")
        
        speech_segments = self.vad.extract_speech_segments(audio, timestamps)
        silence_segments = self.vad.extract_silence_segments(audio, timestamps, sr)
        
        speech_ratio = self.vad.get_speech_ratio(timestamps, duration)
        print(f"  Speech ratio: {speech_ratio:.1%}")
        
        if speech_segments:
            speech_only = np.concatenate(speech_segments)
            if save_intermediate:
                AudioUtils.save_audio(output_dir / "02_speech_only.wav", speech_only, sr)
        else:
            speech_only = np.array([])
        
        if silence_segments:
            noise_only = np.concatenate(silence_segments)
            if save_intermediate:
                AudioUtils.save_audio(output_dir / "03_noise_only.wav", noise_only, sr)
            snr_original = AudioUtils.calculate_snr(speech_only, noise_only)
        else:
            noise_only = np.array([])
            snr_original = float('inf')
        
        print(f"  Original SNR: {snr_original:.2f} dB")
        if yield_progress:
            yield {"step": 2, "status": "VAD", "elapsed": time.time() - pipeline_start}
        
        print("\n[4/6] Denoising audio...")
        denoise_start = time.time()
        denoised = self.denoiser.denoise(audio, sr)
        denoise_time = time.time() - denoise_start
        print(f"✓ Denoising complete ({denoise_time:.2f}s)")
        
        if save_intermediate:
            AudioUtils.save_audio(output_dir / "04_denoised.wav", denoised, sr)
        
        denoised_speech = self.vad.extract_speech_segments(denoised, timestamps)
        if denoised_speech:
            denoised_speech_only = np.concatenate(denoised_speech)
        else:
            denoised_speech_only = np.array([])
        
        if len(noise_only) > 0 and len(denoised_speech_only) > 0:
            snr_cleaned = AudioUtils.calculate_snr(denoised_speech_only, noise_only)
            snr_improvement = snr_cleaned - snr_original
        else:
            snr_cleaned = float('inf')
            snr_improvement = 0.0
        
        print(f"  Cleaned SNR: {snr_cleaned:.2f} dB")
        print(f"  Improvement: +{snr_improvement:.2f} dB")
        if yield_progress:
            yield {"step": 3, "status": "DENOISE", "elapsed": time.time() - pipeline_start}
        
        AudioUtils.save_audio(output_dir / "final_denoised.wav", denoised, sr)
        
        print("\n[5/6] Transcribing speech...")
        transcribe_start = time.time()
        transcription_result = self.transcriber.transcribe(denoised, sr)
        transcribe_time = time.time() - transcribe_start
        print(f"✓ Transcription complete ({transcribe_time:.2f}s)")
        print(f"  Language: {transcription_result['language']}")
        print(f"  Text preview: {transcription_result['text'][:100]}...")
        if yield_progress:
            yield {"step": 4, "status": "ASR", "elapsed": time.time() - pipeline_start}
        
        print("\n[6/6] Computing metrics...")
        wer_score = None
        cer_score = None
        if ground_truth:
            wer_score = AudioUtils.calculate_wer(ground_truth, transcription_result['text'])
            cer_score = AudioUtils.calculate_cer(ground_truth, transcription_result['text'])
            print(f"✓ WER: {wer_score:.4f} ({wer_score*100:.2f}%)")
            print(f"  CER: {cer_score:.4f} ({cer_score*100:.2f}%)")
        
        total_time = time.time() - pipeline_start
        rtf = AudioUtils.calculate_rtf(total_time, duration)
        
        print(f"\n✓ Total processing time: {total_time:.2f}s")
        print(f"  Real-time factor: {rtf:.3f}x")
        
        results = {
            "metadata": {
                "input_file": str(audio_path),
                "output_dir": str(output_dir),
                "timestamp": datetime.now().isoformat(),
                "audio_duration_sec": round(duration, 2),
                "sample_rate": sr
            },
            "noise_analysis": {
                "category": noise_result['category'],
                "top_prediction": noise_result['top_prediction']['class'],
                "confidence": round(noise_result['top_prediction']['confidence'], 3),
                "all_predictions": [
                    {"class": p['class'], "confidence": round(p['confidence'], 3)}
                    for p in noise_result['all_non_speech'][:3]
                ]
            },
            "vad_analysis": {
                "speech_segments": len(timestamps),
                "speech_ratio": round(speech_ratio, 3),
                "timestamps": timestamps
            },
            "audio_quality": {
                "snr_original_db": round(snr_original, 2) if snr_original != float('inf') else None,
                "snr_cleaned_db": round(snr_cleaned, 2) if snr_cleaned != float('inf') else None,
                "snr_improvement_db": round(snr_improvement, 2)
            },
            "transcription": {
                "text": transcription_result['text'],
                "language": transcription_result['language'],
                "model": transcription_result['model']
            },
            "accuracy": {
                "wer": round(wer_score, 4) if wer_score else None,
                "cer": round(cer_score, 4) if cer_score else None,
                "ground_truth": ground_truth
            },
            "performance": {
                "total_time_sec": round(total_time, 2),
                "denoise_time_sec": round(denoise_time, 2),
                "transcribe_time_sec": round(transcribe_time, 2),
                "rtf": round(rtf, 3)
            }
        }
        
        with open(output_dir / "results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        with open(output_dir / "transcription.txt", 'w', encoding='utf-8') as f:
            f.write(f"KANNADA SPEECH TRANSCRIPTION\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"File: {audio_path.name}\n")
            f.write(f"Duration: {duration:.1f}s\n")
            f.write(f"Language: {transcription_result['language']}\n")
            f.write(f"Model: {transcription_result['model']}\n\n")
            f.write("=" * 70 + "\n")
            f.write("TRANSCRIPTION\n")
            f.write("=" * 70 + "\n\n")
            f.write(transcription_result['text'])
            f.write("\n\n")
            if ground_truth:
                f.write("=" * 70 + "\n")
                f.write("GROUND TRUTH\n")
                f.write("=" * 70 + "\n\n")
                f.write(ground_truth)
                f.write("\n\n")
                f.write(f"WER: {wer_score:.4f} ({wer_score*100:.2f}%)\n")
                f.write(f"CER: {cer_score:.4f} ({cer_score*100:.2f}%)\n")
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        print(f"✓ Results saved to: {output_dir}/")
        print(f"  - final_denoised.wav")
        print(f"  - transcription.txt")
        print(f"  - results.json")
        if save_intermediate:
            print(f"  - intermediate files (01-04)")
        print("="*70 + "\n")
        
        if yield_progress:
            yield {"step": 5, "status": "COMPLETE", "results": results, "elapsed": time.time() - pipeline_start}
        else:
            return results


if __name__ == "__main__":
    print("Testing RecordedPipeline...")
    print("=" * 70)
    
    test_audio_path = Path("data/Nikhil_Indoor.mp3")
    if not test_audio_path.exists():
        print("✗ Test audio file not found: Nikhil_Indoor.mp3")
        exit(1)
    
    pipeline = RecordedPipeline()
    
    results = pipeline.process(
        audio_path=test_audio_path,
        output_dir=Path("output/pipeline_test"),
        ground_truth=None,
        save_intermediate=True
    )
    
    print("\n✓ Pipeline test complete!")
    print(f"✓ Check output/pipeline_test/ for results")