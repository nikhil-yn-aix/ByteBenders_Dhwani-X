import numpy as np
from pathlib import Path
from typing import Dict, List
import time
import json
import pandas as pd
from datetime import datetime
from config import Config
from audio_utils import AudioUtils
from transcriber import Transcriber
from pipeline_recorded import RecordedPipeline


class Evaluator:
    
    def __init__(self, config: Config = None):
        self.config = config or Config.default()
        self.pipeline = RecordedPipeline(config)
        self.transcriber = Transcriber(config, language="kn")
        
        self.noise_types = ['clean', 'traffic', 'indoor', 'crowd', 'construction']
        
    def evaluate_folder(
        self,
        speaker_folder: Path,
        output_dir: Path = None,
        speaker_id: str = None
    ) -> Dict:
        
        speaker_folder = Path(speaker_folder)
        if not speaker_folder.exists():
            raise ValueError(f"Folder not found: {speaker_folder}")
        
        if speaker_id is None:
            speaker_id = speaker_folder.name
        
        if output_dir is None:
            output_dir = Path("evaluate") / speaker_id
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print("\n" + "="*70)
        print(f"EVALUATION: {speaker_id}")
        print("="*70)
        print(f"Input folder: {speaker_folder}")
        print(f"Output folder: {output_dir}")
        print("="*70 + "\n")
        
        audio_files = self._find_audio_files(speaker_folder)
        
        if not audio_files:
            print("✗ No audio files found matching pattern")
            return {}
        
        print(f"Found {len(audio_files)} audio files:")
        for noise_type, path in audio_files.items():
            print(f"  ✓ {noise_type}: {path.name}")
        print()
        
        all_results = {}
        comparison_data = []
        
        for noise_type in self.noise_types:
            if noise_type not in audio_files:
                print(f"⚠ Skipping {noise_type} (file not found)")
                continue
            
            audio_path = audio_files[noise_type]
            
            print(f"\n{'='*70}")
            print(f"Processing: {noise_type.upper()}")
            print(f"{'='*70}")
            
            result_dir = output_dir / noise_type
            result_dir.mkdir(exist_ok=True, parents=True)
            
            print(f"\n[BASELINE] Transcription without preprocessing...")
            baseline_result = self._run_baseline(audio_path, result_dir)
            
            print(f"\n[PIPELINE] Full preprocessing + transcription...")
            pipeline_result = self._run_pipeline(audio_path, result_dir)
            
            comparison = self._compare_results(
                baseline_result,
                pipeline_result,
                noise_type
            )
            
            all_results[noise_type] = {
                'baseline': baseline_result,
                'pipeline': pipeline_result,
                'comparison': comparison
            }
            
            comparison_data.append(comparison)
            
            self._save_individual_report(result_dir, noise_type, all_results[noise_type])
        
        self._save_summary_report(output_dir, all_results, comparison_data)
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
        print(f"✓ Results saved to: {output_dir}/")
        print(f"  - summary_report.json")
        print(f"  - comparison_table.csv")
        print(f"  - Individual folders for each noise type")
        print("="*70 + "\n")
        
        return all_results
    
    def _find_audio_files(self, folder: Path) -> Dict[str, Path]:
        
        audio_files = {}
        
        for noise_type in self.noise_types:
            pattern = f"*{noise_type}.wav"
            matches = list(folder.glob(pattern))
            
            if not matches:
                pattern = f"*{noise_type}.mp3"
                matches = list(folder.glob(pattern))
            
            if matches:
                audio_files[noise_type] = matches[0]
        
        return audio_files
    
    def _run_baseline(self, audio_path: Path, output_dir: Path) -> Dict:
        
        baseline_dir = output_dir / "baseline"
        baseline_dir.mkdir(exist_ok=True, parents=True)
        
        start_time = time.time()
        
        audio, sr = AudioUtils.load_audio(audio_path, sr=16000)
        duration = len(audio) / sr
        
        AudioUtils.save_audio(baseline_dir / "input.wav", audio, sr)
        
        transcription = self.transcriber.transcribe(audio, sr)
        
        processing_time = time.time() - start_time
        rtf = AudioUtils.calculate_rtf(processing_time, duration)
        
        result = {
            'transcription': transcription['text'],
            'audio_duration_sec': round(duration, 2),
            'processing_time_sec': round(processing_time, 2),
            'rtf': round(rtf, 3),
            'wer': None,
            'cer': None
        }
        
        with open(baseline_dir / "baseline_result.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        with open(baseline_dir / "transcription.txt", 'w', encoding='utf-8') as f:
            f.write("BASELINE TRANSCRIPTION (No Preprocessing)\n")
            f.write("="*70 + "\n\n")
            f.write(transcription['text'])
            f.write("\n")
        
        print(f"✓ Baseline transcription: {len(transcription['text'])} characters")
        print(f"  Processing time: {processing_time:.2f}s (RTF: {rtf:.3f}x)")
        
        return result
    
    def _run_pipeline(self, audio_path: Path, output_dir: Path) -> Dict:
        
        pipeline_dir = output_dir / "pipeline"
        pipeline_dir.mkdir(exist_ok=True, parents=True)
        
        result = self.pipeline.process(
            audio_path=audio_path,
            output_dir=pipeline_dir,
            ground_truth=None,
            save_intermediate=True
        )
        
        return result
    
    def _compare_results(
        self,
        baseline: Dict,
        pipeline: Dict,
        noise_type: str
    ) -> Dict:
        
        baseline_wer = baseline.get('wer')
        pipeline_wer = pipeline['accuracy'].get('wer')
        
        baseline_cer = baseline.get('cer')
        pipeline_cer = pipeline['accuracy'].get('cer')
        
        wer_improvement = None
        if baseline_wer is not None and pipeline_wer is not None:
            wer_improvement = round((baseline_wer - pipeline_wer) * 100, 2)
        
        cer_improvement = None
        if baseline_cer is not None and pipeline_cer is not None:
            cer_improvement = round((baseline_cer - pipeline_cer) * 100, 2)
        
        comparison = {
            'noise_type': noise_type,
            'audio_duration_sec': baseline['audio_duration_sec'],
            
            'baseline': {
                'transcription': baseline['transcription'],
                'processing_time_sec': baseline['processing_time_sec'],
                'rtf': baseline['rtf'],
                'wer': baseline_wer,
                'cer': baseline_cer
            },
            
            'pipeline': {
                'transcription': pipeline['transcription']['text'],
                'noise_category': pipeline['noise_analysis']['category'],
                'noise_confidence': pipeline['noise_analysis']['confidence'],
                'snr_original_db': pipeline['audio_quality']['snr_original_db'],
                'snr_cleaned_db': pipeline['audio_quality']['snr_cleaned_db'],
                'snr_improvement_db': pipeline['audio_quality']['snr_improvement_db'],
                'processing_time_sec': pipeline['performance']['total_time_sec'],
                'rtf': pipeline['performance']['rtf'],
                'wer': pipeline_wer,
                'cer': pipeline_cer
            },
            
            'improvements': {
                'snr_improvement_db': pipeline['audio_quality']['snr_improvement_db'],
                'wer_improvement_percent': wer_improvement,
                'cer_improvement_percent': cer_improvement,
                'processing_overhead_sec': round(
                    pipeline['performance']['total_time_sec'] - baseline['processing_time_sec'],
                    2
                )
            }
        }
        
        return comparison
    
    def _save_individual_report(self, result_dir: Path, noise_type: str, results: Dict):
        
        with open(result_dir / "comparison.json", 'w', encoding='utf-8') as f:
            json.dump(results['comparison'], f, indent=2, ensure_ascii=False)
        
        with open(result_dir / "comparison.txt", 'w', encoding='utf-8') as f:
            comp = results['comparison']
            
            f.write(f"COMPARISON REPORT: {noise_type.upper()}\n")
            f.write("="*70 + "\n\n")
            
            f.write("BASELINE (No Preprocessing)\n")
            f.write("-"*70 + "\n")
            f.write(f"Transcription: {comp['baseline']['transcription']}\n")
            f.write(f"Processing Time: {comp['baseline']['processing_time_sec']:.2f}s\n")
            f.write(f"RTF: {comp['baseline']['rtf']:.3f}x\n")
            f.write(f"WER: {comp['baseline']['wer']}\n")
            f.write(f"CER: {comp['baseline']['cer']}\n\n")
            
            f.write("PIPELINE (With Preprocessing)\n")
            f.write("-"*70 + "\n")
            f.write(f"Detected Noise: {comp['pipeline']['noise_category']} "
                   f"({comp['pipeline']['noise_confidence']:.3f})\n")
            f.write(f"SNR Original: {comp['pipeline']['snr_original_db']} dB\n")
            f.write(f"SNR Cleaned: {comp['pipeline']['snr_cleaned_db']} dB\n")
            f.write(f"SNR Improvement: +{comp['pipeline']['snr_improvement_db']} dB\n\n")
            f.write(f"Transcription: {comp['pipeline']['transcription']}\n")
            f.write(f"Processing Time: {comp['pipeline']['processing_time_sec']:.2f}s\n")
            f.write(f"RTF: {comp['pipeline']['rtf']:.3f}x\n")
            f.write(f"WER: {comp['pipeline']['wer']}\n")
            f.write(f"CER: {comp['pipeline']['cer']}\n\n")
            
            f.write("IMPROVEMENTS\n")
            f.write("-"*70 + "\n")
            f.write(f"SNR Improvement: +{comp['improvements']['snr_improvement_db']} dB\n")
            f.write(f"WER Improvement: {comp['improvements']['wer_improvement_percent']}%\n")
            f.write(f"CER Improvement: {comp['improvements']['cer_improvement_percent']}%\n")
            f.write(f"Processing Overhead: +{comp['improvements']['processing_overhead_sec']:.2f}s\n")
    
    def _save_summary_report(self, output_dir: Path, all_results: Dict, comparison_data: List[Dict]):
        
        summary = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'total_files_evaluated': len(all_results),
            'noise_types': list(all_results.keys()),
            'results': all_results
        }
        
        with open(output_dir / "summary_report.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        table_data = []
        for comp in comparison_data:
            row = {
                'Noise Type': comp['noise_type'],
                'Duration (s)': comp['audio_duration_sec'],
                
                'Baseline WER': comp['baseline']['wer'],
                'Pipeline WER': comp['pipeline']['wer'],
                'WER Improvement (%)': comp['improvements']['wer_improvement_percent'],
                
                'Baseline CER': comp['baseline']['cer'],
                'Pipeline CER': comp['pipeline']['cer'],
                'CER Improvement (%)': comp['improvements']['cer_improvement_percent'],
                
                'SNR Original (dB)': comp['pipeline']['snr_original_db'],
                'SNR Cleaned (dB)': comp['pipeline']['snr_cleaned_db'],
                'SNR Improvement (dB)': comp['improvements']['snr_improvement_db'],
                
                'Baseline Time (s)': comp['baseline']['processing_time_sec'],
                'Pipeline Time (s)': comp['pipeline']['processing_time_sec'],
                'Overhead (s)': comp['improvements']['processing_overhead_sec'],
                
                'Baseline RTF': comp['baseline']['rtf'],
                'Pipeline RTF': comp['pipeline']['rtf']
            }
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        df.to_csv(output_dir / "comparison_table.csv", index=False)
        
        with open(output_dir / "summary_report.txt", 'w', encoding='utf-8') as f:
            f.write("EVALUATION SUMMARY REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Total Files Evaluated: {len(all_results)}\n")
            f.write(f"Noise Types: {', '.join(all_results.keys())}\n\n")
            
            f.write("="*70 + "\n")
            f.write("RESULTS BY NOISE TYPE\n")
            f.write("="*70 + "\n\n")
            
            for noise_type, results in all_results.items():
                comp = results['comparison']
                f.write(f"{noise_type.upper()}\n")
                f.write("-"*70 + "\n")
                f.write(f"  SNR Improvement: +{comp['improvements']['snr_improvement_db']} dB\n")
                f.write(f"  WER Improvement: {comp['improvements']['wer_improvement_percent']}%\n")
                f.write(f"  CER Improvement: {comp['improvements']['cer_improvement_percent']}%\n")
                f.write(f"  Processing Overhead: +{comp['improvements']['processing_overhead_sec']:.2f}s\n")
                f.write("\n")
        
        print(f"\n✓ Summary CSV saved: comparison_table.csv")


if __name__ == "__main__":
    print("KANNADA SPEECH DENOISING EVALUATION")
    print("="*70)
    
    speaker_folder = Path("speaker_audios/speaker_001")
    
    if not speaker_folder.exists():
        print(f"✗ Folder not found: {speaker_folder}")
        print("  Please ensure you have run audio_maker.py first")
        exit(1)
    
    evaluator = Evaluator()
    
    results = evaluator.evaluate_folder(
        speaker_folder=speaker_folder,
        output_dir=Path("evaluate/speaker_001"),
        speaker_id="speaker_001"
    )
    
    print("\n✓ Evaluation complete!")
    print("✓ Check evaluate/speaker_001/ for detailed results")