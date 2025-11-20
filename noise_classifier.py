import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from pathlib import Path
from typing import List, Dict
from config import Config
from audio_utils import AudioUtils


class NoiseClassifier:
    
    def __init__(self, config: Config = None):
        self.config = config or Config.default()
        self.model = None
        self.class_names = None
        self._load_model()
    
    def _load_model(self):
        self.model = hub.load(self.config.models.yamnet_url)
        
        class_map_path = self.model.class_map_path().numpy().decode('utf-8')
        
        with open(class_map_path, 'r') as f:
            lines = f.readlines()
        
        self.class_names = [line.strip().split(',')[2].strip('"') for line in lines[1:]]
    
    def classify(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        top_k: int = 10
    ) -> List[Dict[str, float]]:
        if sr != 16000:
            audio = AudioUtils.resample_audio(audio, sr, 16000)
        
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
        
        scores, embeddings, spectrogram = self.model(audio)
        
        mean_scores = np.mean(scores.numpy(), axis=0)
        
        top_indices = np.argsort(mean_scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'class': self.class_names[idx],
                'confidence': float(mean_scores[idx])
            })
        
        return results
    
    def filter_non_speech(
        self,
        predictions: List[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        speech_keywords = [
            'speech', 'narration', 'conversation', 'voice', 'talk',
            'silence', 'quiet', 'music', 'singing'
        ]
        
        filtered = []
        for pred in predictions:
            class_lower = pred['class'].lower()
            if not any(keyword in class_lower for keyword in speech_keywords):
                filtered.append(pred)
        
        return filtered
    
    def get_noise_type_mapping(self) -> Dict[str, List[str]]:
        return {
            'traffic': ['traffic', 'motor vehicle', 'car', 'vehicle', 'engine', 
                       'car horn', 'truck', 'bus', 'emergency vehicle'],
            'construction': ['jackhammer', 'drill', 'chainsaw', 'power tool', 'sawing',
                           'hammer', 'construction', 'drilling'],
            'crowd': ['crowd', 'hubbub', 'chatter', 'babble', 'restaurant',
                     'cafeteria', 'party', 'people', 'laughter'],
            'indoor': ['washing machine', 'vacuum cleaner', 'fan', 'air conditioning', 
                      'appliance', 'clock', 'refrigerator', 'dishwasher', 'microwave',
                      'blender', 'hair dryer', 'inside']
        }
    
    def map_to_noise_category(
        self,
        predictions: List[Dict[str, float]]
    ) -> str:
        mapping = self.get_noise_type_mapping()
        
        for pred in predictions:
            class_name = pred['class'].lower()
            for category, keywords in mapping.items():
                if any(keyword in class_name for keyword in keywords):
                    return category
        
        return "unknown"
    
    def analyze_background_noise(
        self,
        audio: np.ndarray,
        sr: int = 16000
    ) -> Dict:
        all_predictions = self.classify(audio, sr, top_k=20)
        
        non_speech = self.filter_non_speech(all_predictions)
        
        category = self.map_to_noise_category(non_speech) if non_speech else "unknown"
        
        return {
            'category': category,
            'top_prediction': non_speech[0] if non_speech else all_predictions[0],
            'all_non_speech': non_speech[:5]
        }


if __name__ == "__main__":
    print("Testing NoiseClassifier...")
    print("=" * 60)
    
    test_audio_path = Path("Nikhil_Indoor.mp3")
    if not test_audio_path.exists():
        print("✗ Test audio file not found: Nikhil_Indoor.mp3")
        exit(1)
    
    audio, sr = AudioUtils.load_audio(test_audio_path, sr=16000)
    print(f"Loaded audio: {len(audio)/sr:.1f}s @ {sr}Hz")
    
    classifier = NoiseClassifier()
    print("YAMNet model loaded")
    print("=" * 60)
    
    result = classifier.analyze_background_noise(audio, sr)
    
    print("\nBackground Noise Analysis:")
    print("-" * 60)
    print(f"Detected Category: {result['category'].upper()}")
    print(f"\nTop Background Noise:")
    print(f"  {result['top_prediction']['class']}: {result['top_prediction']['confidence']:.3f}")
    
    print(f"\nTop 5 Non-Speech Sounds:")
    for i, pred in enumerate(result['all_non_speech'], 1):
        print(f"  {i}. {pred['class']}: {pred['confidence']:.3f}")
    
    print("=" * 60)
    print("✓ NoiseClassifier working correctly")