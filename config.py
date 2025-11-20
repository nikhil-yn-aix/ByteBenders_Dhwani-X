from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    yamnet_url: str = "https://tfhub.dev/google/yamnet/1"
    silero_vad_repo: str = "snakers4/silero-vad"
    demucs_model: str = "dns64"
    resemble_enhance_repo: str = "ResembleAI/resemble-enhance"
    indicwhisper_model: str = "ai4bharat/indicwhisper-kannada"
    
    
@dataclass
class AudioConfig:
    sample_rate: int = 16000
    high_quality_sr: int = 44100
    chunk_duration: float = 1.0
    context_duration: float = 3.0
    hop_duration: float = 0.5
    vad_threshold: float = 0.5
    vad_min_speech_duration: float = 0.25
    vad_min_silence_duration: float = 0.5
    
    @property
    def chunk_samples(self) -> int:
        return int(self.chunk_duration * self.sample_rate)
    
    @property
    def context_samples(self) -> int:
        return int(self.context_duration * self.sample_rate)
    
    @property
    def hop_samples(self) -> int:
        return int(self.hop_duration * self.sample_rate)


@dataclass
class SNRConfig:
    traffic_target: float = 10.0
    indoor_target: float = 14.0
    crowd_target: float = 12.0
    construction_target: float = 7.0
    min_improvement: float = 15.0


@dataclass
class PathConfig:
    esc50_dir: Path = Path("./ESC-50-master")
    output_dir: Path = Path("./output")
    models_cache_dir: Path = Path("./models_cache")
    
    def __post_init__(self):
        self.esc50_dir = Path(self.esc50_dir)
        self.output_dir = Path(self.output_dir)
        self.models_cache_dir = Path(self.models_cache_dir)
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.models_cache_dir.mkdir(exist_ok=True, parents=True)


@dataclass
class BufferConfig:
    queue_maxsize: int = 100
    processing_timeout: float = 0.5
    max_utterance_duration: float = 30.0
    noise_update_interval: float = 5.0


class Config:
    def __init__(
        self,
        models: Optional[ModelConfig] = None,
        audio: Optional[AudioConfig] = None,
        snr: Optional[SNRConfig] = None,
        paths: Optional[PathConfig] = None,
        buffer: Optional[BufferConfig] = None
    ):
        self.models = models or ModelConfig()
        self.audio = audio or AudioConfig()
        self.snr = snr or SNRConfig()
        self.paths = paths or PathConfig()
        self.buffer = buffer or BufferConfig()
    
    @classmethod
    def default(cls):
        return cls()


if __name__ == "__main__":
    print("Testing Config...")
    
    config = Config.default()
    
    assert config.models.yamnet_url == "https://tfhub.dev/google/yamnet/1"
    assert config.audio.sample_rate == 16000
    assert config.audio.chunk_samples == 16000
    assert config.audio.context_samples == 48000
    assert config.snr.traffic_target == 10.0
    assert config.paths.output_dir.exists()
    assert config.buffer.queue_maxsize == 100
    print("✓ All config tests passed")
    
    custom_config = Config(
        audio=AudioConfig(sample_rate=44100, chunk_duration=2.0),
        snr=SNRConfig(traffic_target=15.0)
    )
    
    assert custom_config.audio.sample_rate == 44100
    assert custom_config.audio.chunk_samples == 88200
    assert custom_config.snr.traffic_target == 15.0
    assert custom_config.models.yamnet_url == "https://tfhub.dev/google/yamnet/1"
    print("✓ Custom config tests passed")
    
    print("\n✓ Config module working correctly")