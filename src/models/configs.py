from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class WhisperConfig:
    model_size_or_path: str | Path
    device: str
    beam_size: int
    language: str
    word_timestamps: bool

    def model_kwargs(self) -> Dict:
        """Returning some arguments in **kwargs format for model"""
        return {
            "model_size_or_path": self.model_size_or_path,
            "device": self.device,
        }

    def transcribe_kwargs(self) -> Dict:
        """Returning some arguments in **kwargs format for eval"""
        return {
            "beam_size": self.beam_size,
            "language": self.language,
            "word_timestamps": self.word_timestamps,
        }


@dataclass
class SileroConfig:
    model_path: Path
    device: str
    sample_rate: int
    window_size: int
    speech_threshold: float
    silence_threshold: float
    min_speech_duration_ms: int
    min_silence_duration_ms: int
