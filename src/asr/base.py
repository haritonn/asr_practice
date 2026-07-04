from abc import ABC, abstractmethod
from pathlib import Path

from ..models.asr import TranscribeResult


class BaseAsr(ABC):
    @abstractmethod
    def transcribe(self, audio: Path) -> TranscribeResult:
        """Required method for getting (raw) text from audio"""
        raise NotImplementedError

    def unload(self) -> None:
        """Optional method for clearning space after model usage"""
        raise NotImplementedError

    def __enter__(self):
        """Optional method for 'with' context"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Unloading on exiting 'with' context"""
        self.unload()
        return False
