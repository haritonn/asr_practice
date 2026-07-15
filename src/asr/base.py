from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from ..models.asr import TranscribeResult
from ..models.vad import SpeechSegment


class BaseAsr(ABC):
    @abstractmethod
    def transcribe(
        self, audio: Path, speech_segments: Sequence[SpeechSegment]
    ) -> TranscribeResult:
        """Required method for getting (raw) text from audio"""
        raise NotImplementedError

    def unload(self) -> None:
        """Optional method for clearning space after model usage"""
        pass

    def __enter__(self):
        """Optional method for 'with' context"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Unloading on exiting 'with' context"""
        self.unload()
        return False
