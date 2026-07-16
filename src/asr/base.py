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
        """Release resources when an adapter owns them."""
        return None

    def __enter__(self):
        """Return this adapter as a context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Unload the adapter when leaving the context."""
        self.unload()
        return False
