from abc import ABC, abstractmethod
from pathlib import Path

from src.models.diarization import DiarizationResult


class BaseDiarizer(ABC):
    @abstractmethod
    def diarize(self, audio_path: Path) -> DiarizationResult:
        """Return anonymous speaker turns for one audio file."""
        raise NotImplementedError
