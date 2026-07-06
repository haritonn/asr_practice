from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from ..models.vad import SpeechSegment


class BaseVoiceDetection(ABC):
    @abstractmethod
    def detect(self, audio: Path) -> List[SpeechSegment]:
        """Required method for getting voice & removing noise"""
        raise NotImplementedError

    def unload(self) -> None:
        """Optional method for clearning space after model usage"""
        pass
