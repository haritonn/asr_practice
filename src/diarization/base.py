from abc import ABC, abstractmethod


class BaseDiarizer(ABC):
    @abstractmethod
    def diarize(self, audio_path):
        """Return anonymous speaker turns for one audio file."""
        raise NotImplementedError

    def unload(self):
        """Release model resources after diarization."""
        return None
