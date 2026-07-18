from abc import ABC, abstractmethod


class BaseAsr(ABC):
    @abstractmethod
    def transcribe(self, audio, speech_segments):
        """Required method for getting (raw) text from audio"""
        raise NotImplementedError

    def unload(self):
        """Release resources when an adapter owns them."""
        return None

    def __enter__(self):
        """Return this adapter as a context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Unload the adapter when leaving the context."""
        self.unload()
        return False
