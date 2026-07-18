from abc import ABC, abstractmethod


class BaseVoiceDetection(ABC):
    @abstractmethod
    def detect(self, audio):
        """Required method for getting voice & removing noise"""
        raise NotImplementedError

    def unload(self):
        """Release resources when an adapter owns them."""
        return None
