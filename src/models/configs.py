from dataclasses import dataclass


@dataclass
class WhisperConfig:
    model_size_or_path: str
    device: str
    beam_size: int
    language: str
    word_timestamps: bool

    def model_kwargs(self) -> dict:
        """Returning some arguments in **kwargs format for model"""
        return {
            "model_size_or_path": self.model_size_or_path,
            "device": self.device,
        }

    def transcribe_kwargs(self) -> dict:
        """Returning some arguments in **kwargs format for eval"""
        return {
            "beam_size": self.beam_size,
            "language": self.language,
            "word_timestamps": self.word_timestamps,
        }
