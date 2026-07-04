import gc
from pathlib import Path

from faster_whisper import WhisperModel

from ..models.asr import Segment, TranscribeResult, Word
from ..models.configs import WhisperConfig
from .base import BaseAsr


class WhisperAsr(BaseAsr):
    def __init__(self, config: WhisperConfig):
        self.config = config
        self._init_model()

    def _init_model(self):
        self._model = WhisperModel(**self.config.model_kwargs())

    def transcribe(self, audio: Path) -> TranscribeResult:
        if self._model is None:
            raise RuntimeError("Model has been unloaded before")
        whisper_segments, info = self._model.transcribe(
            audio, **self.config.transcribe_kwargs()
        )

        segments = [
            self._convert_segment(whisper_segment)
            for whisper_segment in whisper_segments
        ]

        text = " ".join(segment.text for segment in segments)
        return TranscribeResult(text=text, segments=segments, language=info.language)

    def _convert_segment(self, raw_segment):
        return Segment(
            text=raw_segment.text,
            start=raw_segment.start,
            end=raw_segment.end,
            words=[
                Word(text=word.word, start=word.start, end=word.end)
                for word in raw_segment.words
            ],
        )

    def unload(self) -> None:
        self._model = None
        gc.collect()
        # torch.cuda.empty_cache() idk about this one
