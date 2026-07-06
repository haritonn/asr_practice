import gc
from pathlib import Path
from typing import List

import numpy as np
from faster_whisper import WhisperModel
from pydub import AudioSegment

from ..models.asr import Segment, TranscribeResult, Word
from ..models.configs import WhisperConfig
from ..models.vad import SpeechSegment
from .base import BaseAsr


class WhisperAsr(BaseAsr):
    def __init__(self, config: WhisperConfig):
        self.config = config
        self._init_model()

    def _init_model(self):
        self._model = WhisperModel(**self.config.model_kwargs())

    def transcribe(
        self, audio: Path, speech_segments: List[SpeechSegment]
    ) -> TranscribeResult:
        if self._model is None:
            raise RuntimeError("Model has been unloaded before")

        if not speech_segments:
            return TranscribeResult(
                text="", segments=[], language="ru"
            )  # language is bad field in this case...

        audio_segment = AudioSegment.from_file(audio)
        all_segments: list[Segment] = []
        all_words: list[str] = []
        detected_language = None

        for speech_segment in speech_segments:
            chunk_audio = self._slice_audio(audio_segment, speech_segment)
            whisper_segments, info = self._model.transcribe(
                chunk_audio, **self.config.transcribe_kwargs()
            )

            if detected_language is None:
                detected_language = info.language

            converted = [
                self._convert_segment(ws, offset=speech_segment.start)
                for ws in whisper_segments
            ]
            all_segments.extend(converted)
            all_words.extend(seg.text for seg in converted)

        text = " ".join(all_words).strip()
        return TranscribeResult(
            text=text,
            segments=all_segments,
            language=detected_language,
        )

    def _slice_audio(
        self, audio: AudioSegment, speech_segment: SpeechSegment
    ) -> np.ndarray:
        start_ms = max(0, int(speech_segment.start * 1000))
        end_ms = max(start_ms, int(speech_segment.end * 1000))
        chunk = audio[start_ms:end_ms].set_channels(1).set_frame_rate(16000)

        samples = np.array(chunk.get_array_of_samples(), dtype=np.float32)
        max_val = float(1 << (8 * chunk.sample_width - 1))
        samples = samples / max_val
        return samples

    def _convert_segment(self, raw_segment, offset: float = 0.0) -> Segment:
        words = []
        if raw_segment.words:
            words = [
                Word(
                    text=word.word,
                    start=word.start + offset,
                    end=word.end + offset,
                )
                for word in raw_segment.words
            ]

        return Segment(
            text=raw_segment.text,
            start=raw_segment.start + offset,
            end=raw_segment.end + offset,
            words=words,
        )

    def unload(self) -> None:
        self._model = None
        gc.collect()
