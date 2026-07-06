from pathlib import Path
from typing import List

import librosa
import numpy as np
from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

from ..models.configs import SileroConfig
from ..models.vad import SpeechSegment
from .base import BaseVoiceDetection


class SileroVoiceDetection(BaseVoiceDetection):
    def __init__(self, conf: SileroConfig):
        self.config = conf
        self._model = load_silero_vad()

    def detect(self, audio: Path) -> List[SpeechSegment]:
        wav = self._load_audio(audio)

        timestamps = get_speech_timestamps(
            wav,
            self._model,
            sampling_rate=self.config.sample_rate,
            threshold=self.config.speech_threshold,
            min_speech_duration_ms=self.config.min_speech_duration_ms,
            min_silence_duration_ms=self.config.min_silence_duration_ms,
            return_seconds=True,
        )

        return [
            SpeechSegment(start=item["start"], end=item["end"]) for item in timestamps
        ]

    def _load_audio(self, audio: Path) -> np.ndarray:
        audio, _ = librosa.load(audio, sr=self.config.sample_rate, mono=True)
        return audio.astype(np.float32)

    def unload(self):
        self._model = None
