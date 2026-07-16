"""Offline adapter for pyannote Community-1 speaker diarization."""

from __future__ import annotations

import os
from pathlib import Path

import soundfile as sf
import torch
from pyannote.audio import Pipeline

from src.diarization.base import BaseDiarizer
from src.models.configs import PyannoteDiarizationConfig
from src.models.diarization import DiarizationResult, SpeakerTurn
from src.runtime.resources import release_accelerator_memory


class PyannoteCommunityDiarizer(BaseDiarizer):
    """Run the gated Community-1 model in its dedicated pyannote runtime."""

    def __init__(
        self,
        config: PyannoteDiarizationConfig | None = None,
        token: str | None = None,
    ):
        self.config = config or PyannoteDiarizationConfig()
        if self.config.num_speakers is not None and (
            self.config.min_speakers is not None or self.config.max_speakers is not None
        ):
            raise ValueError("num_speakers cannot be combined with min/max_speakers")
        if (
            self.config.min_speakers is not None
            and self.config.max_speakers is not None
            and self.config.min_speakers > self.config.max_speakers
        ):
            raise ValueError("min_speakers must not exceed max_speakers")

        access_token = token or os.environ.get("HF_TOKEN")
        if not access_token:
            raise RuntimeError(
                "Set HF_TOKEN or pass --hf-token after accepting model terms."
            )
        self._token = access_token
        self._pipeline = None

    def _ensure_loaded(self) -> None:
        if self._pipeline is None:
            self._pipeline = Pipeline.from_pretrained(
                self.config.model_id, token=self._token
            )
            self._pipeline.to(torch.device(self.config.device))

    def diarize(self, audio_path: Path) -> DiarizationResult:
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        self._ensure_loaded()
        # Supplying a waveform avoids a runtime dependency on torchcodec/FFmpeg and
        # keeps CPU and CUDA inference paths identical.
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio.T.copy())
        output = self._pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            **self.config.inference_kwargs(),
        )
        annotation = (
            output.exclusive_speaker_diarization
            if self.config.output == "exclusive"
            else output.speaker_diarization
        )
        raw_turns = list(annotation.itertracks(yield_label=True))
        speaker_map = {
            label: f"SPEAKER_{index:02d}"
            for index, label in enumerate(
                sorted({label for _, _, label in raw_turns}, key=str)
            )
        }
        turns = sorted(
            (
                SpeakerTurn(
                    start=float(segment.start),
                    end=float(segment.end),
                    speaker_id=speaker_map[label],
                )
                for segment, _, label in raw_turns
                if segment.end > segment.start
            ),
            key=lambda turn: (turn.start, turn.end, turn.speaker_id),
        )
        return DiarizationResult(turns=turns, num_speakers=len(speaker_map))

    def unload(self) -> None:
        if self._pipeline is None:
            return
        self._pipeline = None
        release_accelerator_memory()
