"""GigaAM CTC adapter for the common ASR pipeline.

GigaAM's public inference wrapper does not expose word timestamps.  The
adapter therefore returns one timestamped segment per VAD chunk.  This keeps
speaker attribution valid while making the timestamp granularity explicit.
"""

from __future__ import annotations

import gc
import importlib
from pathlib import Path
from typing import Sequence

import soundfile as sf

from ..models.asr import Segment, TranscribeResult
from ..models.configs import GigaAmConfig
from ..models.vad import SpeechSegment
from .base import BaseAsr


class GigaAmAsr(BaseAsr):
    def __init__(self, config: GigaAmConfig):
        self.config = config
        self._init_model()

    def _init_model(self) -> None:
        try:
            from transformers import AutoModel

            self._model = AutoModel.from_pretrained(
                self.config.model_id,
                revision=self.config.revision,
                trust_remote_code=True,
                local_files_only=self.config.local_files_only,
            )
        except OSError as error:
            raise RuntimeError(
                "GigaAM checkpoint is not available locally. Download it first or "
                "run with a network connection and local_files_only=False."
            ) from error

    def transcribe(
        self, audio: Path, speech_segments: Sequence[SpeechSegment]
    ) -> TranscribeResult:
        if self._model is None:
            raise RuntimeError("Model has been unloaded")
        if not speech_segments:
            return TranscribeResult(text="", segments=[], language="ru")

        import torch
        from torch.utils.data import DataLoader

        waveform, sample_rate = sf.read(audio, dtype="float32")
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        chunks = []
        kept_segments = []
        for segment in speech_segments:
            start = max(0, round(segment.start * sample_rate))
            end = min(len(waveform), round(segment.end * sample_rate))
            if end > start:
                chunks.append(torch.from_numpy(waveform[start:end].copy()))
                kept_segments.append(segment)
        if not chunks:
            return TranscribeResult(text="", segments=[], language="ru")

        module = importlib.import_module(self._model.model.__class__.__module__)
        dataset = module.AudioDataset(chunks, tokenizer=None)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=module.AudioDataset.collate,
        )
        device = torch.device(self.config.device)
        self._model.model.to(device).eval()
        texts: list[str] = []
        with torch.inference_mode():
            for wav_pad, wav_lens in dataloader:
                wav_pad = wav_pad.to(device).to(self._model.model._dtype)
                wav_lens = wav_lens.to(device)
                encoded, encoded_len = self._model.model.forward(wav_pad, wav_lens)
                texts.extend(
                    text for text, _ in self._model.model._decode(encoded, encoded_len, wav_lens, False)
                )

        segments = [
            Segment(text=text, start=chunk.start, end=chunk.end, words=[])
            for chunk, text in zip(kept_segments, texts)
        ]
        return TranscribeResult(
            text=" ".join(text.strip() for text in texts if text.strip()),
            segments=segments,
            language="ru",
        )

    def unload(self) -> None:
        self._model = None
        gc.collect()
