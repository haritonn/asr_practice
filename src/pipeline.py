from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from src.asr.base import BaseAsr
from src.asr.terminology import CtcTerminologyRecognizer
from src.audio.normalization import CanonicalAudio, normalize_audio
from src.diarization.base import BaseDiarizer
from src.diarization.transcript import assign_speakers, speaker_for_interval
from src.models.diarization import (
    DiarizedTranscript,
    SpeakerAttributedProductMention,
    TimedTranscriptSegment,
)
from src.vad.base import BaseVoiceDetection


@dataclass(frozen=True, slots=True)
class PipelineRuntime:
    """Measured duration of one sequential pipeline execution."""

    audio_seconds: float
    total_seconds: float
    stages: dict[str, float]
    canonical_audio: CanonicalAudio

    def to_dict(self) -> dict:
        return {
            "audio_seconds": round(self.audio_seconds, 3),
            "total_seconds": round(self.total_seconds, 3),
            "realtime_factor": round(
                self.total_seconds / max(self.audio_seconds, 1e-9), 3
            ),
            "stages": {name: round(seconds, 3) for name, seconds in self.stages.items()},
            "canonical_audio": {
                "duration_seconds": round(self.canonical_audio.duration_seconds, 3),
                "sample_rate": self.canonical_audio.sample_rate,
                "channels": self.canonical_audio.channels,
                "subtype": self.canonical_audio.subtype,
            },
        }


class DiarizedSpeechPipeline:
    """Run VAD, ASR and diarization and produce one speaker-attributed transcript."""

    def __init__(
        self,
        asr: BaseAsr,
        vad: BaseVoiceDetection,
        diarizer: BaseDiarizer,
        terminology: CtcTerminologyRecognizer | None = None,
    ):
        self.asr = asr
        self.vad = vad
        self.diarizer = diarizer
        self.terminology = terminology
        self.last_runtime: PipelineRuntime | None = None

    def transcribe(self, audio_path: Path) -> DiarizedTranscript:
        started_at = time.perf_counter()
        stage_seconds: dict[str, float] = {}
        stage_started_at = time.perf_counter()
        with TemporaryDirectory(prefix="asr_canonical_") as temporary_directory:
            canonical_audio = normalize_audio(
                audio_path, Path(temporary_directory) / "audio_16khz_mono.wav"
            )
            stage_seconds["normalization"] = time.perf_counter() - stage_started_at

            stage_started_at = time.perf_counter()
            try:
                speech_segments = self.vad.detect(canonical_audio.path)
            finally:
                self.vad.unload()
                stage_seconds["vad"] = time.perf_counter() - stage_started_at

            stage_started_at = time.perf_counter()
            try:
                transcription = self.asr.transcribe(canonical_audio.path, speech_segments)
            finally:
                self.asr.unload()
                stage_seconds["asr"] = time.perf_counter() - stage_started_at

            stage_started_at = time.perf_counter()
            try:
                diarization = self.diarizer.diarize(canonical_audio.path)
            finally:
                self.diarizer.unload()
                stage_seconds["diarization"] = time.perf_counter() - stage_started_at

            product_mentions = []
            if self.terminology is not None:
                stage_started_at = time.perf_counter()
                try:
                    product_mentions = [
                        SpeakerAttributedProductMention(
                            product_id=mention.product_id,
                            canonical_name=mention.canonical_name,
                            score=mention.score,
                            start=mention.start,
                            end=mention.end,
                            speaker_id=speaker_for_interval(
                                mention.start, mention.end, diarization.turns
                            ),
                        )
                        for mention in self.terminology.recognize(
                            canonical_audio.path, speech_segments
                        )
                    ]
                finally:
                    self.terminology.unload()
                    stage_seconds["terminology"] = time.perf_counter() - stage_started_at
            else:
                stage_seconds["terminology"] = 0.0

            stage_started_at = time.perf_counter()
            segments = [
                TimedTranscriptSegment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text.strip(),
                )
                for segment in transcription.segments
                if segment.text.strip()
            ]
            result = DiarizedTranscript(
                audio_path=audio_path,
                text=transcription.text,
                language=transcription.language,
                num_speakers=diarization.num_speakers,
                speaker_turns=diarization.turns,
                segments=assign_speakers(segments, diarization.turns),
                product_mentions=product_mentions,
            )
            stage_seconds["assembly"] = time.perf_counter() - stage_started_at
        self.last_runtime = PipelineRuntime(
            audio_seconds=canonical_audio.duration_seconds,
            total_seconds=time.perf_counter() - started_at,
            stages=stage_seconds,
            canonical_audio=canonical_audio,
        )
        return result

    def transcribe_to_json(self, audio_path: Path, output_path: Path) -> DiarizedTranscript:
        result = self.transcribe(audio_path)
        output_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result
