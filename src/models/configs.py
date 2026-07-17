from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class ContextGraphConfig:
    context_weight: float
    keyword_threshold: float
    ctc_alignment_weight: float


@dataclass(frozen=True, slots=True)
class WhisperConfig:
    model_size_or_path: str | Path
    device: str
    compute_type: str | None
    beam_size: int
    language: str
    word_timestamps: bool

    def model_kwargs(self) -> dict:
        kwargs = {"model_size_or_path": self.model_size_or_path, "device": self.device}
        if self.compute_type is not None:
            kwargs["compute_type"] = self.compute_type
        return kwargs

    def transcribe_kwargs(self) -> dict:
        return {
            "beam_size": self.beam_size,
            "language": self.language,
            "word_timestamps": self.word_timestamps,
        }


@dataclass(frozen=True, slots=True)
class GigaAmConfig:
    """Configuration for a locally cached GigaAM CTC checkpoint."""

    model_id: str = "ai-sage/GigaAM-Multilingual"
    revision: str = "large_ctc"
    device: str = "cuda"
    batch_size: int = 4
    local_files_only: bool = True


@dataclass(frozen=True, slots=True)
class TerminologyConfig:
    catalog_path: Path
    model_path: Path
    device: str = "cpu"
    context_weight: float = 12.0
    keyword_threshold: float = -12.0
    ctc_alignment_weight: float = 0.0
    # Calibrated on the synthetic dev split for 95% precision.  This controls
    # reviewer-facing colouring only; keyword_threshold still controls recall.
    confirmed_score_threshold: float = 121.53621653914357


@dataclass(frozen=True, slots=True)
class PyannoteDiarizationConfig:
    """Configuration for the isolated pyannote Community-1 runtime."""

    model_id: str = "pyannote/speaker-diarization-community-1"
    device: str = "cpu"
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    output: Literal["regular", "exclusive"] = "regular"

    def inference_kwargs(self) -> dict[str, int]:
        if self.num_speakers is not None:
            return {"num_speakers": self.num_speakers}
        result = {}
        if self.min_speakers is not None:
            result["min_speakers"] = self.min_speakers
        if self.max_speakers is not None:
            result["max_speakers"] = self.max_speakers
        return result


@dataclass(frozen=True, slots=True)
class NemoDiarizationConfig:
    """Configuration for the isolated NeMo runtime."""

    vad_model: str = "vad_multilingual_marblenet"
    speaker_model: str = "titanet_large"
    device: str = "cpu"
    max_num_speakers: int = 8
    num_speakers: int | None = None
    onset: float = 0.7
    offset: float = 0.3
    pad_onset: float = 0.2
    pad_offset: float = 0.2
    min_duration_on: float = 0.2
    min_duration_off: float = 0.2
    max_rp_threshold: float = 0.3
    external_vad_manifest: str | None = None
    num_workers: int = 0


@dataclass(frozen=True, slots=True)
class SileroConfig:
    """Parameters used by the Silero VAD adapter."""

    device: str = "cpu"
    sample_rate: int = 16_000
    speech_threshold: float = 0.7
    silence_threshold: float = 0.35
    min_speech_duration_ms: int = 100
    min_silence_duration_ms: int = 500
