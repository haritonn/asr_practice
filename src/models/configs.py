from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class ContextGraphConfig:
    context_weight: float
    keyword_threshold: float
    ctc_alignment_weight: float


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
    model_path: str
    window_size: int
    device: str
    sample_rate: int
    speech_threshold: float
    silence_threshold: float
    min_speech_duration_ms: int
    min_silence_duration_ms: int
