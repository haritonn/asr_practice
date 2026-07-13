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
