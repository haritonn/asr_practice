"""Reference/prediction pair used only when an evaluation set has labels."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class InferenceResults:
    audio_path: Path
    ground_truth: str
    predicted: str
