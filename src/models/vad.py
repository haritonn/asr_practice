from dataclasses import dataclass
from typing import Set, Tuple


@dataclass(slots=True)
class SpeechSegment:
    start: float
    end: float
