from dataclasses import dataclass


@dataclass(slots=True)
class SpeechSegment:
    start: float
    end: float
