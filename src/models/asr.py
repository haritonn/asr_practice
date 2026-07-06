from dataclasses import dataclass


@dataclass(slots=True)
class Word:
    text: str
    start: float
    end: float


@dataclass(slots=True)
class Segment:
    text: str
    start: float
    end: float
    words: list[Word]


@dataclass(slots=True)
class TranscribeResult:
    text: str
    segments: list[Segment]
    language: str
