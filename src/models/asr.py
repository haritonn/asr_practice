from dataclasses import dataclass


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Segment:
    text: str
    start: float
    end: float
    words: list[Word]


@dataclass
class TranscribeResult:
    text: str
    segments: list[Segment]
    language: str
