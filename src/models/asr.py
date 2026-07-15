from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProductMention:
    product_id: str
    canonical_name: str
    score: float
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class Word:
    text: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class Segment:
    text: str
    start: float
    end: float
    words: list[Word]


@dataclass(frozen=True, slots=True)
class TranscribeResult:
    text: str
    segments: list[Segment]
    language: str | None
