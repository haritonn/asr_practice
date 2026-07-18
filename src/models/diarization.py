from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    start: float
    end: float
    speaker_id: str


@dataclass(frozen=True, slots=True)
class DiarizationResult:
    turns: Sequence[SpeakerTurn]
    num_speakers: int


@dataclass(frozen=True, slots=True)
class TimedTranscriptSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class DiarizedTranscriptSegment:
    start: float
    end: float
    text: str
    speaker_id: str | None


@dataclass(frozen=True, slots=True)
class SpeakerAttributedProductMention:
    product_id: str
    canonical_name: str
    score: float
    start: float
    end: float
    speaker_id: str | None


@dataclass(frozen=True, slots=True)
class DiarizedTranscript:
    audio_path: Path
    text: str
    language: str | None
    num_speakers: int
    speaker_turns: Sequence[SpeakerTurn]
    segments: Sequence[DiarizedTranscriptSegment]
    product_mentions: Sequence[SpeakerAttributedProductMention] = ()

    def to_dict(self) -> dict:
        return {
            "audio_path": str(self.audio_path.resolve()),
            "text": self.text,
            "language": self.language,
            "num_speakers": self.num_speakers,
            "speaker_turns": [
                {"start": turn.start, "end": turn.end, "speaker_id": turn.speaker_id}
                for turn in self.speaker_turns
            ],
            "segments": [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "speaker_id": segment.speaker_id,
                }
                for segment in self.segments
            ],
            "product_mentions": [
                {
                    "product_id": mention.product_id,
                    "canonical_name": mention.canonical_name,
                    "score": mention.score,
                    "start": mention.start,
                    "end": mention.end,
                    "speaker_id": mention.speaker_id,
                }
                for mention in self.product_mentions
            ],
        }
