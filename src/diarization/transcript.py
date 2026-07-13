"""Attach timestamped ASR segments to anonymous speaker turns."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from src.models.diarization import (
    DiarizedTranscriptSegment,
    SpeakerTurn,
    TimedTranscriptSegment,
)


def assign_speakers(
    segments: Sequence[TimedTranscriptSegment], turns: Sequence[SpeakerTurn]
) -> list[DiarizedTranscriptSegment]:
    """Assign every ASR segment to the speaker with greatest time overlap."""
    result = []
    for segment in segments:
        overlap_by_speaker: dict[str, float] = defaultdict(float)
        for turn in turns:
            overlap = max(0.0, min(segment.end, turn.end) - max(segment.start, turn.start))
            if overlap > 0:
                overlap_by_speaker[turn.speaker_id] += overlap
        if overlap_by_speaker:
            maximum = max(overlap_by_speaker.values())
            candidates = [
                speaker
                for speaker, overlap in overlap_by_speaker.items()
                if abs(overlap - maximum) <= 1e-9
            ]
            speaker_id = min(candidates)
        else:
            speaker_id = None
        result.append(
            DiarizedTranscriptSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                speaker_id=speaker_id,
            )
        )
    return result
