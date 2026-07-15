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
        result.append(
            DiarizedTranscriptSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                speaker_id=speaker_for_interval(segment.start, segment.end, turns),
            )
        )
    return result


def speaker_for_interval(
    start: float, end: float, turns: Sequence[SpeakerTurn]
) -> str | None:
    """Return the speaker with the greatest overlap for a time interval."""
    overlap_by_speaker: dict[str, float] = defaultdict(float)
    for turn in turns:
        overlap = max(0.0, min(end, turn.end) - max(start, turn.start))
        if overlap > 0:
            overlap_by_speaker[turn.speaker_id] += overlap
    if not overlap_by_speaker:
        return None
    maximum = max(overlap_by_speaker.values())
    return min(
        speaker
        for speaker, overlap in overlap_by_speaker.items()
        if abs(overlap - maximum) <= 1e-9
    )
