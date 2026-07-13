"""Create JSON and optional RTTM speaker-diarization artifacts for one audio file."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from src.diarization import PyannoteCommunityDiarizer, assign_speakers
from src.models.configs import PyannoteDiarizationConfig
from src.models.diarization import TimedTranscriptSegment


def _load_transcript(path: Path) -> list[TimedTranscriptSegment]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Transcript JSON must be a list of {start, end, text} objects.")
    try:
        return [
            TimedTranscriptSegment(
                start=float(item["start"]), end=float(item["end"]), text=str(item["text"])
            )
            for item in data
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Transcript JSON must be a list of {start, end, text} objects.") from error


def _write_rttm(path: Path, recording_id: str, turns) -> None:
    with path.open("w", encoding="utf-8") as destination:
        for turn in turns:
            destination.write(
                f"SPEAKER {recording_id} 1 {turn.start:.3f} "
                f"{turn.end - turn.start:.3f} <NA> <NA> {turn.speaker_id} <NA> <NA>\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-rttm", type=Path)
    parser.add_argument("--transcript-json", type=Path)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--exclusive", action="store_true")
    args = parser.parse_args()

    config = PyannoteDiarizationConfig(
        device=args.device,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        output="exclusive" if args.exclusive else "regular",
    )
    result = PyannoteCommunityDiarizer(config, token=args.hf_token).diarize(args.audio)
    payload = {
        "audio_path": str(args.audio.resolve()),
        "num_speakers": result.num_speakers,
        "turns": [asdict(turn) for turn in result.turns],
    }
    if args.transcript_json:
        payload["transcript"] = [
            asdict(segment)
            for segment in assign_speakers(_load_transcript(args.transcript_json), result.turns)
        ]

    output_json = args.output_json or args.audio.with_suffix(".diarization.json")
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_rttm:
        _write_rttm(args.output_rttm, args.audio.stem, result.turns)
    print(output_json)


if __name__ == "__main__":
    main()
