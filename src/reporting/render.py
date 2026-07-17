"""Console and Typst renderers for a diarized transcript."""

from __future__ import annotations

import json
from pathlib import Path

from src.models.diarization import DiarizedTranscript


# This is a decoder ranking score, not a probability.  The default comes from
# the dev-split calibration targeting at least 95% precision.
DEFAULT_CONFIRMED_SCORE_THRESHOLD = 121.53621653914357


def terminology_status(
    score: float, confirmed_score_threshold: float = DEFAULT_CONFIRMED_SCORE_THRESHOLD
) -> str:
    """Classify a detected term for the reviewer-facing report."""
    if score < confirmed_score_threshold:
        return "review"
    return "confirmed"


def dialogue_rows(
    result: DiarizedTranscript,
    confirmed_score_threshold: float | None = None,
    merge_same_speaker_gap_seconds: float = 0.0,
) -> list[dict]:
    """Combine ASR segments into speaker-attributed, reviewer-facing dialogue rows."""
    if merge_same_speaker_gap_seconds < 0:
        raise ValueError("merge_same_speaker_gap_seconds must be non-negative.")
    rows = []
    for segment in result.segments:
        mentions = [
            mention
            for mention in result.product_mentions
            if min(segment.end, mention.end) > max(segment.start, mention.start)
        ]
        terms = list(dict.fromkeys(mention.canonical_name for mention in mentions))
        term_details = []
        seen_term_ids = set()
        for mention in mentions:
            if mention.product_id in seen_term_ids:
                continue
            seen_term_ids.add(mention.product_id)
            term_details.append(
                {
                    "name": mention.canonical_name,
                    "score": mention.score,
                    "status": terminology_status(
                        mention.score,
                        confirmed_score_threshold
                        if confirmed_score_threshold is not None
                        else DEFAULT_CONFIRMED_SCORE_THRESHOLD,
                    ),
                }
            )
        rows.append(
            {
                "speaker_id": segment.speaker_id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "terms": terms,
                "term_details": term_details,
            }
        )
    return _merge_adjacent_speaker_rows(rows, merge_same_speaker_gap_seconds)


def _merge_adjacent_speaker_rows(rows: list[dict], max_gap_seconds: float) -> list[dict]:
    """Merge consecutive rows from one speaker separated by a short VAD pause."""
    merged: list[dict] = []
    for row in rows:
        if (
            merged
            and row["speaker_id"] == merged[-1]["speaker_id"]
            and row["start"] - merged[-1]["end"] <= max_gap_seconds
        ):
            previous = merged[-1]
            previous["end"] = row["end"]
            previous["text"] = " ".join((previous["text"], row["text"])).strip()
            _merge_row_terms(previous, row)
        else:
            merged.append(
                {
                    **row,
                    "terms": list(row["terms"]),
                    "term_details": list(row["term_details"]),
                }
            )
    return merged


def _merge_row_terms(previous: dict, row: dict) -> None:
    """Keep term labels and details unique after joining dialogue rows."""
    previous["terms"] = list(dict.fromkeys((*previous["terms"], *row["terms"])))
    existing_names = {detail["name"] for detail in previous["term_details"]}
    for detail in row["term_details"]:
        if detail["name"] not in existing_names:
            previous["term_details"].append(detail)
            existing_names.add(detail["name"])


def format_dialogue(rows: list[dict]) -> str:
    """Render dialogue rows in the compact format intended for people."""
    lines = []
    previous_speaker = None
    for row in rows:
        if previous_speaker is not None and row["speaker_id"] != previous_speaker:
            lines.append("")
        speaker = (row["speaker_id"] or "UNKNOWN").replace("_", "")
        terms = ", ".join(row["terms"]) or "—"
        lines.extend(
            (
                f"[{speaker}] {_seconds(row['start'])} - {_seconds(row['end'])}",
                f"{row['text']} | {terms}",
            )
        )
        previous_speaker = row["speaker_id"]
    return "\n".join(lines)


def format_quality_metrics(metrics: dict) -> str:
    """Render available quality metrics for the console."""
    if not metrics["available"]:
        return f"Quality metrics: unavailable ({metrics['reason']})"

    lines = [
        "Quality metrics:",
        f"ASR: WER {metrics['asr']['wer']:.2%}, CER {metrics['asr']['cer']:.2%}",
    ]
    if "diarization" in metrics:
        lines.append(f"Diarization: DER {metrics['diarization']['der']:.2%}")
    if "speaker_count" in metrics:
        speaker_count = metrics["speaker_count"]
        lines.append(
            "Speakers: "
            f"{speaker_count['predicted']}/{speaker_count['expected']} "
            f"({'OK' if speaker_count['correct'] else 'mismatch'})"
        )
    if "terminology" in metrics:
        terminology = metrics["terminology"]
        lines.append(
            "Terminology: "
            f"P {terminology['precision']:.2%}, R {terminology['recall']:.2%}, "
            f"F1 {terminology['f1']:.2%}"
        )
    return "\n".join(lines)


def format_runtime_metrics(runtime: dict) -> str:
    """Render stage timings and runtime configuration for the console."""
    stage_lines = [
        f"{name}: {seconds:.1f}s" for name, seconds in runtime["stages"].items()
    ]
    canonical_audio = runtime["canonical_audio"]
    devices = ", ".join(
        f"{stage}={device}" for stage, device in runtime["devices"].items()
    )
    return "\n".join(
        [
            "Runtime:",
            "Canonical audio: "
            f"{canonical_audio['channels']} channel(s), "
            f"{canonical_audio['sample_rate']} Hz, {canonical_audio['subtype']}",
            *stage_lines,
            f"Total: {runtime['total_seconds']:.1f}s",
            f"Audio: {runtime['audio_seconds']:.1f}s",
            f"RTF: {runtime['realtime_factor']:.2f}x",
            f"Devices: {devices}",
        ]
    )


def write_typst_document(
    project_root: Path,
    json_path: Path,
    confirmed_color: str = "#1b7f3a",
    review_color: str = "#a96800",
) -> Path:
    """Write the fixed Typst report that reads the generated JSON result."""
    typst_path = project_root / "report.typ"
    try:
        json_reference = json_path.resolve().relative_to(project_root).as_posix()
    except ValueError as error:
        raise ValueError(
            f"JSON must be inside the project directory {project_root}: {json_path}"
        ) from error

    typst_path.write_text(
        _typst_template(json_reference, json_path, confirmed_color, review_color),
        encoding="utf-8",
    )
    return typst_path


def _seconds(value: float) -> str:
    return f"{value:.1f}s"


def _typst_template(
    json_reference: str,
    json_path: Path,
    confirmed_color: str,
    review_color: str,
) -> str:
    return f'''#set page(paper: "a4", margin: 18mm)
#set text(font: "DejaVu Sans", size: 10pt)

#let report = json({json.dumps(json_reference, ensure_ascii=False)})
#let seconds(value) = str(calc.round(value * 10) / 10) + " s"
#let term_label(term) = {{
  if term.at("status") == "confirmed" {{
    text(fill: rgb({json.dumps(confirmed_color)}))[#term.at("name")]
  }} else {{
    text(fill: rgb({json.dumps(review_color)}))[#term.at("name") + " (проверить)"]
  }}
}}

= Файл {json_path}

#for row in report.at("dialogue") [
  #block(
    width: 100%,
    inset: 8pt,
    radius: 4pt,
    fill: luma(245),
  )[
    *[#row.at("speaker_id")]* \\
    #seconds(row.at("start")) — #seconds(row.at("end"))

    #row.at("text")

    _Термины:_ #if row.at("term_details").len() > 0 [
      #for (index, term) in row.at("term_details").enumerate() {{
        if index > 0 [, ]
        term_label(term)
      }}
    ] else [—]
  ]

  #v(8pt)
]

= Метрики качества

#let quality = report.at("quality")
#if quality.at("available") [
  - ASR WER: #(quality.at("asr").at("wer") * 100)%
  - ASR CER: #(quality.at("asr").at("cer") * 100)%

  #if "diarization" in quality [
    - DER: #(quality.at("diarization").at("der") * 100)%
  ]

  #if "terminology" in quality [
    - Терминология — Precision: #(quality.at("terminology").at("precision") * 100)%
    - Терминология — Recall: #(quality.at("terminology").at("recall") * 100)%
    - Терминология — F1: #(quality.at("terminology").at("f1") * 100)%
  ]
] else [
  Метрики недоступны: #quality.at("reason")
]

= Производительность

#let runtime = report.at("runtime")
- Длительность аудио: #seconds(runtime.at("audio_seconds"))
- Полное время: #seconds(runtime.at("total_seconds"))
- RTF: #runtime.at("realtime_factor")
- Формат: #runtime.at("canonical_audio").at("sample_rate") Hz,
  mono, #runtime.at("canonical_audio").at("subtype")
'''
