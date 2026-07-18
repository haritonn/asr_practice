import json

import jiwer
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

from src.metrics import normalize_text


def quality_metrics(result, reference_path):
    """Evaluate a result when a sidecar reference is available for the audio."""
    if not reference_path.is_file():
        return {
            "available": False,
            "reason": f"Reference file is missing: {reference_path}",
        }

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_turns = reference.get("turns", [])
    reference_text = reference.get("reference") or " ".join(
        turn["text"] for turn in reference_turns
    )
    if not reference_text:
        return {
            "available": False,
            "reason": f"Reference file has no reference text: {reference_path}",
        }

    metrics = {
        "available": True,
        "reference_file": str(reference_path),
        "asr": {
            "wer": jiwer.wer(
                normalize_text(reference_text), normalize_text(result.text)
            ),
            "cer": jiwer.cer(
                normalize_text(reference_text), normalize_text(result.text)
            ),
        },
    }
    expected_speakers = reference.get(
        "num_speakers", reference.get("expected_num_speakers")
    )
    if expected_speakers is not None:
        metrics["speaker_count"] = {
            "expected": expected_speakers,
            "predicted": result.num_speakers,
            "correct": result.num_speakers == expected_speakers,
        }
    if reference_turns:
        metrics["diarization"] = _diarization_metrics(result, reference_turns)
    if (expected_products := reference.get("expected_product_ids")) is not None:
        metrics["terminology"] = _terminology_metrics(result, expected_products)
    return metrics


def _diarization_metrics(result, reference_turns):
    reference_annotation = Annotation(uri="reference")
    hypothesis_annotation = Annotation(uri="hypothesis")
    for turn in reference_turns:
        reference_annotation[Segment(turn["start"], turn["end"])] = str(
            turn["speaker_id"]
        )
    for turn in result.speaker_turns:
        hypothesis_annotation[Segment(turn.start, turn.end)] = turn.speaker_id
    diarization_error_rate = DiarizationErrorRate(collar=0.25, skip_overlap=False)
    return {
        "der": diarization_error_rate(reference_annotation, hypothesis_annotation),
        "collar_seconds": 0.25,
    }


def _terminology_metrics(result, expected_product_ids):
    expected = set(expected_product_ids)
    predicted = {mention.product_id for mention in result.product_mentions}
    true_positive = len(expected & predicted)
    precision = true_positive / max(len(predicted), 1)
    recall = true_positive / max(len(expected), 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }
