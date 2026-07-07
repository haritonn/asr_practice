from typing import Dict, List

import jiwer

from src.models.inference import InferenceResults


def normalize_text(text: str) -> str:
    """Removing anything except words for metrics computation"""
    transformation = jiwer.Compose(
        [jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()]
    )

    return transformation(text)


def compute_cer(refs: list[str], preds: list[str]):
    """Computing WER for single pair or as average for lists. Input should be normalized"""
    avg_cer = jiwer.cer(refs, preds)
    cers = [jiwer.cer(ref, pred) for ref, pred in zip(refs, preds)]

    return {"average": avg_cer, "sample_scores": cers}


def compute_wer(refs: list[str], preds: list[str]):
    """Computing WER for single pair or as average for lists. Input should be normalized"""
    avg_wer = jiwer.wer(refs, preds)
    wers = [jiwer.wer(ref, pred) for ref, pred in zip(refs, preds)]

    return {"average": avg_wer, "sample_scores": wers}


def compute_wil(refs: list[str], preds: list[str]):
    """Computing WIL for single pair or as average for lists. Input should be normalized"""
    avg_wil = jiwer.wil(refs, preds)
    wils = [jiwer.wil(ref, pred) for ref, pred in zip(refs, preds)]

    return {"average": avg_wil, "sample_scores": wils}


def compute_wip(refs: list[str], preds: list[str]):
    """Computing WIL for single pair or as average for lists. Input should be normalized"""
    avg_wip = jiwer.wip(refs, preds)
    wips = [jiwer.wip(ref, pred) for ref, pred in zip(refs, preds)]

    return {"average": avg_wip, "sample_scores": wips}


def compute_metrics(inference_data: List[InferenceResults]) -> Dict:
    """Computing & returning all implemented metrics (currently: CER, WER)"""
    refs = [obj.ground_truth for obj in inference_data]
    preds = [obj.predicted for obj in inference_data]
    if len(refs) != len(preds):
        raise ValueError("refs and preds must be the same length")
    normalized_refs, normalized_preds = (
        [normalize_text(text) for text in refs],
        [normalize_text(text) for text in preds],
    )

    cer, wer = (
        compute_cer(normalized_refs, normalized_preds),
        compute_wer(normalized_refs, normalized_preds),
    )
    wil, wip = (
        compute_wil(normalized_refs, normalized_preds),
        compute_wip(normalized_refs, normalized_preds),
    )

    return {"cer": cer, "wer": wer, "wil": wil, "wip": wip}
