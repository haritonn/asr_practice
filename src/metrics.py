from typing import List

import jiwer

from src.models.inference import InferenceResults, Metrics


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


def compute_metrics(inference_data: List[InferenceResults]) -> Metrics:
    """Computing & returning all implemented metrics (currently: CER, WER)"""
    refs = [obj.ground_truth for obj in inference_data]
    preds = [obj.predicted for obj in inference_data]
    if len(refs) != len(preds):
        raise ValueError("refs and preds must be the same length")
    normalized_refs, normalized_preds = (
        [normalize_text(text) for text in refs],
        [normalize_text(text) for text in preds],
    )

    cer_res, wer_res = (
        compute_cer(normalized_refs, normalized_preds),
        compute_wer(normalized_refs, normalized_preds),
    )
    wil_res, wip_res = (
        compute_wil(normalized_refs, normalized_preds),
        compute_wip(normalized_refs, normalized_preds),
    )

    per_item = []
    for i, res in enumerate(inference_data):
        per_item.append(
            {
                "audio_path": res.audio_path,
                "ground_truth": res.ground_truth,
                "predicted": res.predicted,
                "wer": wer_res["sample_scores"][i],
                "cer": cer_res["sample_scores"][i],
                "wil": wil_res["sample_scores"][i],
                "wip": wip_res["sample_scores"][i],
            }
        )

    return Metrics(
        wer=wer_res["average"],
        cer=cer_res["average"],
        wil=wil_res["average"],
        wip=wip_res["average"],
        per_item=per_item,
    )
