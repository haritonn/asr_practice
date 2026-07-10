"""Calibrate CTC graph weight on development and report held-out test metrics."""

from __future__ import annotations

import json
from zipfile import BadZipFile
from collections import defaultdict
from pathlib import Path

import numpy as np
from tabulate import tabulate

from src.catalog import ProductCatalog
from src.context_graph import ContextGraphRecognizer
from src.metrics import compute_metrics
from src.models.configs import ContextGraphConfig
from src.models.inference import InferenceResults
from src.nemo_runtime import quiet_nemo_transcribe, silence_nemo_configuration_logs


def _load_config() -> tuple[dict, Path]:
    root = Path(__file__).resolve().parents[1]
    with (root / "config" / "benchmark.json").open(encoding="utf-8") as source:
        return json.load(source), root


def _load_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _cache_path(cache_dir: Path, record_id: str) -> Path:
    return cache_dir / f"{record_id}.npz"


def _read_cached(record: dict, cache_dir: Path) -> dict | None:
    path = _cache_path(cache_dir, record["id"])
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as cached:
            return {
                **record,
                "base_text": str(cached["base_text"]),
                "logprobs": cached["logprobs"],
            }
    except (BadZipFile, OSError, ValueError):
        path.unlink(missing_ok=True)
        return None


def _transcribe(model, records: list[dict], cache_dir: Path) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = {record["id"]: _read_cached(record, cache_dir) for record in records}
    missing = [record for record in records if cached[record["id"]] is None]
    if missing:
        print(f"Computing {len(missing)} missing CTC alignments; completed batches are cached.")
    for offset in range(0, len(missing), 16):
        batch = missing[offset : offset + 16]
        with quiet_nemo_transcribe():
            hypotheses = model.transcribe(
                [record["audio_path"] for record in batch],
                batch_size=16,
                return_hypotheses=True,
                verbose=False,
            )
        for record, hypothesis in zip(batch, hypotheses):
            alignment = hypothesis.alignments
            if hasattr(alignment, "cpu"):
                alignment = alignment.cpu().numpy()
            path = _cache_path(cache_dir, record["id"])
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as destination:
                np.savez(
                    destination,
                    base_text=np.array(hypothesis.text),
                    logprobs=np.asarray(alignment),
                )
            temporary.replace(path)
        print(f"Cached {offset + len(batch)}/{len(missing)} missing alignments.", flush=True)
    return [_read_cached(record, cache_dir) for record in records]


def _evaluate(
    records: list[dict],
    recognizer: ContextGraphRecognizer,
) -> tuple[dict, list[dict]]:
    scored = []
    true_positive = false_positive = false_negative = 0
    negative_total = negative_with_prediction = 0
    for record in records:
        mentions = recognizer.recognize(record["logprobs"], 0.0, 1.0)
        predicted_ids = sorted({mention.product_id for mention in mentions})
        expected_ids = set(record["expected_product_ids"])
        predicted = set(predicted_ids)
        true_positive += len(expected_ids & predicted)
        false_positive += len(predicted - expected_ids)
        false_negative += len(expected_ids - predicted)
        if not expected_ids:
            negative_total += 1
            negative_with_prediction += bool(predicted)
        scored.append(
            {
                **{key: value for key, value in record.items() if key != "logprobs"},
                "predicted_product_ids": predicted_ids,
                "predicted_canonical_names": [
                    recognizer.catalog.products[item].canonical_name for item in predicted_ids
                ],
                "correct": expected_ids == predicted,
            }
        )
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return (
        {
            "samples": len(records),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "negative_false_positive_rate": negative_with_prediction / max(negative_total, 1),
        },
        scored,
    )


def _print_summary(metrics: dict) -> None:
    print("Held-out product-recognition metrics:")
    print(
        tabulate(
            [
                [
                    metrics["samples"],
                    f"{metrics['precision']:.2%}",
                    f"{metrics['recall']:.2%}",
                    f"{metrics['f1']:.2%}",
                    f"{metrics['negative_false_positive_rate']:.2%}",
                ]
            ],
            headers=["Samples", "Precision", "Recall", "F1", "Negative FPR"],
            tablefmt="rounded_outline",
        )
    )


def _base_asr_metrics(records: list[dict]) -> dict:
    metrics = compute_metrics(
        [
            InferenceResults(
                audio_path=Path(record["audio_path"]),
                ground_truth=record["tts_text"],
                predicted=record["base_text"],
            )
            for record in records
        ]
    )
    print("\nBase CTC transcription metrics on held-out test:")
    print(metrics.pretty_print())
    return {"wer": metrics.wer, "cer": metrics.cer, "wil": metrics.wil, "wip": metrics.wip}


def _print_by_condition(results: list[dict]) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in results:
        groups[item["condition"]].append(item)
    rows = []
    for condition, items in sorted(groups.items()):
        entity_items = [item for item in items if item["expected_product_ids"]]
        exact = sum(item["correct"] for item in entity_items) / max(len(entity_items), 1)
        negative_items = [item for item in items if not item["expected_product_ids"]]
        fpr = sum(bool(item["predicted_product_ids"]) for item in negative_items) / max(
            len(negative_items), 1
        )
        rows.append([condition, len(items), f"{exact:.2%}", f"{fpr:.2%}"])
    print("\nHeld-out test by acoustic condition:")
    print(
        tabulate(
            rows,
            headers=["Condition", "Samples", "Exact entity set", "Negative FPR"],
            tablefmt="rounded_outline",
        )
    )


def _print_predictions(results: list[dict]) -> None:
    """Full held-out view, equivalent to the old main.py per-item output."""
    rows = [
        [
            item["id"],
            item["condition"],
            item["tts_text"],
            item["base_text"],
            ", ".join(item["expected_product_ids"]) or "-",
            ", ".join(item["predicted_product_ids"]) or "-",
            "yes" if item["correct"] else "no",
        ]
        for item in results
    ]
    print("\nHeld-out ground truth vs prediction (global graph):")
    print(
        tabulate(
            rows,
            headers=[
                "ID",
                "Condition",
                "Ground truth (TTS)",
                "Base ASR prediction",
                "Expected products",
                "Predicted products",
                "Exact",
            ],
            tablefmt="rounded_outline",
        )
    )


def _print_errors(results: list[dict]) -> None:
    errors = [item for item in results if not item["correct"]]
    rows = [
        [
            item["id"],
            item["condition"],
            ", ".join(item["expected_product_ids"]) or "-",
            ", ".join(item["predicted_product_ids"]) or "-",
            item["base_text"],
        ]
        for item in errors[:30]
    ]
    print("\nFirst 30 held-out errors:")
    print(
        tabulate(
            rows or [["No errors", "", "", "", ""]],
            headers=["ID", "Condition", "Expected IDs", "Predicted IDs", "Base ASR text"],
            tablefmt="rounded_outline",
        )
    )


def main() -> None:
    config, root = _load_config()
    output = root / config["output_dir"]
    manifest_path = output / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError("Synthetic corpus is missing. Run `uv run python -m scripts.build_synthetic_corpus`.")
    model_path = root / config["model_path"]
    if not model_path.is_file():
        raise FileNotFoundError("NeMo checkpoint is missing. Run `uv run python -m scripts.download_model`.")

    silence_nemo_configuration_logs()
    import nemo.collections.asr as nemo_asr

    catalog = ProductCatalog.load(root / config["catalog_path"])
    records = _load_manifest(manifest_path)
    model = nemo_asr.models.ASRModel.restore_from(str(model_path), map_location=config["device"]).eval()
    print(f"Preparing CTC alignments for {len(records)} records...")
    aligned = _transcribe(model, records, output / "alignments")
    dev = [record for record in aligned if record["split"] == "dev"]
    test = [record for record in aligned if record["split"] == "test"]

    calibration: list[tuple[float, dict]] = []
    for weight in config["candidate_context_weights"]:
        recognizer = ContextGraphRecognizer(
            model,
            catalog,
            ContextGraphConfig(
                context_weight=weight,
                keyword_threshold=config["keyword_threshold"],
                ctc_alignment_weight=config["ctc_alignment_weight"],
            ),
        )
        metrics, _ = _evaluate(dev, recognizer)
        calibration.append((weight, metrics))
    best_weight, development_metrics = max(
        calibration,
        key=lambda item: (item[1]["f1"], item[1]["precision"], item[1]["recall"]),
    )
    print("Development calibration (global graph):")
    print(
        tabulate(
            [
                [weight, f"{metrics['precision']:.2%}", f"{metrics['recall']:.2%}", f"{metrics['f1']:.2%}"]
                for weight, metrics in calibration
            ],
            headers=["Context weight", "Precision", "Recall", "F1"],
            tablefmt="rounded_outline",
        )
    )

    recognizer = ContextGraphRecognizer(
        model,
        catalog,
        ContextGraphConfig(
            context_weight=best_weight,
            keyword_threshold=config["keyword_threshold"],
            ctc_alignment_weight=config["ctc_alignment_weight"],
        ),
    )
    test_metrics, test_results = _evaluate(test, recognizer)
    print(f"\nSelected global context weight: {best_weight}")
    _print_summary(test_metrics)
    base_asr = _base_asr_metrics(test)
    _print_by_condition(test_results)
    _print_predictions(test_results)
    _print_errors(test_results)

    report = {
        "catalog_version": catalog.version,
        "selected_context_weight": best_weight,
        "development": development_metrics,
        "base_asr": base_asr,
        "test": test_metrics,
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "results.jsonl").open("w", encoding="utf-8") as destination:
        for item in test_results:
            destination.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\nSaved: {output / 'report.json'} and {output / 'results.jsonl'}")


if __name__ == "__main__":
    main()
