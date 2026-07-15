"""Run the end-to-end speaker-attributed speech-to-text pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _evaluate_result(result, reference_path: Path) -> dict:
    """Score one pipeline result against a small JSON reference sidecar."""
    import jiwer

    from src.metrics import normalize_text

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    expected_products = set(reference.get("expected_product_ids", []))
    predicted_products = {mention.product_id for mention in result.product_mentions}
    true_positive = len(expected_products & predicted_products)
    false_positive = len(predicted_products - expected_products)
    false_negative = len(expected_products - predicted_products)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "reference_file": str(reference_path),
        "reference_text": reference["reference"],
        "asr": {
            "wer": jiwer.wer(normalize_text(reference["reference"]), normalize_text(result.text)),
            "cer": jiwer.cer(normalize_text(reference["reference"]), normalize_text(result.text)),
            "wil": jiwer.wil(normalize_text(reference["reference"]), normalize_text(result.text)),
            "wip": jiwer.wip(normalize_text(reference["reference"]), normalize_text(result.text)),
        },
        "terminology": {
            "expected_product_ids": sorted(expected_products),
            "predicted_product_ids": sorted(predicted_products),
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        },
        "speaker_count_correct": result.num_speakers == reference.get("expected_num_speakers"),
    }


def main() -> None:
    if "--benchmark" in sys.argv:
        sys.argv.remove("--benchmark")
        from scripts.evaluate_e2e import main as benchmark_main

        benchmark_main()
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Input audio file")
    parser.add_argument("--output", type=Path, help="Path to the resulting JSON")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Compute metrics from --reference-file or <audio>.reference.json.",
    )
    parser.add_argument(
        "--reference-file",
        type=Path,
        help="JSON with reference text, expected_product_ids and expected_num_speakers.",
    )
    parser.add_argument("--asr-backend", choices=("whisper", "gigaam"), default="whisper")
    parser.add_argument("--asr-model", default="medium", help="faster-whisper model name or path")
    parser.add_argument("--asr-device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--asr-compute-type", default=None, help="For example: int8, float16 or int8_float16")
    parser.add_argument("--asr-beam-size", type=int, default=1)
    parser.add_argument("--language", default="ru")
    parser.add_argument("--gigaam-model-id", default="ai-sage/GigaAM-Multilingual")
    parser.add_argument("--gigaam-revision", default="large_ctc")
    parser.add_argument("--gigaam-batch-size", type=int, default=4)
    parser.add_argument(
        "--allow-gigaam-download",
        action="store_true",
        help="Allow GigaAM model download instead of requiring a local cache.",
    )

    parser.add_argument("--vad-device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--vad-sample-rate", type=int, default=16_000)
    parser.add_argument("--vad-speech-threshold", type=float, default=0.7)
    parser.add_argument("--vad-min-speech-ms", type=int, default=100)
    parser.add_argument("--vad-min-silence-ms", type=int, default=500)

    parser.add_argument("--diarizer-backend", choices=("nemo", "pyannote"), default="nemo")
    parser.add_argument("--diarization-device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument(
        "--nemo-vad-model",
        type=Path,
        default=ROOT / "models/nemo/vad_multilingual_marblenet.nemo",
    )
    parser.add_argument(
        "--nemo-speaker-model", type=Path, default=ROOT / "models/nemo/titanet-l.nemo"
    )
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--max-num-speakers", type=int, default=8)
    parser.add_argument("--diarization-onset", type=float, default=0.7)
    parser.add_argument("--diarization-offset", type=float, default=0.3)
    parser.add_argument("--diarization-pad-onset", type=float, default=0.2)
    parser.add_argument("--diarization-pad-offset", type=float, default=0.2)
    parser.add_argument("--diarization-min-duration-on", type=float, default=0.2)
    parser.add_argument("--diarization-min-duration-off", type=float, default=0.2)
    parser.add_argument("--diarization-max-rp-threshold", type=float, default=0.3)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--pyannote-model-id", default="pyannote/speaker-diarization-community-1")
    parser.add_argument("--pyannote-exclusive", action="store_true")
    parser.add_argument("--disable-terminology", action="store_true")
    parser.add_argument("--terminology-device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--terminology-catalog", type=Path, default=ROOT / "resources/fire_catalog.json")
    parser.add_argument("--terminology-model", type=Path, default=ROOT / "models/stt_ru_conformer_ctc_large.nemo")
    parser.add_argument("--terminology-context-weight", type=float, default=12.0)
    parser.add_argument("--terminology-keyword-threshold", type=float, default=-12.0)
    args = parser.parse_args()

    output = args.output or args.audio.with_suffix(".diarized.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    from src.asr.gigaam import GigaAmAsr
    from src.asr.terminology import CtcTerminologyRecognizer
    from src.asr.whisper import WhisperAsr
    from src.diarization.nemo import NemoDiarizer
    from src.models.configs import (
        GigaAmConfig,
        NemoDiarizationConfig,
        PyannoteDiarizationConfig,
        SileroConfig,
        TerminologyConfig,
        WhisperConfig,
    )
    from src.pipeline import DiarizedSpeechPipeline
    from src.vad.silero_vad import SileroVoiceDetection

    if args.asr_backend == "whisper":
        asr = WhisperAsr(
            WhisperConfig(
                model_size_or_path=args.asr_model,
                device=args.asr_device,
                compute_type=args.asr_compute_type,
                beam_size=args.asr_beam_size,
                language=args.language,
                word_timestamps=True,
            )
        )
    else:
        asr = GigaAmAsr(
            GigaAmConfig(
                model_id=args.gigaam_model_id,
                revision=args.gigaam_revision,
                device=args.asr_device,
                batch_size=args.gigaam_batch_size,
                local_files_only=not args.allow_gigaam_download,
            )
        )

    terminology = None
    if not args.disable_terminology:
        terminology = CtcTerminologyRecognizer(
            TerminologyConfig(
                catalog_path=args.terminology_catalog,
                model_path=args.terminology_model,
                device=args.terminology_device,
                context_weight=args.terminology_context_weight,
                keyword_threshold=args.terminology_keyword_threshold,
            )
        )

    if args.diarizer_backend == "nemo":
        diarizer = NemoDiarizer(
            NemoDiarizationConfig(
                device=args.diarization_device,
                vad_model=str(args.nemo_vad_model),
                speaker_model=str(args.nemo_speaker_model),
                num_speakers=args.num_speakers,
                max_num_speakers=args.max_num_speakers,
                onset=args.diarization_onset,
                offset=args.diarization_offset,
                pad_onset=args.diarization_pad_onset,
                pad_offset=args.diarization_pad_offset,
                min_duration_on=args.diarization_min_duration_on,
                min_duration_off=args.diarization_min_duration_off,
                max_rp_threshold=args.diarization_max_rp_threshold,
            )
        )
    else:
        from src.diarization.pyannote import PyannoteCommunityDiarizer

        diarizer = PyannoteCommunityDiarizer(
            PyannoteDiarizationConfig(
                model_id=args.pyannote_model_id,
                device=args.diarization_device,
                num_speakers=args.num_speakers,
                max_speakers=None if args.num_speakers is not None else args.max_num_speakers,
                output="exclusive" if args.pyannote_exclusive else "regular",
            ),
            token=args.hf_token,
        )

    pipeline = DiarizedSpeechPipeline(
        asr=asr,
        vad=SileroVoiceDetection(
            SileroConfig(
                model_path=Path(),
                window_size=512,
                device=args.vad_device,
                sample_rate=args.vad_sample_rate,
                speech_threshold=args.vad_speech_threshold,
                silence_threshold=0.35,
                min_speech_duration_ms=args.vad_min_speech_ms,
                min_silence_duration_ms=args.vad_min_silence_ms,
            )
        ),
        diarizer=diarizer,
        terminology=terminology,
    )
    result = pipeline.transcribe(args.audio)
    payload = result.to_dict()
    if args.evaluate:
        reference_path = args.reference_file or args.audio.with_suffix(".reference.json")
        if not reference_path.is_file():
            raise FileNotFoundError(
                f"Evaluation reference is missing: {reference_path}. "
                "Pass --reference-file or create the sidecar JSON."
            )
        payload["evaluation"] = _evaluate_result(result, reference_path)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output}")
    print(f"Detected speakers: {result.num_speakers}")
    print(f"Transcript: {result.text}")
    if args.evaluate:
        metrics = payload["evaluation"]
        print(
            "ASR: "
            f"WER {metrics['asr']['wer']:.2%}, CER {metrics['asr']['cer']:.2%}, "
            f"WIL {metrics['asr']['wil']:.2%}, WIP {metrics['asr']['wip']:.2%}"
        )
        print(
            "Terminology: "
            f"P {metrics['terminology']['precision']:.2%}, "
            f"R {metrics['terminology']['recall']:.2%}, "
            f"F1 {metrics['terminology']['f1']:.2%}"
        )


if __name__ == "__main__":
    main()
