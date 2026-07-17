"""Run the end-to-end speaker-attributed speech-to-text pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from src.asr.terminology import CtcTerminologyRecognizer
from src.asr.whisper import WhisperAsr
from src.app_config import AppConfig, load_app_config
from src.diarization.pyannote import PyannoteCommunityDiarizer
from src.evaluation.quality import quality_metrics
from src.pipeline import DiarizedSpeechPipeline
from src.reporting.render import (
    dialogue_rows,
    format_dialogue,
    format_quality_metrics,
    format_runtime_metrics,
    write_typst_document,
)
from src.runtime.preflight import format_preflight, run_preflight
from src.vad.silero_vad import SileroVoiceDetection


ROOT = Path(__file__).resolve().parents[1]
MIN_CUDA_MEMORY_BYTES = 8 * 1024**3


def _default_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    total_memory = torch.cuda.get_device_properties(0).total_memory
    return "cuda" if total_memory >= MIN_CUDA_MEMORY_BYTES else "cpu"


def _resolve_device(device: str) -> str:
    return _default_device() if device == "auto" else device


def _build_pipeline(
    config: AppConfig,
    asr_device: str,
    vad_device: str,
    terminology_device: str,
    diarization_device: str,
    num_speakers: int | None,
) -> DiarizedSpeechPipeline:
    asr_config = replace(config.asr, device=asr_device)
    terminology_config = replace(config.terminology, device=terminology_device)
    diarization_config = replace(
        config.diarization,
        device=diarization_device,
        num_speakers=num_speakers
        if num_speakers is not None
        else config.diarization.num_speakers,
    )
    asr = WhisperAsr(asr_config)
    terminology = CtcTerminologyRecognizer(terminology_config)
    diarizer = PyannoteCommunityDiarizer(diarization_config)
    return DiarizedSpeechPipeline(
        asr=asr,
        vad=SileroVoiceDetection(replace(config.vad, device=vad_device)),
        diarizer=diarizer,
        terminology=terminology,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Путь к аудиофайлу")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/default.yaml",
        help="YAML-конфигурация (по умолчанию: config/default.yaml)",
    )
    parser.add_argument("--output", type=Path, help="Куда записать JSON-результат")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        help="Переопределяет устройства ASR и терминологии из YAML",
    )
    parser.add_argument(
        "--num-speakers", type=int, help="Точное число спикеров, если оно известно"
    )
    parser.add_argument(
        "--typst", action="store_true", help="Создать report.typ в корне проекта"
    )
    args = parser.parse_args()

    config = load_app_config(args.config, ROOT)
    asr_device = _resolve_device(args.device or config.asr.device)
    vad_device = config.vad.device
    terminology_device = _resolve_device(args.device or config.terminology.device)
    diarization_device = _resolve_device(config.diarization.device)
    output = args.output or args.audio.with_suffix(".diarized.json")
    preflight = run_preflight(
        args.audio,
        output_path=output,
        asr_device=asr_device,
        vad_device=vad_device,
        terminology_device=terminology_device,
        diarization_device=diarization_device,
        terminology_catalog=config.terminology.catalog_path,
        terminology_model=config.terminology.model_path,
        pyannote_model_id=config.diarization.model_id,
    )
    print(format_preflight(preflight))
    if not preflight.passed:
        raise SystemExit("Preflight failed; no models were loaded.")

    output.parent.mkdir(parents=True, exist_ok=True)
    pipeline = _build_pipeline(
        config,
        asr_device,
        vad_device,
        terminology_device,
        diarization_device,
        args.num_speakers,
    )
    result = pipeline.transcribe(args.audio)
    runtime = pipeline.last_runtime
    if runtime is None:
        raise RuntimeError("Pipeline completed without runtime metrics.")

    dialogue = dialogue_rows(
        result,
        confirmed_score_threshold=pipeline.terminology_confirmation_threshold,
    )
    quality = quality_metrics(result, args.audio.with_suffix(".reference.json"))
    runtime_data = runtime.to_dict()
    runtime_data["devices"] = preflight.plan["devices"]
    payload = {
        **result.to_dict(),
        "dialogue": dialogue,
        "quality": quality,
        "runtime": runtime_data,
        "preflight": preflight.to_dict(),
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote {output}")
    if args.typst:
        print(
            f"Wrote {write_typst_document(ROOT, output, config.report.confirmed_color, config.report.review_color)}"
        )
    print(f"Detected speakers: {result.num_speakers}")
    print(format_dialogue(dialogue))
    print()
    print(format_quality_metrics(quality))
    print()
    print(format_runtime_metrics(runtime_data))


if __name__ == "__main__":
    main()
