"""Run the end-to-end speaker-attributed speech-to-text pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.asr.terminology import CtcTerminologyRecognizer
from src.asr.whisper import WhisperAsr
from src.diarization.pyannote import PyannoteCommunityDiarizer
from src.evaluation.quality import quality_metrics
from src.models.configs import (
    PyannoteDiarizationConfig,
    SileroConfig,
    TerminologyConfig,
    WhisperConfig,
)
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
CATALOG_PATH = ROOT / "resources/fire_catalog.json"
TERMINOLOGY_MODEL_PATH = ROOT / "models/stt_ru_conformer_ctc_large.nemo"
MIN_CUDA_MEMORY_BYTES = 8 * 1024**3


def _default_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    total_memory = torch.cuda.get_device_properties(0).total_memory
    return "cuda" if total_memory >= MIN_CUDA_MEMORY_BYTES else "cpu"


def _build_pipeline(device: str, num_speakers: int | None) -> DiarizedSpeechPipeline:
    asr = WhisperAsr(
        WhisperConfig(
            model_size_or_path="medium",
            device=device,
            compute_type="int8" if device == "cpu" else None,
            beam_size=1,
            language="ru",
            word_timestamps=True,
        )
    )
    terminology = CtcTerminologyRecognizer(
        TerminologyConfig(
            catalog_path=CATALOG_PATH,
            model_path=TERMINOLOGY_MODEL_PATH,
            device=device,
        )
    )
    diarizer = PyannoteCommunityDiarizer(
        PyannoteDiarizationConfig(
            # Community-1 does not fit into the project's 4 GB GPU.
            device="cpu",
            num_speakers=num_speakers,
            output="exclusive",
        )
    )
    return DiarizedSpeechPipeline(
        asr=asr,
        vad=SileroVoiceDetection(SileroConfig(device=device)),
        diarizer=diarizer,
        terminology=terminology,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Путь к аудиофайлу")
    parser.add_argument("--output", type=Path, help="Куда записать JSON-результат")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Устройство для распознавания (по умолчанию: auto)",
    )
    parser.add_argument(
        "--num-speakers", type=int, help="Точное число спикеров, если оно известно"
    )
    parser.add_argument(
        "--typst", action="store_true", help="Создать report.typ в корне проекта"
    )
    args = parser.parse_args()

    device = _default_device() if args.device == "auto" else args.device
    output = args.output or args.audio.with_suffix(".diarized.json")
    preflight = run_preflight(
        args.audio,
        output_path=output,
        asr_device=device,
        terminology_device=device,
        diarization_device="cpu",
        terminology_catalog=CATALOG_PATH,
        terminology_model=TERMINOLOGY_MODEL_PATH,
    )
    print(format_preflight(preflight))
    if not preflight.passed:
        raise SystemExit("Preflight failed; no models were loaded.")

    output.parent.mkdir(parents=True, exist_ok=True)
    pipeline = _build_pipeline(device, args.num_speakers)
    result = pipeline.transcribe(args.audio)
    runtime = pipeline.last_runtime
    if runtime is None:
        raise RuntimeError("Pipeline completed without runtime metrics.")

    dialogue = dialogue_rows(result)
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
        print(f"Wrote {write_typst_document(ROOT, output)}")
    print(f"Detected speakers: {result.num_speakers}")
    print(format_dialogue(dialogue))
    print()
    print(format_quality_metrics(quality))
    print()
    print(format_runtime_metrics(runtime_data))


if __name__ == "__main__":
    main()
