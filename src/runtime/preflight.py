import os
from dataclasses import dataclass

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from pydub import AudioSegment

from src.audio.normalization import CANONICAL_CHANNELS, CANONICAL_SAMPLE_RATE

PYANNOTE_MODEL_ID = "pyannote/speaker-diarization-community-1"


@dataclass(frozen=True, slots=True)
class PreflightReport:
    passed: bool
    plan: dict
    errors: list[str]
    warnings: list[str]

    def to_dict(self):
        return {
            "passed": self.passed,
            "plan": self.plan,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def run_preflight(
    audio_path,
    *,
    output_path,
    asr_device,
    vad_device,
    terminology_device,
    diarization_device,
    terminology_catalog,
    terminology_model,
    pyannote_model_id=PYANNOTE_MODEL_ID,
):
    """Validate local inputs and describe the planned resource allocation."""
    errors = []
    warnings = []
    audio = {}
    if not audio_path.is_file():
        errors.append(f"Audio file is missing: {audio_path}")
    else:
        try:
            decoded_audio = AudioSegment.from_file(audio_path)
            audio = {
                "path": str(audio_path.resolve()),
                "duration_seconds": round(len(decoded_audio) / 1_000, 3),
                "sample_rate": decoded_audio.frame_rate,
                "channels": decoded_audio.channels,
                "format": audio_path.suffix.lstrip(".").upper() or "unknown",
            }
            if not decoded_audio:
                errors.append(f"Audio file is empty: {audio_path}")
        except Exception as error:
            errors.append(f"Audio cannot be decoded: {error}")

    for name, path in (
        ("Terminology catalog", terminology_catalog),
        ("Terminology model", terminology_model),
    ):
        if not path.is_file():
            errors.append(f"{name} is missing: {path}")

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        free_memory, _ = torch.cuda.mem_get_info(0)
        cuda = {
            "available": True,
            "name": properties.name,
            "total_memory_mib": round(properties.total_memory / 1024**2),
            "free_memory_mib": round(free_memory / 1024**2),
        }
    else:
        cuda = {
            "available": False,
            "name": None,
            "total_memory_mib": None,
            "free_memory_mib": None,
        }

    cuda_stages = [
        name
        for name, device in (
            ("vad", vad_device),
            ("asr", asr_device),
            ("terminology", terminology_device),
            ("diarization", diarization_device),
        )
        if device == "cuda"
    ]
    if cuda_stages and not cuda_available:
        errors.append(
            "CUDA was requested for "
            f"{', '.join(sorted(cuda_stages))}, but no CUDA device is available."
        )

    token_present = bool(os.environ.get("HF_TOKEN"))
    pyannote_cached = False
    if not token_present:
        errors.append("HF_TOKEN is missing; pyannote Community-1 requires it.")
    else:
        try:
            snapshot_download(repo_id=pyannote_model_id, local_files_only=True)
            pyannote_cached = True
        except LocalEntryNotFoundError:
            warnings.append(
                "The primary pyannote model is not fully cached; the first run requires "
                "access to Hugging Face."
            )

    if not output_path.parent.exists():
        warnings.append(f"Output directory will be created: {output_path.parent}")

    plan = {
        "audio": audio,
        "canonical_audio": {
            "sample_rate": CANONICAL_SAMPLE_RATE,
            "channels": CANONICAL_CHANNELS,
            "encoding": "PCM_16 WAV",
        },
        "devices": {
            "vad": vad_device,
            "asr": asr_device,
            "diarization": diarization_device,
            "terminology": terminology_device,
        },
        "strategy": "sequential; every model is unloaded before the next stage",
        "cuda": cuda,
        "pyannote": {
            "model_id": pyannote_model_id,
            "token_present": token_present,
            "primary_model_cached": pyannote_cached,
        },
    }
    return PreflightReport(
        passed=not errors,
        plan=plan,
        errors=errors,
        warnings=warnings,
    )


def format_preflight(report):
    """Render a concise human-readable preflight report."""
    status = "OK" if report.passed else "FAILED"
    lines = [f"Preflight: {status}"]
    audio = report.plan["audio"]
    if audio:
        lines.append(
            "Audio: "
            f"{audio['duration_seconds']:.1f}s, {audio['channels']} channel(s), "
            f"{audio['sample_rate']} Hz"
        )
    canonical_audio = report.plan["canonical_audio"]
    lines.append(
        "Canonical audio: "
        f"{canonical_audio['channels']} channel(s), "
        f"{canonical_audio['sample_rate']} Hz, {canonical_audio['encoding']}"
    )
    lines.append(
        "Devices: "
        + ", ".join(
            f"{stage}={device}" for stage, device in report.plan["devices"].items()
        )
    )
    cuda = report.plan["cuda"]
    if cuda["available"]:
        lines.append(
            f"GPU: {cuda['name']}, {cuda['free_memory_mib']}/"
            f"{cuda['total_memory_mib']} MiB free"
        )
    else:
        lines.append("GPU: unavailable")
    lines.append(f"Strategy: {report.plan['strategy']}")
    lines.extend(f"WARNING: {warning}" for warning in report.warnings)
    lines.extend(f"ERROR: {error}" for error in report.errors)
    return "\n".join(lines)
