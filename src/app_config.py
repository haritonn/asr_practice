"""Loading and validation of the YAML runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.models.configs import (
    PyannoteDiarizationConfig,
    SileroConfig,
    TerminologyConfig,
    WhisperConfig,
)


@dataclass(frozen=True, slots=True)
class ReportConfig:
    confirmed_color: str
    review_color: str


@dataclass(frozen=True, slots=True)
class AppConfig:
    asr: WhisperConfig
    vad: SileroConfig
    diarization: PyannoteDiarizationConfig
    terminology: TerminologyConfig
    report: ReportConfig


def load_app_config(path: Path, project_root: Path) -> AppConfig:
    """Load a strict YAML configuration, resolving paths from project root."""
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file is missing: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("Configuration root must be a mapping.")
    _reject_unknown(loaded, {"asr", "vad", "diarization", "terminology", "report"}, "root")

    asr = _section(loaded, "asr")
    vad = _section(loaded, "vad")
    diarization = _section(loaded, "diarization")
    terminology = _section(loaded, "terminology")
    report = _section(loaded, "report")
    _reject_unknown(
        asr,
        {"model_size_or_path", "device", "compute_type", "beam_size", "language", "word_timestamps"},
        "asr",
    )
    _reject_unknown(
        vad,
        {"device", "sample_rate", "speech_threshold", "silence_threshold", "min_speech_duration_ms", "min_silence_duration_ms"},
        "vad",
    )
    _reject_unknown(
        diarization,
        {"model_id", "device", "num_speakers", "min_speakers", "max_speakers", "output"},
        "diarization",
    )
    _reject_unknown(
        terminology,
        {"catalog_path", "model_path", "device", "context_weight", "keyword_threshold", "ctc_alignment_weight", "confirmed_score_threshold"},
        "terminology",
    )
    _reject_unknown(report, {"confirmed_color", "review_color"}, "report")

    _validate_device(asr.get("device"), "asr.device")
    if vad.get("device") != "cpu":
        raise ValueError("vad.device must be cpu: the current Silero adapter is CPU-only.")
    _validate_device(diarization.get("device"), "diarization.device")
    _validate_device(terminology.get("device"), "terminology.device")
    for key in ("confirmed_color", "review_color"):
        value = report.get(key)
        if not isinstance(value, str) or not _is_hex_color(value):
            raise ValueError(f"report.{key} must be a #RRGGBB colour.")

    return AppConfig(
        asr=WhisperConfig(**asr),
        vad=SileroConfig(**vad),
        diarization=PyannoteDiarizationConfig(**diarization),
        terminology=TerminologyConfig(
            **{
                **terminology,
                "catalog_path": _project_path(project_root, terminology["catalog_path"]),
                "model_path": _project_path(project_root, terminology["model_path"]),
            }
        ),
        report=ReportConfig(**report),
    )


def _section(document: dict[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{name}' must be a mapping.")
    return value


def _reject_unknown(values: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown key(s) in {section}: {', '.join(unknown)}")


def _validate_device(value: Any, name: str) -> None:
    if value not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"{name} must be one of: auto, cpu, cuda.")


def _is_hex_color(value: str) -> bool:
    return len(value) == 7 and value.startswith("#") and all(
        character in "0123456789abcdefABCDEF" for character in value[1:]
    )


def _project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path
