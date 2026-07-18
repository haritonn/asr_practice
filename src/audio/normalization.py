from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
from pydub import AudioSegment


CANONICAL_SAMPLE_RATE = 16_000
CANONICAL_CHANNELS = 1
CANONICAL_SAMPLE_WIDTH_BYTES = 2


@dataclass(frozen=True, slots=True)
class CanonicalAudio:
    path: Path
    duration_seconds: float
    sample_rate: int
    channels: int
    subtype: str

    def to_dict(self):
        return {
            "path": str(self.path),
            "duration_seconds": round(self.duration_seconds, 3),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "subtype": self.subtype,
        }


def normalize_audio(source_path, destination_path):
    """Convert input audio to mono 16 kHz signed-16-bit PCM WAV."""
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    canonical = (
        AudioSegment.from_file(source_path)
        .set_channels(CANONICAL_CHANNELS)
        .set_frame_rate(CANONICAL_SAMPLE_RATE)
        .set_sample_width(CANONICAL_SAMPLE_WIDTH_BYTES)
    )
    canonical.export(destination_path, format="wav")
    info = sf.info(destination_path)
    if (
        info.samplerate != CANONICAL_SAMPLE_RATE
        or info.channels != CANONICAL_CHANNELS
        or info.subtype != "PCM_16"
    ):
        raise RuntimeError(
            "Audio normalization produced an unexpected format: "
            f"{info.channels} channel(s), {info.samplerate} Hz, {info.subtype}"
        )
    return CanonicalAudio(
        path=destination_path,
        duration_seconds=float(info.duration),
        sample_rate=info.samplerate,
        channels=info.channels,
        subtype=info.subtype,
    )
