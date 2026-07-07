from pathlib import Path
from typing import List

from src.asr.whisper import WhisperAsr
from src.models.configs import SileroConfig, WhisperConfig
from src.models.inference import InferenceResults
from src.vad.silero_vad import SileroVoiceDetection

# Mock-values, should be removed in future
GD = "Какие крупнейшие сражения ознаменовали коренной перелом в ходе Великой Отечественной Войны?"
SAMPLE = Path("src/sample.opus")


def inference() -> List[InferenceResults]:
    """Inference on validation dataset. Config parameters hard-coded for now, will be fixed in future."""
    whisper_config = WhisperConfig(
        model_size_or_path="medium",
        device="cpu",
        beam_size=1,
        language="ru",
        word_timestamps=False,
    )
    silero_config = SileroConfig(
        model_path=Path("src/silero_vad_v6.2.1.onnx"),  # not needed for now
        window_size=512,
        device="cpu",
        sample_rate=16000,
        speech_threshold=0.7,
        silence_threshold=0.35,
        min_speech_duration_ms=100,
        min_silence_duration_ms=500,
    )

    try:
        asr = WhisperAsr(whisper_config)
        vad = SileroVoiceDetection(silero_config)
    except Exception as e:
        raise RuntimeError("Failed to load models") from e

    speech_segments = vad.detect(SAMPLE)
    result = asr.transcribe(SAMPLE, speech_segments)

    return [InferenceResults(audio_path=SAMPLE, ground_truth=GD, predicted=result.text)]
