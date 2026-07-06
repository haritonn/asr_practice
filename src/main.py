import logging
from pathlib import Path

from src.asr.whisper import WhisperAsr
from src.metrics import compute_metrics
from src.models.configs import SileroConfig, WhisperConfig
from src.vad.silero_vad import SileroVoiceDetection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    whisper_config = WhisperConfig(
        model_size_or_path="medium",
        device="cpu",
        beam_size=1,
        language="ru",
        word_timestamps=False,
    )
    silero_config = SileroConfig(
        model_path=Path("src/silero_vad_v6.2.1.onnx"),
        window_size=512,
        device="cpu",
        sample_rate=16000,
        speech_threshold=0.7,
        silence_threshold=0.35,
        min_speech_duration_ms=100,
        min_silence_duration_ms=500,
    )
    try:
        logger.info("Loading model")
        asr = WhisperAsr(whisper_config)
        logger.info("Loading ONNX for silero-vad")
        vad = SileroVoiceDetection(silero_config)
    except Exception as e:
        logger.error(f"Error while loading models: {e}")
        return

    logger.info("Ok")
    sample = Path("src/sample.wav")
    logger.info("Beginnig transcribe & voice detection")

    speech_segments = vad.detect(sample)
    logger.info(f"Found {len(speech_segments)} speech segments")
    for i, seg in enumerate(speech_segments):
        logger.debug(f"Segment {i}: {seg.start:.2f}s - {seg.end:.2f}s")

    result = asr.transcribe(sample, speech_segments)
    gd = "Какие крупнейшие сражения ознаменовали коренной перелом в ходе Великой Отечественной Войны?"

    logger.info("Computing metrics")
    metrics = compute_metrics([gd], [result.text])

    print("Prediction: ", result.text)
    print("GD: ", gd)
    print("Metrics: ", metrics)


if __name__ == "__main__":
    main()
