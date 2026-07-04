import logging
from pathlib import Path

from src.asr.whisper import WhisperAsr
from src.metrics import compute_metrics
from src.models.configs import WhisperConfig

logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    whisper_config = WhisperConfig(
        model_size_or_path="medium",
        device="cuda",
        beam_size=3,
        language="ru",
        word_timestamps=True,
    )
    try:
        logger.info("Loading model")
        model = WhisperAsr(whisper_config)
    except Exception as e:
        logger.error(f"Error while loading model: {e}")
        return

    logger.info("Ok")
    sample = Path("src/sample.opus")
    logger.info("Beginnig transcribe")
    result = model.transcribe(sample)
    gd = "Какие крупнейшие сражения ознаменовали коренной перелом в ходе Великой Отечественной Войны?"

    logger.info("Computing metrics")
    metrics = compute_metrics([gd], [result.text])

    print("Prediction: ", result.text)
    print("Metrics: ", metrics)


if __name__ == "__main__":
    main()
