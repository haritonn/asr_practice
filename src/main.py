import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from src.asr.whisper import WhisperAsr
from src.metrics import compute_metrics
from src.models.configs import SileroConfig, WhisperConfig
from src.models.inference import InferenceResults, Metrics
from src.pipeline import inference
from src.vad.silero_vad import SileroVoiceDetection


def main(args):
    level = logging.INFO if args.logger else logging.ERROR
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logging.getLogger("faster_whisper").setLevel(level)
    logger = logging.getLogger(__name__)

    if args.validation_path is not None:
        path = args.validation_path
        logger.info(f"Trying to read from {path}")
        inference_data = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    inference_data.append(
                        InferenceResults(
                            audio_path=Path(obj["audio_path"]),
                            ground_truth=obj["ground_truth"],
                            predicted=obj["predicted"],
                        )
                    )

            logger.info("Done. Computing metrics...")

            metrics_dict = compute_metrics(inference_data)

            per_item = []
            for i, res in enumerate(inference_data):
                per_item.append(
                    {
                        "audio_path": res.audio_path,
                        "ground_truth": res.ground_truth,
                        "predicted": res.predicted,
                        "wer": metrics_dict["wer"]["sample_scores"][i],
                        "cer": metrics_dict["cer"]["sample_scores"][i],
                        "wil": metrics_dict["wil"]["sample_scores"][i],
                        "wip": metrics_dict["wip"]["sample_scores"][i],
                    }
                )

            metrics = Metrics(
                wer=metrics_dict["wer"]["average"],
                cer=metrics_dict["cer"]["average"],
                wil=metrics_dict["wil"]["average"],
                wip=metrics_dict["wip"]["average"],
                per_item=per_item,
            )

            print(
                "\n"
                + metrics.pretty_print(
                    show_per_item=True, show_full_text=True, max_text_len=50
                )
            )
        except Exception as e:
            logger.exception(f"Error occured during reading: {e}")
            return

    else:
        logger.info("Making inference on validation data")
        results = inference()
        logger.info("Computing metrics...")

        metrics_dict = compute_metrics(results)

        per_item = []
        for i, res in enumerate(results):
            per_item.append(
                {
                    "audio_path": res.audio_path,
                    "ground_truth": res.ground_truth,
                    "predicted": res.predicted,
                    "wer": metrics_dict["wer"]["sample_scores"][i],
                    "cer": metrics_dict["cer"]["sample_scores"][i],
                    "wil": metrics_dict["wil"]["sample_scores"][i],
                    "wip": metrics_dict["wip"]["sample_scores"][i],
                }
            )

        metrics = Metrics(
            wer=metrics_dict["wer"]["average"],
            cer=metrics_dict["cer"]["average"],
            wil=metrics_dict["wil"]["average"],
            wip=metrics_dict["wip"]["average"],
            per_item=per_item,
        )

        print(
            "\n"
            + metrics.pretty_print(
                show_per_item=True, show_full_text=True, max_text_len=50
            )
        )
        if (results_path := args.results_path) is not None:
            logger.info(f"Writing into {results_path}")
            with open(results_path, "w", encoding="utf-8") as f:
                for result in results:
                    record = asdict(result)
                    record["audio_path"] = str(record["audio_path"])  # Path -> str
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(f"Wrote {len(results)} into {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation_path", type=str, required=False)
    parser.add_argument("--results_path", type=str, required=False)
    parser.add_argument("--logger", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    main(args)
    print("\nSuccess!")
