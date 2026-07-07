import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import List

from src.metrics import compute_metrics
from src.models.inference import InferenceResults
from src.pipeline import inference


def display_metrics(data: List[InferenceResults]):
    """Computing & pretty-printing metrics"""
    metrics = compute_metrics(data)
    print(
        "\n"
        + metrics.pretty_print(show_per_item=True, show_full_text=True, max_text_len=50)
    )


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
            display_metrics(inference_data)

        except Exception as e:
            logger.exception(f"Error occured during reading: {e}")
            return

    else:
        logger.info("Making inference on validation data")
        results = inference()
        logger.info("Computing metrics...")
        display_metrics(results)

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
