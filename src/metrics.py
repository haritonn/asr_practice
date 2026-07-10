"""Text-transcription metrics for an explicitly labelled evaluation set."""

from __future__ import annotations

from dataclasses import dataclass

import jiwer
from tabulate import tabulate

from .models.inference import InferenceResults


@dataclass(frozen=True, slots=True)
class TextMetrics:
    wer: float
    cer: float
    wil: float
    wip: float

    def pretty_print(self) -> str:
        return tabulate(
            [[f"{self.wer:.2%}", f"{self.cer:.2%}", f"{self.wil:.2%}", f"{self.wip:.2%}"]],
            headers=["WER", "CER", "WIL", "WIP"],
            tablefmt="rounded_outline",
        )


def normalize_text(text: str) -> str:
    return jiwer.Compose(
        [jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()]
    )(text)


def compute_metrics(records: list[InferenceResults]) -> TextMetrics:
    if not records:
        raise ValueError("At least one labelled inference record is required")
    references = [normalize_text(record.ground_truth) for record in records]
    predictions = [normalize_text(record.predicted) for record in records]
    return TextMetrics(
        wer=jiwer.wer(references, predictions),
        cer=jiwer.cer(references, predictions),
        wil=jiwer.wil(references, predictions),
        wip=jiwer.wip(references, predictions),
    )
