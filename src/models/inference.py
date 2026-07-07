import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from tabulate import tabulate


@dataclass
class InferenceResults:
    audio_path: Path
    ground_truth: str
    predicted: str


@dataclass
class Metrics:
    wer: float
    cer: float
    wil: float
    wip: float
    per_item: Optional[List[Dict]] = None

    def pretty_print(
        self,
        show_per_item: bool = False,
        tablefmt: str = "grid",
        max_text_len: int = 40,
        show_full_text: bool = False,
    ) -> str:
        lines = ["Average Metrics:"]
        lines.append(f"  WER: {self.wer:.2%}")
        lines.append(f"  CER: {self.cer:.2%}")
        lines.append(f"  WIL: {self.wil:.2%}")
        lines.append(f"  WIP: {self.wip:.2%}")

        if show_per_item and self.per_item:
            lines.append("\n Per-Item Metrics:")

            def truncate(text):
                return text[:max_text_len] + "…" if len(text) > max_text_len else text

            table_data = []
            for idx, item in enumerate(self.per_item):
                table_data.append(
                    [
                        str(item["audio_path"]),
                        truncate(item["ground_truth"]),
                        truncate(item["predicted"]),
                        f"{item['wer']:.2%}",
                        f"{item['cer']:.2%}",
                        f"{item['wil']:.2%}",
                        f"{item['wip']:.2%}",
                    ]
                )
            headers = [
                "#",
                "Audio",
                "Ground Truth",
                "Predicted",
                "WER",
                "CER",
                "WIL",
                "WIP",
            ]
            for i, row in enumerate(table_data):
                row.insert(0, i + 1)
            lines.append(tabulate(table_data, headers=headers, tablefmt=tablefmt))

            if show_full_text:
                lines.append("\nFull Texts:")
                for idx, item in enumerate(self.per_item):
                    lines.append(f"\n[{idx + 1}] Audio: {item['audio_path']}")
                    lines.append(f"    Ground Truth: {item['ground_truth']}")
                    lines.append(f"    Predicted:     {item['predicted']}")

        return "\n".join(lines)
