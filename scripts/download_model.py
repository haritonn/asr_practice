"""Download the fixed NeMo checkpoint used by the application configuration."""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path


MODEL_URL = (
    "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/"
    "stt_ru_conformer_ctc_large/versions/1.13.0/files/stt_ru_conformer_ctc_large.nemo"
)


def main() -> None:
    destination = Path("models/stt_ru_conformer_ctc_large.nemo")
    if destination.is_file():
        print(f"Already present: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".nemo.part")
    print(f"Downloading {destination.name} (about 464 MB)...")
    try:
        with urllib.request.urlopen(MODEL_URL) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
