"""Build a speaker-disjoint synthetic regression corpus from benchmark.json.

The corpus is intentionally generated outside version control. Its manifest
contains all provenance needed to reproduce a result from the fixed catalog.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pydub import AudioSegment
from pydub.effects import speedup

from src.catalog import Product, ProductCatalog


SINGLE_TEMPLATES = (
    "На {location} установлен {product}.",
    "Проверьте связь с устройством {product}.",
    "Для этой линии нужен {product}.",
    "В журнале сработал {product}.",
    "Замените на объекте {product}.",
    "На плане указан {product}.",
)
MULTI_TEMPLATES = (
    "Проверьте {first} и {second} в {location}.",
    "На этаже подключены {first}, затем {second}.",
)
NEGATIVE_TEMPLATES = (
    "Проверьте линию связи в {location}.",
    "На объекте требуется плановая проверка.",
    "Передайте журнал событий дежурному инженеру.",
    "Откройте схему пожарной сигнализации.",
    "В помещении завершены пусконаладочные работы.",
)
TEST_NEGATIVE_TEMPLATES = (
    "Дежурный инженер проверяет журнал в {location}.",
    "На объекте завершите обход пожарных зон.",
    "Подготовьте отчёт о техническом обслуживании.",
    "Согласуйте время плановых работ на этаже.",
    "В помещении не зафиксировано тревожных событий.",
)
LOCATIONS = ("серверной", "вестибюле", "архиве", "коридоре", "складе", "щитовой")


def _load_config() -> tuple[dict, Path]:
    root = Path(__file__).resolve().parents[1]
    with (root / "config" / "benchmark.json").open(encoding="utf-8") as source:
        return json.load(source), root


def _write_noise(audio: np.ndarray, sample_rate: int, path: Path, seed: int) -> None:
    rms = max(float(np.sqrt(np.mean(np.square(audio)))), 1e-6)
    noise_rms = rms / (10 ** (12 / 20))
    noise = np.random.default_rng(seed).normal(0, noise_rms, size=audio.shape)
    sf.write(path, np.clip(audio + noise, -1.0, 1.0), sample_rate, subtype="PCM_16")


def _write_reverb(audio: np.ndarray, sample_rate: int, path: Path) -> None:
    delayed = audio.astype(np.float64).copy()
    for delay_seconds, gain in ((0.035, 0.35), (0.071, 0.20), (0.113, 0.12)):
        delay = int(delay_seconds * sample_rate)
        delayed[delay:] += gain * audio[:-delay]
    peak = max(float(np.max(np.abs(delayed))), 1.0)
    sf.write(path, (0.95 * delayed / peak).astype(np.float32), sample_rate, subtype="PCM_16")


def _write_condition(
    clean: np.ndarray,
    sample_rate: int,
    clean_path: Path,
    path: Path,
    condition: str,
    seed: int,
) -> None:
    if condition == "clean":
        shutil.copyfile(clean_path, path)
    elif condition == "noise":
        _write_noise(clean, sample_rate, path, seed)
    elif condition == "reverb":
        _write_reverb(clean, sample_rate, path)
    else:
        audio = AudioSegment.from_file(clean_path).set_channels(1)
        if condition == "fast":
            audio = speedup(audio, playback_speed=1.12)
        elif condition == "telephone":
            audio = audio.set_frame_rate(8_000).set_channels(1)
        else:
            raise ValueError(f"Unknown acoustic condition: {condition}")
        audio.export(path, format="wav")


def _tts(model, text: str, speaker: str, sample_rate: int) -> np.ndarray:
    return model.apply_tts(text=text, speaker=speaker, sample_rate=sample_rate).cpu().numpy()


def _records_for_split(catalog: ProductCatalog, split: str) -> list[dict]:
    products = list(catalog.products.values())
    # Every benchmark phrase must be a pronunciation accepted by production.
    # The split is held out by speakers, templates and acoustic conditions, not
    # by an intentionally unavailable spelling variant.
    forms = lambda product: product.spoken_forms[0 if split == "dev" else -1]
    records: list[dict] = []
    template_offset = 0 if split == "dev" else 1
    for index, product in enumerate(products):
        records.append(
            {
                "kind": "single",
                "expected_product_ids": [product.id],
                "text": SINGLE_TEMPLATES[(index + template_offset) % len(SINGLE_TEMPLATES)].format(
                    location=LOCATIONS[(index + template_offset) % len(LOCATIONS)],
                    product=forms(product),
                ),
            }
        )

    for first_index, second_index in ((0, 8), (9, 11), (12, 21), (15, 20)):
        first, second = products[first_index], products[second_index]
        records.append(
            {
                "kind": "multi",
                "expected_product_ids": [first.id, second.id],
                "text": MULTI_TEMPLATES[(len(records) + template_offset) % len(MULTI_TEMPLATES)].format(
                    location=LOCATIONS[(len(records) + template_offset) % len(LOCATIONS)],
                    first=forms(first),
                    second=forms(second),
                ),
            }
        )

    negative_templates = NEGATIVE_TEMPLATES if split == "dev" else TEST_NEGATIVE_TEMPLATES
    for index, template in enumerate(negative_templates):
        records.append(
            {
                "kind": "negative",
                "expected_product_ids": [],
                "text": template.format(location=LOCATIONS[index]),
            }
        )
    return records


def _write_split(
    model,
    catalog: ProductCatalog,
    output: Path,
    split: str,
    speakers: list[str],
    conditions: list[str],
    sample_rate: int,
) -> list[dict]:
    records = []
    for template_index, template in enumerate(_records_for_split(catalog, split)):
        for speaker_index, speaker in enumerate(speakers):
            item_id = f"{split}-{template_index:03d}-{speaker}"
            clean_path = output / "audio" / f"{item_id}-source.wav"
            generated = [
                output / "audio" / f"{item_id}-{condition}.wav" for condition in conditions
            ]
            if not all(path.is_file() for path in generated):
                with torch.inference_mode():
                    clean = _tts(model, template["text"], speaker, sample_rate)
                sf.write(clean_path, clean, sample_rate, subtype="PCM_16")
            for condition_index, condition in enumerate(conditions):
                audio_path = output / "audio" / f"{item_id}-{condition}.wav"
                if not audio_path.is_file():
                    _write_condition(
                        clean,
                        sample_rate,
                        clean_path,
                        audio_path,
                        condition,
                        seed=10_000 * template_index + 100 * speaker_index + condition_index,
                    )
                records.append(
                    {
                        "id": f"{item_id}-{condition}",
                        "split": split,
                        "kind": template["kind"],
                        "audio_path": str(audio_path),
                        "speaker": speaker,
                        "condition": condition,
                        "tts_text": template["text"],
                        "expected_product_ids": template["expected_product_ids"],
                        "catalog_version": catalog.version,
                    }
                )
    return records


def main() -> None:
    config, root = _load_config()
    catalog = ProductCatalog.load(root / config["catalog_path"])
    output = root / config["output_dir"]
    (output / "audio").mkdir(parents=True, exist_ok=True)
    print("Loading Silero Russian TTS model...")
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="ru",
        speaker="v4_ru",
    )
    records = _write_split(
        model,
        catalog,
        output,
        "dev",
        config["dev_speakers"],
        config["conditions"],
        config["sample_rate"],
    )
    records.extend(
        _write_split(
            model,
            catalog,
            output,
            "test",
            config["test_speakers"],
            config["conditions"],
            config["sample_rate"],
        )
    )
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for record in records:
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {output}")


if __name__ == "__main__":
    main()
