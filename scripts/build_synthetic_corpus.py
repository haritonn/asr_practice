"""Building synthetic benchmark for testing current model"""

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

DEV_SINGLE_TEMPLATES = (
    "На {location} установлен {product}.",
    "Проверьте связь с устройством {product}.",
    "Для этой линии нужен {product}.",
    "В журнале сработал {product}.",
    "Замените на объекте {product}.",
    "На плане указан {product}.",
)
TEST_SINGLE_TEMPLATES = (
    "Во время обхода проверили {product} в {location}.",
    "В журнал обслуживания внесли состояние {product}.",
    "Для контрольного запуска подготовили {product}.",
    "В помещении {location} осмотрели {product}.",
    "Инженер подтвердил работу устройства {product}.",
    "На схеме объекта отмечен {product}.",
)
DEV_MULTI_TEMPLATES = (
    "Проверьте {first} и {second} в {location}.",
    "На этаже подключены {first}, затем {second}.",
)
TEST_MULTI_TEMPLATES = (
    "В {location} последовательно осмотрели {first} и {second}.",
    "В журнале указали, что проверены {first}, а также {second}.",
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
DEV_HARD_NEGATIVE_TEMPLATES = (
    "Сигнал о плановом осмотре передали дежурному инженеру в начале смены.",
    "В третьем секторе рубежа проверили освещение, двери и свободный проход.",
    "На линии оставили два резервных канала для плановых работ.",
)
TEST_HARD_NEGATIVE_TEMPLATES = (
    "Дежурный принял сигнал о завершении обхода и сделал запись в журнале.",
    "На границе третьего участка осмотрели кабельный канал и защитные крышки.",
    "В отчёте указали двести пятьдесят метров кабеля, использованного при ремонте.",
)
LOCATIONS = ("серверной", "вестибюле", "архиве", "коридоре", "складе", "щитовой")
MULTI_PAIRS = {
    "dev": ((0, 8), (9, 11), (12, 21), (15, 20), (0, 1), (17, 18)),
    "test": ((0, 8), (9, 11), (12, 21), (15, 20), (1, 2), (12, 13)),
}
NOISE_SNRS_DB = (8.0, 12.0, 16.0)
REVERB_PROFILES = {
    "small_room": ((0.027, 0.25), (0.053, 0.14), (0.081, 0.08)),
    "large_room": ((0.041, 0.42), (0.089, 0.27), (0.147, 0.16)),
}


def _long_records_for_split(catalog: ProductCatalog, split: str) -> list[dict]:
    forms = lambda product: product.spoken_forms[0 if split == "dev" else -1]
    product_ids = (
        ("bolid.s2000-kdl-2i", "rubezh.r3-rubezh-kau2", "rubezh.r3-rubezh-biu")
        if split == "dev"
        else ("bolid.signal-20m", "rubezh.r3-rubezh-kau2", "rubezh.r3-rubezh-biu")
    )
    products = [catalog.products[product_id] for product_id in product_ids]
    positive_sentences = (
        "Перед началом плановой проверки инженер сверил список помещений и подготовил журнал работ.",
        "Осмотр решили проводить последовательно, чтобы не пропустить состояние линий и источников питания.",
        f"В серверной сначала проверили устройство {forms(products[0])} и записали показания индикаторов.",
        f"На следующем участке требовалось проверить {forms(products[1])}, установленный рядом с основным шкафом.",
        "Проверка заняла несколько минут, потому что инженер повторно измерил напряжение и осмотрел клеммы.",
        f"В журнале отдельно указали состояние {forms(products[2])} и время завершения контрольного запуска.",
        "Дополнительных тревожных событий во время осмотра не возникло, питание оставалось стабильным.",
        "Результаты проверки внесли в журнал, а повторный осмотр назначили на следующую рабочую смену.",
    )
    negative_a = (
        "Утром инженер получил задание провести обычный осмотр помещений на втором этаже.",
        "Сначала он проверил состояние дверей, освещение и свободный доступ к техническим шкафам.",
        "В серверной измерили температуру воздуха и убедились, что вентиляция работает без перебоев.",
        "В журнале за предыдущую смену не было записей о тревогах или нарушениях связи.",
        "Дежурный подтвердил, что питание в помещениях не отключалось и ремонтные работы не проводились.",
        "Особое внимание уделили маркировке линий и актуальности схемы на внутренней стороне шкафа.",
        "Все замечания записали в рабочий блокнот без указания конкретных моделей устройств.",
        "По итогам проверки неисправностей не обнаружили, поэтому срочные действия не потребовались.",
    )
    negative_b = (
        "Во второй половине дня сотрудник готовил общий отчёт о техническом обслуживании объекта.",
        "Он собрал журналы нескольких смен и сверил даты выполненных плановых проверок.",
        "Часть записей пришлось уточнить, поскольку в них отсутствовало время начала обхода.",
        "После уточнения в отчёт добавили сведения об осмотре помещений и проверке резервного питания.",
        "Названия конкретных приборов в разговоре не обсуждались и в документ не добавлялись.",
        "Отдельно отметили, что за отчётный период ложных тревог на объекте не зарегистрировано.",
        "Готовый документ передали руководителю для проверки и последующего согласования.",
        "Работу завершили без дополнительных заявок на ремонт или замену оборудования.",
    )

    def record(kind: str, sentences: tuple[str, ...], expected: list[str]) -> dict:
        return {
            "kind": kind,
            "template_group": kind,
            "expected_product_ids": expected,
            "text": " ".join(sentences),
            "sentences": sentences,
            "long_form": True,
        }

    return [
        record("long_positive", positive_sentences, list(product_ids)),
        record("long_negative", negative_a, []),
        record("long_negative", negative_b, []),
    ]


def _load_config() -> tuple[dict, Path]:
    root = Path(__file__).resolve().parents[1]
    with (root / "config" / "benchmark.json").open(encoding="utf-8") as source:
        return json.load(source), root


def _write_noise(audio: np.ndarray, sample_rate: int, path: Path, seed: int) -> None:
    rms = max(float(np.sqrt(np.mean(np.square(audio)))), 1e-6)
    snr_db = NOISE_SNRS_DB[seed % len(NOISE_SNRS_DB)]
    noise_rms = rms / (10 ** (snr_db / 20))
    noise = np.random.default_rng(seed).normal(0, noise_rms, size=audio.shape)
    sf.write(path, np.clip(audio + noise, -1.0, 1.0), sample_rate, subtype="PCM_16")


def _write_reverb(audio: np.ndarray, sample_rate: int, path: Path, seed: int) -> None:
    delayed = audio.astype(np.float64).copy()
    profile = tuple(REVERB_PROFILES.values())[seed % len(REVERB_PROFILES)]
    for delay_seconds, gain in profile:
        delay = int(delay_seconds * sample_rate)
        delayed[delay:] += gain * audio[:-delay]
    peak = max(float(np.max(np.abs(delayed))), 1.0)
    sf.write(
        path, (0.95 * delayed / peak).astype(np.float32), sample_rate, subtype="PCM_16"
    )


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
        _write_reverb(clean, sample_rate, path, seed)
    else:
        audio = AudioSegment.from_file(clean_path).set_channels(1)
        if condition == "fast":
            audio = speedup(audio, playback_speed=1.12)
        elif condition == "telephone":
            audio = audio.set_frame_rate(8_000).set_channels(1)
        else:
            raise ValueError(f"Unknown acoustic condition: {condition}")
        audio.export(path, format="wav")


def _acoustic_variant(condition: str, seed: int) -> str:
    if condition == "noise":
        return f"snr_{NOISE_SNRS_DB[seed % len(NOISE_SNRS_DB)]:g}db"
    if condition == "reverb":
        return tuple(REVERB_PROFILES)[seed % len(REVERB_PROFILES)]
    if condition == "fast":
        return "speed_1.12"
    if condition == "telephone":
        return "telephone_8khz"
    return "clean"


def _tts(model, text: str, speaker: str, sample_rate: int) -> np.ndarray:
    return (
        model.apply_tts(text=text, speaker=speaker, sample_rate=sample_rate)
        .cpu()
        .numpy()
    )


def _render_record(model, record: dict, speaker: str, sample_rate: int) -> np.ndarray:
    sentences = record.get("sentences")
    if not sentences:
        return _tts(model, record["text"], speaker, sample_rate)

    pieces = []
    for index, sentence in enumerate(sentences):
        pieces.append(_tts(model, sentence, speaker, sample_rate))
        if index + 1 < len(sentences):
            pause_seconds = 0.45 + 0.15 * (index % 3)
            pieces.append(np.zeros(int(sample_rate * pause_seconds), dtype=np.float32))
    return np.concatenate(pieces)


def _records_for_split(catalog: ProductCatalog, split: str) -> list[dict]:
    products = list(catalog.products.values())
    # Every benchmark phrase must be a pronunciation accepted by production.
    # The split is held out by speakers, templates and pronunciation variants,
    # not by an intentionally unavailable spelling variant.
    forms = lambda product: product.spoken_forms[0 if split == "dev" else -1]
    records = []
    template_offset = 0 if split == "dev" else 1
    single_templates = DEV_SINGLE_TEMPLATES if split == "dev" else TEST_SINGLE_TEMPLATES
    multi_templates = DEV_MULTI_TEMPLATES if split == "dev" else TEST_MULTI_TEMPLATES
    for index, product in enumerate(products):
        records.append(
            {
                "kind": "single",
                "template_group": "single",
                "expected_product_ids": [product.id],
                "text": single_templates[
                    (index + template_offset) % len(single_templates)
                ].format(
                    location=LOCATIONS[(index + template_offset) % len(LOCATIONS)],
                    product=forms(product),
                ),
            }
        )

    for first_index, second_index in MULTI_PAIRS[split]:
        first, second = products[first_index], products[second_index]
        records.append(
            {
                "kind": "multi",
                "template_group": "multi",
                "expected_product_ids": [first.id, second.id],
                "text": multi_templates[
                    (len(records) + template_offset) % len(multi_templates)
                ].format(
                    location=LOCATIONS[
                        (len(records) + template_offset) % len(LOCATIONS)
                    ],
                    first=forms(first),
                    second=forms(second),
                ),
            }
        )

    negative_templates = (
        NEGATIVE_TEMPLATES if split == "dev" else TEST_NEGATIVE_TEMPLATES
    )
    for index, template in enumerate(negative_templates):
        records.append(
            {
                "kind": "negative",
                "template_group": "negative",
                "expected_product_ids": [],
                "text": template.format(location=LOCATIONS[index]),
            }
        )
    hard_negative_templates = (
        DEV_HARD_NEGATIVE_TEMPLATES if split == "dev" else TEST_HARD_NEGATIVE_TEMPLATES
    )
    for template in hard_negative_templates:
        records.append(
            {
                "kind": "hard_negative",
                "template_group": "hard_negative",
                "expected_product_ids": [],
                "text": template,
            }
        )
    records.extend(_long_records_for_split(catalog, split))
    return records


def _write_split(
    model,
    catalog: ProductCatalog,
    output: Path,
    split: str,
    speakers: list[str],
    conditions: list[str],
    sample_rate: int,
    long_form_config: dict,
    corpus_version: str,
) -> list[dict]:
    records = []
    for template_index, template in enumerate(_records_for_split(catalog, split)):
        template_speakers = speakers
        template_conditions = conditions
        if template.get("long_form"):
            template_speakers = speakers[: long_form_config["speakers_per_split"]]
            template_conditions = long_form_config["conditions"]
        for speaker_index, speaker in enumerate(template_speakers):
            item_id = f"{split}-{template_index:03d}-{speaker}"
            clean_path = output / "audio" / f"{item_id}-source.wav"
            generated = [
                output / "audio" / f"{item_id}-{condition}.wav"
                for condition in template_conditions
            ]
            if not all(path.is_file() for path in generated):
                with torch.inference_mode():
                    clean = _render_record(model, template, speaker, sample_rate)
                sf.write(clean_path, clean, sample_rate, subtype="PCM_16")
                if template.get("long_form"):
                    duration = len(clean) / sample_rate
                    minimum = long_form_config["min_duration_seconds"]
                    maximum = long_form_config["max_duration_seconds"]
                    if not minimum <= duration <= maximum:
                        raise ValueError(
                            f"Long-form sample {item_id} is {duration:.1f}s; "
                            f"expected {minimum}-{maximum}s"
                        )
            for condition_index, condition in enumerate(template_conditions):
                audio_path = output / "audio" / f"{item_id}-{condition}.wav"
                seed = 1_009 * template_index + 101 * speaker_index + condition_index
                if not audio_path.is_file():
                    _write_condition(
                        clean,
                        sample_rate,
                        clean_path,
                        audio_path,
                        condition,
                        seed=seed,
                    )
                records.append(
                    {
                        "id": f"{item_id}-{condition}",
                        "split": split,
                        "kind": template["kind"],
                        "audio_path": str(audio_path),
                        "speaker": speaker,
                        "condition": condition,
                        "acoustic_variant": _acoustic_variant(condition, seed),
                        "tts_text": template["text"],
                        "expected_product_ids": template["expected_product_ids"],
                        "duration_seconds": round(sf.info(audio_path).duration, 3),
                        "long_form": bool(template.get("long_form")),
                        "template_group": template.get(
                            "template_group", template["kind"]
                        ),
                        "corpus_version": corpus_version,
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
        config["long_form"],
        config["corpus_version"],
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
            config["long_form"],
            config["corpus_version"],
        )
    )
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for record in records:
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {output}")


if __name__ == "__main__":
    main()
