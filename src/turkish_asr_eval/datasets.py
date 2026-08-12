from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, cast
import unicodedata

import jiwer
from datasets import Audio, load_dataset
from whisper_normalizer.basic import BasicTextNormalizer


WHISPER_NORMALIZER = BasicTextNormalizer()
ASR_CONTROL_TOKEN = re.compile(r"<[^<>\s]+>")


@dataclass(frozen=True)
class DatasetDefinition:
    hf_name: str
    config: str
    audio_column: str
    reference_column: str
    trust_remote_code: bool = False


SUPPORTED_DATASETS: dict[str, DatasetDefinition] = {
    "fleurs": DatasetDefinition(
        hf_name="google/fleurs",
        config="tr_tr",
        audio_column="audio",
        reference_column="transcription",
    ),
    "common_voice": DatasetDefinition(
        hf_name="fixie-ai/common_voice_17_0",
        config="tr",
        audio_column="audio",
        reference_column="sentence",
    ),
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_name: str
    config: str
    split: str
    audio_column: str
    reference_column: str
    trust_remote_code: bool = False


def parse_dataset_spec(value: str) -> DatasetSpec:
    parts = value.split(":")
    if len(parts) != 2 or any(part.strip() == "" for part in parts):
        raise ValueError(
            "dataset must be in the form '<dataset>:<split>'. Supported datasets: "
            f"{', '.join(sorted(SUPPORTED_DATASETS))}"
        )
    name, split = (part.strip() for part in parts)
    try:
        definition = SUPPORTED_DATASETS[name]
    except KeyError as exc:
        raise ValueError(
            f"unsupported dataset '{name}'. Supported datasets: "
            f"{', '.join(sorted(SUPPORTED_DATASETS))}"
        ) from exc
    return DatasetSpec(
        name=name,
        hf_name=definition.hf_name,
        config=definition.config,
        split=split,
        audio_column=definition.audio_column,
        reference_column=definition.reference_column,
        trust_remote_code=definition.trust_remote_code,
    )


def load_hf_dataset(spec: DatasetSpec, *, streaming: bool = True) -> Any:
    dataset = load_dataset(
        spec.hf_name,
        spec.config,
        split=spec.split,
        streaming=streaming,
        trust_remote_code=spec.trust_remote_code,
    )
    return dataset.cast_column(spec.audio_column, Audio(decode=False))


def extract_audio_input(value: Any) -> Any:
    if isinstance(value, dict):
        if "array" in value and "sampling_rate" in value:
            return {
                "array": value["array"],
                "sampling_rate": value["sampling_rate"],
            }
        if "bytes" in value and value["bytes"]:
            return value["bytes"]
        if "path" in value and value["path"]:
            return value["path"]
    return value


def clean_transcript_artifacts(text: str) -> str:
    """Remove non-spoken ASR tokens and canonicalize Turkish dotted i."""
    text = unicodedata.normalize("NFC", text)
    text = ASR_CONTROL_TOKEN.sub(" ", text)
    # Some datasets/models store Turkish i as LATIN SMALL LETTER I + COMBINING DOT.
    return text.replace("i\u0307", "i")


def normalize_text(text: str) -> str:
    """Normalize Turkish transcripts consistently before ASR scoring."""
    text = clean_transcript_artifacts(text)
    # Python's default lowercasing is not locale-aware for Turkish I/i.
    text = text.translate(str.maketrans({"I": "ı", "İ": "i"})).lower()
    text = "".join(
        character
        for character in text
        if not unicodedata.category(character).startswith(("P", "S"))
    )
    return re.sub(r"\s+", " ", text).strip()


def normalize_text_whisper(text: str) -> str:
    """Apply Whisper's multilingual basic text normalizer."""
    return WHISPER_NORMALIZER(clean_transcript_artifacts(text)).strip()


def compute_error_rates(reference: str, prediction: str) -> tuple[float, float]:
    reference = normalize_text(reference)
    prediction = normalize_text(prediction)
    return (
        jiwer.wer(reference, prediction) * 100,
        cast(float, jiwer.cer(reference, prediction)) * 100,
    )


def compute_whisper_error_rates(reference: str, prediction: str) -> tuple[float, float]:
    reference = normalize_text_whisper(reference)
    prediction = normalize_text_whisper(prediction)
    return (
        jiwer.wer(reference, prediction) * 100,
        cast(float, jiwer.cer(reference, prediction)) * 100,
    )


def _edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for row_index, left_item in enumerate(left, start=1):
        current = [row_index]
        for column_index, right_item in enumerate(right, start=1):
            cost = 0 if left_item == right_item else 1
            current.append(
                min(
                    previous[column_index] + 1,
                    current[column_index - 1] + 1,
                    previous[column_index - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def _wer(reference: str, prediction: str) -> float:
    reference_words = reference.split()
    prediction_words = prediction.split()
    if not reference_words:
        return 0.0 if not prediction_words else 100.0
    distance = _edit_distance(reference_words, prediction_words)
    return distance / len(reference_words) * 100


def _cer(reference: str, prediction: str) -> float:
    reference_chars = list(reference)
    prediction_chars = list(prediction)
    if not reference_chars:
        return 0.0 if not prediction_chars else 100.0
    distance = _edit_distance(reference_chars, prediction_chars)
    return distance / len(reference_chars) * 100
