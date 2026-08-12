from __future__ import annotations

from importlib import import_module
from typing import Type

from turkish_asr_eval.engines.base import ASREngine


ENGINE_REGISTRY: dict[str, tuple[str, str]] = {
    "faster_whisper": (
        "turkish_asr_eval.engines.faster_whisper",
        "FasterWhisperEngine",
    ),
    "omnilingual": ("turkish_asr_eval.engines.omnilingual", "OmnilingualEngine"),
    "nemo": ("turkish_asr_eval.engines.nemo", "NemoEngine"),
    "qwen3_asr": ("turkish_asr_eval.engines.qwen3_asr", "Qwen3ASREngine"),
}


class UnknownEngineError(ValueError):
    pass


def available_engines() -> tuple[str, ...]:
    return tuple(sorted(ENGINE_REGISTRY))


def get_engine_class(engine: str) -> Type[ASREngine]:
    try:
        module_name, class_name = ENGINE_REGISTRY[engine]
    except KeyError as exc:
        valid = ", ".join(available_engines())
        raise UnknownEngineError(
            f"Unknown engine '{engine}'. Valid engines: {valid}"
        ) from exc
    module = import_module(module_name)
    return getattr(module, class_name)


def create_engine(engine: str, model: str, **options: object) -> ASREngine:
    return get_engine_class(engine)(model, **options)
