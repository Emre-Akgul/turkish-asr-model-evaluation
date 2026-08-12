import pytest

from turkish_asr_eval.engines.qwen3_asr import Qwen3ASREngine, Qwen3ASRVLLMEngine
from turkish_asr_eval.engines.registry import (
    UnknownEngineError,
    available_engines,
    get_engine_class,
)


def test_all_engine_names_exist():
    assert available_engines() == (
        "faster_whisper",
        "nemo",
        "omnilingual",
        "qwen3_asr_transformers",
        "qwen3_asr_vllm",
    )


def test_unknown_engine_has_helpful_error():
    with pytest.raises(UnknownEngineError, match="Valid engines"):
        get_engine_class("missing")


def test_engine_class_is_loaded_lazily():
    assert get_engine_class("qwen3_asr_transformers") is Qwen3ASREngine
    assert get_engine_class("qwen3_asr_vllm") is Qwen3ASRVLLMEngine
