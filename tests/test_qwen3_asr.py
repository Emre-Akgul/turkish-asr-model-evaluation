from types import SimpleNamespace

import numpy as np
import pytest

from turkish_asr_eval.engines import qwen3_asr
from turkish_asr_eval.engines.qwen3_asr import (
    QWEN3_ASR_MODELS,
    Qwen3ASREngine,
    Qwen3ASRVLLMEngine,
)


class FakeQwenModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, **kwargs):
        self.calls.append(kwargs)
        return [SimpleNamespace(text=" merhaba dünya ")]


def test_all_qwen3_asr_models_are_supported():
    assert QWEN3_ASR_MODELS == {
        "Qwen3-ASR-0.6B": "Qwen/Qwen3-ASR-0.6B",
        "Qwen3-ASR-1.7B": "Qwen/Qwen3-ASR-1.7B",
    }


@pytest.mark.parametrize("model_id", QWEN3_ASR_MODELS.values())
def test_official_model_ids_are_supported(model_id):
    engine = Qwen3ASREngine(model_id)
    assert engine._resolve_model_id(model_id) == model_id


def test_qwen3_asr_forces_turkish_and_passes_audio_tuple():
    engine = Qwen3ASREngine("Qwen3-ASR-0.6B")
    engine._model = FakeQwenModel()
    audio = {"array": [[0.1, 0.3], [0.2, 0.4]], "sampling_rate": 16000}

    prediction = engine.transcribe(audio)

    assert prediction == "merhaba dünya"
    call = engine._model.calls[0]
    assert call["language"] == "Turkish"
    array, sampling_rate = call["audio"]
    np.testing.assert_allclose(array, [0.2, 0.3])
    assert array.dtype == np.float32
    assert sampling_rate == 16000


def test_unknown_qwen3_asr_model_has_helpful_error():
    engine = Qwen3ASREngine("missing")
    with pytest.raises(RuntimeError, match="Supported models"):
        engine._resolve_model_id(engine.model)


def test_qwen3_asr_vllm_loads_official_backend(monkeypatch):
    captured = {}

    class FakeQwen3ASRModel:
        @staticmethod
        def LLM(**kwargs):
            captured.update(kwargs)
            return FakeQwenModel()

    monkeypatch.setattr(
        qwen3_asr,
        "_import_qwen3_asr_model",
        lambda: FakeQwen3ASRModel,
    )

    engine = Qwen3ASRVLLMEngine("Qwen3-ASR-0.6B", device="cuda")
    engine.load()

    assert captured == {
        "model": "Qwen/Qwen3-ASR-0.6B",
        "gpu_memory_utilization": 0.8,
        "max_inference_batch_size": 1,
        "max_new_tokens": 256,
    }


def test_qwen3_asr_vllm_rejects_cpu():
    engine = Qwen3ASRVLLMEngine("Qwen3-ASR-0.6B", device="cpu")
    with pytest.raises(ValueError, match="requires a CUDA device"):
        engine.load()
