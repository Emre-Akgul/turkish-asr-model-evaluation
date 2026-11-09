import base64
from io import BytesIO

import numpy as np
import pytest
import soundfile as sf

from turkish_asr_eval.engines.nemotron_transcribe import (
    NEMOTRON_TRANSCRIBE_MODELS,
    NemotronTranscribeVLLMEngine,
)

BF16 = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"


class FakeServer:
    def __init__(self, served=(BF16,)):
        self.served = served
        self.calls = []

    def __call__(self, path, payload):
        self.calls.append((path, payload))
        if path == "/models":
            return {"data": [{"id": model_id} for model_id in self.served]}
        return {"choices": [{"message": {"content": " merhaba dünya "}}]}


def make_engine(monkeypatch, server, model=BF16, **options):
    engine = NemotronTranscribeVLLMEngine(model, **options)
    monkeypatch.setattr(engine, "_request", server)
    return engine


def test_all_variants_are_supported():
    assert set(NEMOTRON_TRANSCRIBE_MODELS.values()) == {
        BF16,
        f"{BF16}-FP8",
        f"{BF16}-NVFP4",
    }


def test_load_rejects_server_serving_another_model(monkeypatch):
    engine = make_engine(monkeypatch, FakeServer(served=(f"{BF16}-FP8",)))
    with pytest.raises(RuntimeError, match="not 'nvidia/NVIDIA-Nemotron"):
        engine.load()


def test_unknown_model_has_helpful_error(monkeypatch):
    engine = make_engine(monkeypatch, FakeServer(), model="missing")
    with pytest.raises(RuntimeError, match="Supported models"):
        engine.load()


def test_base_url_option_wins_over_env(monkeypatch):
    monkeypatch.setenv("NEMOTRON_TRANSCRIBE_BASE_URL", "http://env:1/v1")
    engine = make_engine(monkeypatch, FakeServer(), base_url="http://opt:2/v1/")
    engine.load()
    assert engine._base_url == "http://opt:2/v1"


def test_transcribe_sends_card_request(monkeypatch):
    server = FakeServer()
    engine = make_engine(monkeypatch, server, model="NVIDIA-Nemotron-3.5-Transcribe-30B-A4B")
    engine.load()

    prediction = engine.transcribe(
        {"array": [[0.1, 0.3], [0.2, 0.4]], "sampling_rate": 16000}
    )

    assert prediction == "merhaba dünya"
    path, payload = server.calls[-1]
    assert path == "/chat/completions"
    assert payload["model"] == BF16
    assert payload["temperature"] == 0
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    text, audio = payload["messages"][0]["content"]
    assert text == {"type": "text", "text": "Transcribe the following audio."}
    prefix = "data:audio/wav;base64,"
    url = audio["audio_url"]["url"]
    assert url.startswith(prefix)
    array, sampling_rate = sf.read(
        BytesIO(base64.b64decode(url[len(prefix):])), dtype="float32"
    )
    assert sampling_rate == 16000
    np.testing.assert_allclose(array, [0.2, 0.3])
