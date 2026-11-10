from types import SimpleNamespace

import numpy as np
import pytest

from turkish_asr_eval.engines import balbal
from turkish_asr_eval.engines.balbal import BALBAL_MODELS, BalbalEngine


class FakeModel:
    def __init__(self, model):
        self.model = model
        self.calls = []

    def transcribe(self, audio, sample_rate=None):
        self.calls.append((audio, sample_rate))
        return {"text": " Merhaba dünya. ", "words": []}


@pytest.fixture
def fake_balbal(monkeypatch):
    loaded = []

    def make_model(model):
        loaded.append(FakeModel(model))
        return loaded[-1]

    monkeypatch.setattr(balbal, "_import_balbal", lambda: SimpleNamespace(Model=make_model))
    return loaded


def test_both_turkish_models_are_supported():
    assert BALBAL_MODELS == {
        "balbal-v2": "erayyapagci/balbal-v2",
        "balbal-v2-manas": "erayyapagci/balbal-v2-manas",
    }


@pytest.mark.parametrize("model", ["erayyapagci/balbal-v2-manas", "balbal-v2-manas"])
def test_load_passes_the_repo_id_and_warms_up(fake_balbal, model):
    engine = BalbalEngine(model)
    engine.load()

    (loaded,) = fake_balbal
    assert loaded.model == "erayyapagci/balbal-v2-manas"
    warm_audio, warm_rate = loaded.calls[0]
    assert warm_rate == 16000
    np.testing.assert_array_equal(warm_audio, np.zeros(16000, dtype=np.float32))


def test_transcribe_passes_mono_float32_and_sample_rate(fake_balbal):
    engine = BalbalEngine("erayyapagci/balbal-v2")
    engine.load()

    prediction = engine.transcribe({"array": [[0.1, 0.3], [0.2, 0.4]], "sampling_rate": 22050})

    assert prediction == "Merhaba dünya."
    array, sample_rate = fake_balbal[0].calls[-1]
    np.testing.assert_allclose(array, [0.2, 0.3])
    assert array.dtype == np.float32
    assert sample_rate == 22050


def test_unknown_model_has_helpful_error(fake_balbal):
    with pytest.raises(RuntimeError, match="Supported models"):
        BalbalEngine("erayyapagci/balbal-v2e-zellige").load()


def test_rejects_gpu_devices(fake_balbal):
    with pytest.raises(ValueError, match="CPU only"):
        BalbalEngine("erayyapagci/balbal-v2", device="cuda").load()
