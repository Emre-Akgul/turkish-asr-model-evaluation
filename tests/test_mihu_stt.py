import numpy as np

from turkish_asr_eval.engines import mihu_stt
from turkish_asr_eval.engines.mihu_stt import MihuSTTEngine


class FakeMihuSTT:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, sample_rate, quantized):
        self.calls.append((audio, sample_rate, quantized))
        return " merhaba dünya "


def test_mihu_stt_load_warms_cache_and_passes_quantized(monkeypatch):
    fake = FakeMihuSTT()
    monkeypatch.setattr(mihu_stt, "_import_mihu_stt", lambda: fake)

    engine = MihuSTTEngine("mihuai/turkish-stt", quantized=False)
    engine.load()

    assert len(fake.calls) == 1
    warm_audio, warm_rate, warm_quantized = fake.calls[0]
    assert warm_rate == 16000
    assert warm_quantized is False
    np.testing.assert_array_equal(warm_audio, np.zeros(16000, dtype=np.float32))


def test_mihu_stt_transcribe_passes_array_and_sample_rate():
    engine = MihuSTTEngine("mihuai/turkish-stt")
    engine._mihu_stt = FakeMihuSTT()
    engine._quantized = True
    audio = {"array": [[0.1, 0.3], [0.2, 0.4]], "sampling_rate": 22050}

    prediction = engine.transcribe(audio)

    assert prediction == "merhaba dünya"
    array, sample_rate, quantized = engine._mihu_stt.calls[-1]
    np.testing.assert_allclose(array, [0.2, 0.3])
    assert array.dtype == np.float32
    assert sample_rate == 22050
    assert quantized is True
