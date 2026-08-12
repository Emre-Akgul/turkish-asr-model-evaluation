from turkish_asr_eval.engines.omnilingual import OMNILINGUAL_MODELS, OmnilingualEngine


class FakePipeline:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return ["merhaba"]


def test_omnilingual_ctc_does_not_pass_language():
    engine = OmnilingualEngine("omniASR_CTC_300M_v2")
    engine._uses_lang = False
    engine._pipeline = FakePipeline()

    prediction = engine.transcribe(b"audio")

    assert prediction == "merhaba"
    assert engine._pipeline.calls == [([b"audio"], {"batch_size": 1})]


def test_omnilingual_llm_passes_turkish_language():
    engine = OmnilingualEngine("omniASR_LLM_300M_v2")
    engine._uses_lang = True
    engine._pipeline = FakePipeline()

    prediction = engine.transcribe(b"audio")

    assert prediction == "merhaba"
    assert engine._pipeline.calls == [
        ([b"audio"], {"batch_size": 1, "lang": ["tur_Latn"]})
    ]


def test_omnilingual_resolves_official_checkpoint_urls():
    engine = OmnilingualEngine("model")

    assert (
        engine._resolve_model_name(
            "https://dl.fbaipublicfiles.com/mms/omniASR-CTC-300M-v2.pt"
        )
        == "omniASR_CTC_300M_v2"
    )
    assert (
        engine._resolve_model_name(
            "https://dl.fbaipublicfiles.com/mms/omniASR-LLM-300M-v2.pt"
        )
        == "omniASR_LLM_300M_v2"
    )


def test_all_requested_omnilingual_models_are_supported():
    assert set(OMNILINGUAL_MODELS) == {
        "omniASR_CTC_300M_v2",
        "omniASR_CTC_1B_v2",
        "omniASR_CTC_3B_v2",
        "omniASR_CTC_7B_v2",
        "omniASR_LLM_300M_v2",
        "omniASR_LLM_1B_v2",
        "omniASR_LLM_3B_v2",
        "omniASR_LLM_7B_v2",
    }
