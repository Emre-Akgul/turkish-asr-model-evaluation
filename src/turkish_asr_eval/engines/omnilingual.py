from __future__ import annotations

from typing import Any, TypedDict

import torch
from fairseq2.assets import AssetCard
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

from turkish_asr_eval.engines.base import ASREngine

TURKISH_LANG = "tur_Latn"


class OmnilingualModelMetadata(TypedDict):
    card: str
    checkpoint: str
    model_family: str
    model_arch: str
    uses_lang: bool


OMNILINGUAL_MODELS: dict[str, OmnilingualModelMetadata] = {
    "omniASR_CTC_300M_v2": {
        "card": "omniASR_CTC_300M_v2",
        "checkpoint": "https://dl.fbaipublicfiles.com/mms/omniASR-CTC-300M-v2.pt",
        "model_family": "wav2vec2_asr",
        "model_arch": "300m_v2",
        "uses_lang": False,
    },
    "omniASR_CTC_1B_v2": {
        "card": "omniASR_CTC_1B_v2",
        "checkpoint": "https://dl.fbaipublicfiles.com/mms/omniASR-CTC-1B-v2.pt",
        "model_family": "wav2vec2_asr",
        "model_arch": "1b_v2",
        "uses_lang": False,
    },
    "omniASR_CTC_3B_v2": {
        "card": "omniASR_CTC_3B_v2",
        "checkpoint": "https://dl.fbaipublicfiles.com/mms/omniASR-CTC-3B-v2.pt",
        "model_family": "wav2vec2_asr",
        "model_arch": "3b_v2",
        "uses_lang": False,
    },
    "omniASR_CTC_7B_v2": {
        "card": "omniASR_CTC_7B_v2",
        "checkpoint": "https://dl.fbaipublicfiles.com/mms/omniASR-CTC-7B-v2.pt",
        "model_family": "wav2vec2_asr",
        "model_arch": "7b_v2",
        "uses_lang": False,
    },
    "omniASR_LLM_300M_v2": {
        "card": "omniASR_LLM_300M_v2",
        "checkpoint": "https://dl.fbaipublicfiles.com/mms/omniASR-LLM-300M-v2.pt",
        "model_family": "wav2vec2_llama",
        "model_arch": "300m_v2",
        "uses_lang": True,
    },
    "omniASR_LLM_1B_v2": {
        "card": "omniASR_LLM_1B_v2",
        "checkpoint": "https://dl.fbaipublicfiles.com/mms/omniASR-LLM-1B-v2.pt",
        "model_family": "wav2vec2_llama",
        "model_arch": "1b_v2",
        "uses_lang": True,
    },
    "omniASR_LLM_3B_v2": {
        "card": "omniASR_LLM_3B_v2",
        "checkpoint": "https://dl.fbaipublicfiles.com/mms/omniASR-LLM-3B-v2.pt",
        "model_family": "wav2vec2_llama",
        "model_arch": "3b_v2",
        "uses_lang": True,
    },
    "omniASR_LLM_7B_v2": {
        "card": "omniASR_LLM_7B_v2",
        "checkpoint": "https://dl.fbaipublicfiles.com/mms/omniASR-LLM-7B-v2.pt",
        "model_family": "wav2vec2_llama",
        "model_arch": "7b_v2",
        "uses_lang": True,
    },
}

MODEL_ALIASES: dict[str, str] = {
    model_name: model_name for model_name in OMNILINGUAL_MODELS
}
MODEL_ALIASES.update(
    {
        metadata["checkpoint"]: model_name
        for model_name, metadata in OMNILINGUAL_MODELS.items()
    }
)


class OmnilingualEngine(ASREngine):
    def load(self) -> None:
        model_name = self._resolve_model_name(self.model)
        metadata = OMNILINGUAL_MODELS[model_name]
        model_card: Any = metadata["card"]

        if self.model not in MODEL_ALIASES:
            model_card = AssetCard(
                model_name,
                {
                    "model_family": metadata["model_family"],
                    "model_arch": metadata["model_arch"],
                    "checkpoint": self.model,
                },
            )

        device = str(self.options.get("device") or self._default_device(torch))
        dtype = self._default_dtype(torch, device)
        self._uses_lang = bool(metadata["uses_lang"])
        pipeline_model_card: Any = model_card
        self._pipeline: Any = ASRInferencePipeline(
            model_card=pipeline_model_card,
            device=device,
            dtype=dtype,
        )

    def transcribe(self, audio: Any) -> str:
        kwargs: dict[str, Any] = {"batch_size": 1}
        if self._uses_lang:
            kwargs["lang"] = [TURKISH_LANG]

        result = self._pipeline.transcribe([audio], **kwargs)
        if isinstance(result, (list, tuple)) and result:
            return str(result[0]).strip()
        return str(result).strip()

    def _resolve_model_name(self, model: str) -> str:
        if model in MODEL_ALIASES:
            return MODEL_ALIASES[model]
        if model.endswith(".pt"):
            for model_name, metadata in OMNILINGUAL_MODELS.items():
                if model.endswith(metadata["checkpoint"].rsplit("/", maxsplit=1)[-1]):
                    return model_name
        valid = ", ".join(OMNILINGUAL_MODELS)
        raise RuntimeError(
            f"Unsupported omnilingual model '{model}'. Supported models: {valid}"
        )

    def _default_device(self, torch: Any) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _default_dtype(self, torch: Any, device: str) -> Any:
        if device == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
