from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import soundfile as sf

from turkish_asr_eval.engines.base import ASREngine

TURKISH_LANGUAGE = "Turkish"

QWEN3_ASR_MODELS: dict[str, str] = {
    "Qwen3-ASR-0.6B": "Qwen/Qwen3-ASR-0.6B",
    "Qwen3-ASR-1.7B": "Qwen/Qwen3-ASR-1.7B",
}

MODEL_ALIASES: dict[str, str] = {
    **QWEN3_ASR_MODELS,
    **{model_id: model_id for model_id in QWEN3_ASR_MODELS.values()},
}


class Qwen3ASREngine(ASREngine):
    def load(self) -> None:
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise RuntimeError(
                "The qwen3_asr engine requires the qwen-asr package. Install "
                "it with: pip install qwen-asr==0.0.6"
            ) from exc

        model_id = self._resolve_model_id(self.model)
        device = str(
            self.options.get("device")
            or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        if device == "cuda":
            device = "cuda:0"

        compute_type = self.options.get("compute_type")
        if compute_type is not None:
            try:
                dtype = getattr(torch, str(compute_type))
            except AttributeError as exc:
                raise ValueError(
                    f"Unsupported Qwen3-ASR compute type: {compute_type}"
                ) from exc
        elif device.startswith("cuda"):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        self._model: Any = Qwen3ASRModel.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device,
            max_inference_batch_size=1,
            max_new_tokens=256,
        )

    def transcribe(self, audio: Any) -> str:
        results = self._model.transcribe(
            audio=self._normalize_audio(audio),
            language=TURKISH_LANGUAGE,
        )
        if not results:
            return ""
        first = results[0]
        if hasattr(first, "text"):
            return str(first.text).strip()
        return str(first).strip()

    def _resolve_model_id(self, model: str) -> str:
        try:
            return MODEL_ALIASES[model]
        except KeyError as exc:
            valid = ", ".join(QWEN3_ASR_MODELS)
            raise RuntimeError(
                f"Unsupported Qwen3-ASR model '{model}'. Supported models: {valid}"
            ) from exc

    def _normalize_audio(self, audio: Any) -> Any:
        if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
            return self._prepare_array(audio["array"]), int(audio["sampling_rate"])
        if isinstance(audio, bytes):
            array, sampling_rate = sf.read(BytesIO(audio), dtype="float32")
            return self._prepare_array(array), int(sampling_rate)
        return audio

    def _prepare_array(self, array: Any) -> np.ndarray:
        audio = np.asarray(array, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio
