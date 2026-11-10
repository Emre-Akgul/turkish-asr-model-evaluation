from __future__ import annotations

import threading
from io import BytesIO
from typing import Any

import numpy as np
import soundfile as sf

from turkish_asr_eval.engines.base import ASREngine

DEFAULT_SAMPLE_RATE = 16000

# The Turkish balbal models. The package resolves these Hugging Face repo ids to
# its registry entries, pinned to the weights its release was tested with.
BALBAL_MODELS: dict[str, str] = {
    "balbal-v2": "erayyapagci/balbal-v2",
    "balbal-v2-manas": "erayyapagci/balbal-v2-manas",
}

MODEL_ALIASES: dict[str, str] = {
    **BALBAL_MODELS,
    **{model_id: model_id for model_id in BALBAL_MODELS.values()},
}


def _import_balbal() -> Any:
    import balbal

    return balbal


class BalbalEngine(ASREngine):
    """CPU engine for the `balbal` package (erayyap/balbal).

    balbal runs its ONNX graph and KenLM beam decoder on CPU and writes spoken
    numbers as digits (its default inverse text normalization).
    """

    def load(self) -> None:
        device = self.options.get("device")
        if device not in (None, "cpu"):
            raise ValueError("The balbal engine runs on CPU only")

        try:
            balbal = _import_balbal()
        except ImportError as exc:
            raise RuntimeError(
                "The balbal engine requires the balbal package (Python 3.11 or "
                "3.12). Install it with: pip install balbal==0.6.3"
            ) from exc

        self._model: Any = balbal.Model(self._resolve_model_id(self.model))
        # A balbal model must not be called from two threads at once.
        self._lock = threading.Lock()
        # Warm the ONNX session so the first scored row is not slowed down.
        self._model.transcribe(
            np.zeros(DEFAULT_SAMPLE_RATE, dtype=np.float32),
            sample_rate=DEFAULT_SAMPLE_RATE,
        )

    def transcribe(self, audio: Any) -> str:
        array, sample_rate = self._normalize_audio(audio)
        with self._lock:
            result = self._model.transcribe(array, sample_rate=sample_rate)
        return str(result["text"]).strip()

    def _resolve_model_id(self, model: str) -> str:
        try:
            return MODEL_ALIASES[model]
        except KeyError as exc:
            valid = ", ".join(BALBAL_MODELS.values())
            raise RuntimeError(
                f"Unsupported balbal model '{model}'. Supported models: {valid}"
            ) from exc

    def _normalize_audio(self, audio: Any) -> tuple[np.ndarray, int]:
        if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
            return self._prepare_array(audio["array"]), int(audio["sampling_rate"])
        if isinstance(audio, bytes):
            array, sampling_rate = sf.read(BytesIO(audio), dtype="float32")
            return self._prepare_array(array), int(sampling_rate)
        if isinstance(audio, str):
            array, sampling_rate = sf.read(audio, dtype="float32")
            return self._prepare_array(array), int(sampling_rate)
        raise TypeError(f"Unsupported audio input: {type(audio).__name__}")

    def _prepare_array(self, array: Any) -> np.ndarray:
        audio = np.asarray(array, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio
