from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import soundfile as sf

from turkish_asr_eval.engines.base import ASREngine

DEFAULT_SAMPLE_RATE = 16000


def _import_mihu_stt() -> Any:
    import mihu_stt

    return mihu_stt


class MihuSTTEngine(ASREngine):
    """CPU-only engine for the `turkish-stt` package (mihuai/turkish-stt).

    The `mihu_stt` module resolves a single bundled model and downloads its
    weights from Hugging Face on first use, so `self.model` is accepted for
    CLI/naming consistency but is not passed through to the library.
    """

    def load(self) -> None:
        self._mihu_stt = _import_mihu_stt()
        self._quantized = bool(self.options.get("quantized", True))
        # Trigger the one-time weight download and warm the library's
        # internal recognizer cache so the first scored sample isn't slow.
        self._mihu_stt.transcribe(
            np.zeros(DEFAULT_SAMPLE_RATE, dtype=np.float32),
            sample_rate=DEFAULT_SAMPLE_RATE,
            quantized=self._quantized,
        )

    def transcribe(self, audio: Any) -> str:
        array, sample_rate = self._normalize_audio(audio)
        prediction = self._mihu_stt.transcribe(
            array,
            sample_rate=sample_rate,
            quantized=self._quantized,
        )
        return str(prediction).strip()

    def _normalize_audio(self, audio: Any) -> tuple[np.ndarray, int]:
        if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
            return self._prepare_array(audio["array"]), int(audio["sampling_rate"])
        if isinstance(audio, bytes):
            array, sampling_rate = sf.read(BytesIO(audio), dtype="float32")
            return self._prepare_array(array), int(sampling_rate)
        if isinstance(audio, str):
            array, sampling_rate = sf.read(audio, dtype="float32")
            return self._prepare_array(array), int(sampling_rate)
        return audio, DEFAULT_SAMPLE_RATE

    def _prepare_array(self, array: Any) -> np.ndarray:
        audio = np.asarray(array, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio
