from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from io import BytesIO
from typing import Any

import numpy as np
import soundfile as sf

from turkish_asr_eval.engines.base import ASREngine

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
BASE_URL_ENV = "NEMOTRON_TRANSCRIBE_BASE_URL"
# The model card's exact prompt for its FLEURS ASR results ("fleurs.asr").
TRANSCRIBE_PROMPT = (
    "Audio transcription. Language: as spoken, in its standard writing system. "
    "Wording: verbatim, including fillers and repetitions. Formatting: punctuation "
    "and capitalization; numbers in their standard written form. Output: transcript only."
)
MAX_TOKENS = 4096
REQUEST_TIMEOUT_SECONDS = 600

NEMOTRON_TRANSCRIBE_MODELS: dict[str, str] = {
    "NVIDIA-Nemotron-3.5-Transcribe-30B-A4B": (
        "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
    ),
    "NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8": (
        "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8"
    ),
    "NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4": (
        "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"
    ),
}

MODEL_ALIASES: dict[str, str] = {
    **NEMOTRON_TRANSCRIBE_MODELS,
    **{model_id: model_id for model_id in NEMOTRON_TRANSCRIBE_MODELS.values()},
}


class NemotronTranscribeVLLMEngine(ASREngine):
    """Client for a Nemotron 3.5 Transcribe model served by `vllm serve`.

    The model needs vLLM plus the NeMo Speech vLLM plugin, which lives in its
    own Python 3.13 environment, so the server is started separately with
    `scripts/serve_nemotron_transcribe.sh` and this engine talks to its
    OpenAI-compatible chat completions endpoint. Speculative decoding (MTP or
    the DFlash2 draft model) is a server-side setting; requests are identical.
    """

    def load(self) -> None:
        self._model_id = self._resolve_model_id(self.model)
        self._base_url = str(
            self.options.get("base_url")
            or os.environ.get(BASE_URL_ENV)
            or DEFAULT_BASE_URL
        ).rstrip("/")
        served = self._served_models()
        if self._model_id not in served:
            raise RuntimeError(
                f"vLLM server at {self._base_url} serves {sorted(served)}, "
                f"not '{self._model_id}'. Start it with "
                "scripts/serve_nemotron_transcribe.sh."
            )

    def transcribe(self, audio: Any) -> str:
        encoded = base64.b64encode(self._to_wav_bytes(audio)).decode("ascii")
        payload = {
            "model": self._model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": TRANSCRIBE_PROMPT},
                        {
                            "type": "audio_url",
                            "audio_url": {"url": f"data:audio/wav;base64,{encoded}"},
                        },
                    ],
                }
            ],
            "temperature": 0,
            "top_p": 1.0,
            "seed": 0,
            "max_tokens": MAX_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False},
            "skip_special_tokens": False,
        }
        response = self._request("/chat/completions", payload)
        content = response["choices"][0]["message"].get("content") or ""
        return str(content).strip()

    def _resolve_model_id(self, model: str) -> str:
        try:
            return MODEL_ALIASES[model]
        except KeyError as exc:
            valid = ", ".join(NEMOTRON_TRANSCRIBE_MODELS)
            raise RuntimeError(
                f"Unsupported Nemotron Transcribe model '{model}'. "
                f"Supported models: {valid}"
            ) from exc

    def _served_models(self) -> set[str]:
        response = self._request("/models", None)
        return {str(entry["id"]) for entry in response.get("data", [])}

    def _request(self, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer EMPTY",
            },
        )
        try:
            with urllib.request.urlopen(
                request, timeout=REQUEST_TIMEOUT_SECONDS
            ) as handle:
                return json.loads(handle.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM request to {url} failed: {exc.code} {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach vLLM server at {url}: {exc.reason}. Start it "
                f"with scripts/serve_nemotron_transcribe.sh or set {BASE_URL_ENV}."
            ) from exc

    def _to_wav_bytes(self, audio: Any) -> bytes:
        if isinstance(audio, bytes):
            array, sampling_rate = sf.read(BytesIO(audio), dtype="float32")
        elif isinstance(audio, str):
            array, sampling_rate = sf.read(audio, dtype="float32")
        elif isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
            array, sampling_rate = audio["array"], audio["sampling_rate"]
        else:
            raise TypeError(f"Unsupported audio input: {type(audio).__name__}")

        mono = np.asarray(array, dtype=np.float32)
        if mono.ndim > 1:
            mono = mono.mean(axis=1)
        buffer = BytesIO()
        sf.write(buffer, mono, int(sampling_rate), format="WAV", subtype="FLOAT")
        return buffer.getvalue()
