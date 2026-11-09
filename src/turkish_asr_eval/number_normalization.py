"""Spoken-form normalization of numbers for the TN (text normalization) metrics.

References and predictions disagree on whether numbers are written as digits or
words ("yüzde 20'si" vs "yüzde yirmisi"), so the primary metrics penalize a
correct transcription for its formatting. The TN metrics first convert both
sides to their spoken form with the Turkish grammar of NeMo text processing
(``nemo_text_processing``, lang="tr"), then apply the primary normalization.

Pipeline for each text:

    clean_transcript_artifacts -> Turkish lowercase -> Turkish TN grammar -> normalize_text

Lowercasing comes first because the grammar reads upper case runs as acronyms
("ABD" -> "a be de") but leaves lower case ones alone, while FLEURS references
are lower case and model predictions are cased. Lowercasing both sides keeps the
grammar from treating them differently.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, cast

import jiwer

from turkish_asr_eval.datasets import (
    clean_transcript_artifacts,
    normalize_text,
    turkish_lower,
)

CACHE_DIR_ENV = "TURKISH_ASR_EVAL_TN_CACHE"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "turkish_asr_eval" / "nemo_tn"
INSTALL_HINT = (
    "The TN metrics need NeMo text processing with the Turkish grammar. Install "
    "it into this environment with: pip install -e /path/to/NeMo-text-processing"
)

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_normalize_spoken: Callable[[str], str] | None = None
_grammar_hash: str | None = None


class NumberNormalizationUnavailable(RuntimeError):
    pass


def _grammar_dir() -> Path:
    try:
        import nemo_text_processing.text_normalization.tr as tr_grammar
    except ImportError as exc:
        raise NumberNormalizationUnavailable(INSTALL_HINT) from exc
    return Path(tr_grammar.__file__).parent


def grammar_hash() -> str:
    """Short hash of the Turkish grammar's source and data files."""
    global _grammar_hash
    if _grammar_hash is None:
        digest = hashlib.sha256()
        root = _grammar_dir()
        for path in sorted(root.rglob("*")):
            if path.suffix in {".py", ".tsv"}:
                digest.update(str(path.relative_to(root)).encode())
                digest.update(path.read_bytes())
        _grammar_hash = digest.hexdigest()[:12]
    return _grammar_hash


def _build_normalizer() -> Callable[[str], str]:
    try:
        from nemo_text_processing.text_normalization.normalize import Normalizer
    except ImportError as exc:
        raise NumberNormalizationUnavailable(INSTALL_HINT) from exc

    # Normalizer.normalize() resets this logger to INFO on every call and logs each
    # punctuation post-processing step; a filter outlives those resets.
    logging.getLogger("NeMo-text-processing").addFilter(
        lambda record: record.levelno >= logging.ERROR
    )

    # NeMo names its compiled grammar cache by language only, so a cache built
    # from an older grammar would be reused silently. Keying the directory by the
    # grammar's content hash rebuilds it whenever the grammar changes.
    cache_dir = Path(os.environ.get(CACHE_DIR_ENV) or DEFAULT_CACHE_DIR) / grammar_hash()
    cache_dir.mkdir(parents=True, exist_ok=True)
    normalizer = Normalizer(input_case="cased", lang="tr", cache_dir=str(cache_dir))

    def normalize(text: str) -> str:
        return normalizer.normalize(text, punct_post_process=True)

    return normalize


def ensure_available() -> None:
    """Build the grammar now, so a missing install fails before a long run."""
    global _normalize_spoken
    with _lock:
        if _normalize_spoken is None:
            _normalize_spoken = _build_normalizer()


@lru_cache(maxsize=65536)
def _spoken(text: str) -> str:
    ensure_available()
    normalize = cast(Callable[[str], str], _normalize_spoken)
    try:
        # The grammar is not documented as thread safe; --workers runs rows in threads.
        with _lock:
            return normalize(text)
    except Exception as exc:  # one unreadable sentence must not drop the row
        logger.warning("TN grammar failed, scoring the text without it: %r (%s)", text, exc)
        return text


def normalize_text_tn(text: str) -> str:
    """Normalize a transcript with numbers in spoken form, for the TN metrics."""
    return normalize_text(_spoken(turkish_lower(clean_transcript_artifacts(text))))


def compute_tn_error_rates(reference: str, prediction: str) -> tuple[float, float]:
    reference = normalize_text_tn(reference)
    prediction = normalize_text_tn(prediction)
    return (
        jiwer.wer(reference, prediction) * 100,
        cast(float, jiwer.cer(reference, prediction)) * 100,
    )


def tn_metadata() -> dict[str, Any]:
    """Identifies the grammar behind the TN metrics, for the summary file."""
    return {"tn_normalizer": "nemo_text_processing/tr", "tn_grammar_hash": grammar_hash()}
