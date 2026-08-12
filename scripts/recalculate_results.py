#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

import jiwer

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(REPO_ROOT / "src"))

from turkish_asr_eval.datasets import (
    compute_error_rates,
    compute_whisper_error_rates,
    normalize_text,
    normalize_text_whisper,
)


def recalculate(path: Path) -> None:
    summary = {
        "rows": 0,
        "scored_rows": 0,
        "error_rows": 0,
        "word_errors": 0,
        "reference_words": 0,
        "char_errors": 0,
        "reference_chars": 0,
        "whisper_word_errors": 0,
        "whisper_reference_words": 0,
        "whisper_char_errors": 0,
        "whisper_reference_chars": 0,
    }
    rewritten: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        # Recover sparse/NUL padding left by an interrupted file transfer.
        line = line.lstrip("\0")
        if not line.strip():
            continue
        record = json.loads(line)
        summary["rows"] += 1
        if "error" in record:
            summary["error_rows"] += 1
        if "reference" in record and "prediction" in record and "error" not in record:
            record["wer"], record["cer"] = compute_error_rates(
                record["reference"], record["prediction"]
            )
            record["whisper_wer"], record["whisper_cer"] = (
                compute_whisper_error_rates(record["reference"], record["prediction"])
            )
            reference = normalize_text(record["reference"])
            prediction = normalize_text(record["prediction"])
            words = jiwer.process_words(reference, prediction)
            chars = jiwer.process_characters(reference, prediction)
            summary["scored_rows"] += 1
            summary["word_errors"] += words.substitutions + words.deletions + words.insertions
            summary["reference_words"] += words.hits + words.substitutions + words.deletions
            summary["char_errors"] += chars.substitutions + chars.deletions + chars.insertions
            summary["reference_chars"] += chars.hits + chars.substitutions + chars.deletions
            whisper_reference = normalize_text_whisper(record["reference"])
            whisper_prediction = normalize_text_whisper(record["prediction"])
            whisper_words = jiwer.process_words(whisper_reference, whisper_prediction)
            whisper_chars = jiwer.process_characters(whisper_reference, whisper_prediction)
            summary["whisper_word_errors"] += (
                whisper_words.substitutions
                + whisper_words.deletions
                + whisper_words.insertions
            )
            summary["whisper_reference_words"] += (
                whisper_words.hits
                + whisper_words.substitutions
                + whisper_words.deletions
            )
            summary["whisper_char_errors"] += (
                whisper_chars.substitutions
                + whisper_chars.deletions
                + whisper_chars.insertions
            )
            summary["whisper_reference_chars"] += (
                whisper_chars.hits
                + whisper_chars.substitutions
                + whisper_chars.deletions
            )
        rewritten.append(json.dumps(record, ensure_ascii=False))

    write_atomically(path, "\n".join(rewritten) + "\n")
    summary_path = path.with_suffix(".summary.json")
    output = {
        "rows": summary["rows"],
        "scored_rows": summary["scored_rows"],
        "error_rows": summary["error_rows"],
        "mean_wer": summary["word_errors"] / summary["reference_words"] * 100,
        "mean_cer": summary["char_errors"] / summary["reference_chars"] * 100,
        "mean_whisper_wer": summary["whisper_word_errors"]
        / summary["whisper_reference_words"]
        * 100,
        "mean_whisper_cer": summary["whisper_char_errors"]
        / summary["whisper_reference_chars"]
        * 100,
    }
    write_atomically(summary_path, json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(
        f"{path.name}: WER={output['mean_wer']:.2f}% CER={output['mean_cer']:.2f}% "
        f"Whisper-WER={output['mean_whisper_wer']:.2f}% "
        f"Whisper-CER={output['mean_whisper_cer']:.2f}%"
    )


def write_atomically(path: Path, content: str) -> None:
    existing_mode = path.stat().st_mode & 0o777 if path.exists() else 0o664
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary_path = Path(handle.name)
    temporary_path.chmod(existing_mode)
    try:
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    for path in sorted(RESULTS_DIR.glob("*.jsonl")):
        recalculate(path)


if __name__ == "__main__":
    main()
