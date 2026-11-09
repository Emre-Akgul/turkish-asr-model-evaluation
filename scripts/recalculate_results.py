#!/usr/bin/env python3
"""Re-score result JSONL files in place from their stored references and predictions.

No model is run: every metric is recomputed from the `reference` and `prediction`
fields, and the matching `.summary.json` is rewritten. Pass --tn to also compute
the TN metrics (numbers in spoken form), which need NeMo text processing with the
Turkish grammar installed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(REPO_ROOT / "src"))

from turkish_asr_eval.cli import initial_summary, summary_output, update_summary
from turkish_asr_eval.datasets import compute_error_rates, compute_whisper_error_rates
from turkish_asr_eval.number_normalization import compute_tn_error_rates, ensure_available

TN_FIELDS = ("tn_wer", "tn_cer")


class StaleTNMetrics(RuntimeError):
    pass


def read_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        # Recover sparse/NUL padding left by an interrupted file transfer.
        line = line.lstrip("\0")
        if line.strip():
            records.append(json.loads(line))
    return records


def rescore(record: dict[str, Any], *, tn: bool, drop_tn: bool) -> None:
    if drop_tn:
        for field in TN_FIELDS:
            record.pop(field, None)
    if "reference" not in record or "prediction" not in record or "error" in record:
        return
    reference, prediction = record["reference"], record["prediction"]
    record["wer"], record["cer"] = compute_error_rates(reference, prediction)
    record["whisper_wer"], record["whisper_cer"] = compute_whisper_error_rates(
        reference, prediction
    )
    if tn:
        record["tn_wer"], record["tn_cer"] = compute_tn_error_rates(reference, prediction)


def recalculate(path: Path, *, tn: bool = False, drop_tn: bool = False) -> dict[str, Any]:
    records = read_records(path)
    if not tn and not drop_tn and any(field in r for r in records for field in TN_FIELDS):
        raise StaleTNMetrics(
            f"{path.name} has TN metrics: pass --tn to recompute them or --drop-tn "
            "to remove them"
        )

    summary = initial_summary()
    for record in records:
        rescore(record, tn=tn, drop_tn=drop_tn)
        update_summary(summary, record)

    write_atomically(
        path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    )
    output = summary_output(summary)
    write_atomically(
        path.with_suffix(".summary.json"),
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
    )
    return output


def describe(name: str, output: dict[str, Any]) -> str:
    parts = [f"{name}:"]
    for key, label in (
        ("mean_wer", "WER"),
        ("mean_cer", "CER"),
        ("mean_whisper_wer", "Whisper-WER"),
        ("mean_whisper_cer", "Whisper-CER"),
        ("mean_tn_wer", "TN-WER"),
        ("mean_tn_cer", "TN-CER"),
    ):
        if key in output:
            parts.append(f"{label}={output[key]:.2f}%")
    return " ".join(parts)


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="result JSONL files to re-score (default: every results/*.jsonl)",
    )
    tn_group = parser.add_mutually_exclusive_group()
    tn_group.add_argument(
        "--tn", action="store_true", help="also compute tn_wer/tn_cer"
    )
    tn_group.add_argument(
        "--drop-tn", action="store_true", help="remove existing TN metrics"
    )
    args = parser.parse_args(argv)

    if args.tn:
        ensure_available()

    paths = args.paths or sorted(RESULTS_DIR.glob("*.jsonl"))
    for path in paths:
        try:
            output = recalculate(path, tn=args.tn, drop_tn=args.drop_tn)
        except StaleTNMetrics as exc:
            parser.error(str(exc))
        print(describe(path.name, output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
