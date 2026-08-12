#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATE = REPO_ROOT / "evaluate_model.py"

DATASETS = ("fleurs:test", "common_voice:test")

OMNILINGUAL_MODELS = (
    "omniASR_CTC_300M",
    "omniASR_CTC_1B",
    "omniASR_CTC_3B",
    "omniASR_CTC_7B",
    "omniASR_LLM_300M_v2",
    "omniASR_LLM_1B_v2",
    "omniASR_LLM_3B_v2",
    "omniASR_LLM_7B_v2",
)

FASTER_WHISPER_MODELS = (
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
    "turbo",
)

NEMO_MODELS = ("nvidia/nemotron-3.5-asr-streaming-0.6b",)

QWEN3_ASR_MODELS = (
    "Qwen/Qwen3-ASR-0.6B",
    "Qwen/Qwen3-ASR-1.7B",
)


@dataclass(frozen=True)
class Job:
    engine: str
    model: str
    dataset: str
    name: str
    extra_args: tuple[str, ...] = ()


def slug(value: str) -> str:
    return (
        value.replace("https://", "")
        .replace("http://", "")
        .replace("/", "-")
        .replace(":", "-")
        .replace("_", "_")
    )


def build_jobs(
    datasets: tuple[str, ...],
    faster_device: str,
    faster_compute_type: str,
    limit: int | None,
) -> list[Job]:
    jobs: list[Job] = []
    name_suffix = f"-limit-{limit}" if limit is not None else ""
    limit_args = ("--limit", str(limit)) if limit is not None else ()
    for dataset in datasets:
        dataset_name = slug(dataset)

        for model in OMNILINGUAL_MODELS:
            jobs.append(
                Job(
                    engine="omnilingual",
                    model=model,
                    dataset=dataset,
                    name=f"{model}-{dataset_name}{name_suffix}",
                    extra_args=limit_args,
                )
            )

        for model in FASTER_WHISPER_MODELS:
            jobs.append(
                Job(
                    engine="faster_whisper",
                    model=model,
                    dataset=dataset,
                    name=f"faster_whisper-{model}-{dataset_name}{name_suffix}",
                    extra_args=(
                        "--device",
                        faster_device,
                        "--compute-type",
                        faster_compute_type,
                        *limit_args,
                    ),
                )
            )

        for model in NEMO_MODELS:
            model_name = model.rsplit("/", maxsplit=1)[-1]
            jobs.append(
                Job(
                    engine="nemo",
                    model=model,
                    dataset=dataset,
                    name=f"{model_name}-{dataset_name}{name_suffix}",
                    extra_args=limit_args,
                )
            )

        for model in QWEN3_ASR_MODELS:
            model_name = model.rsplit("/", maxsplit=1)[-1]
            jobs.append(
                Job(
                    engine="qwen3_asr_transformers",
                    model=model,
                    dataset=dataset,
                    name=f"{model_name}-{dataset_name}{name_suffix}",
                    extra_args=("--device", "cuda", *limit_args),
                )
            )

            jobs.append(
                Job(
                    engine="qwen3_asr_vllm",
                    model=model,
                    dataset=dataset,
                    name=f"{model_name}-vllm-{dataset_name}{name_suffix}",
                    extra_args=("--device", "cuda", *limit_args),
                )
            )

    return jobs


def command_for(job: Job) -> list[str]:
    return [
        str(EVALUATE),
        "--engine",
        job.engine,
        "--model",
        job.model,
        "--dataset",
        job.dataset,
        "--name",
        job.name,
        *job.extra_args,
    ]


def run_job(job: Job, *, force: bool, dry_run: bool) -> int:
    summary = REPO_ROOT / "results" / f"{job.name}.summary.json"
    if summary.exists() and not force:
        print(f"SKIP {job.name}: {summary} exists")
        return 0

    cmd = command_for(job)
    print("\nRUN " + " ".join(cmd), flush=True)
    if dry_run:
        return 0

    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run every Turkish ASR evaluation sequentially."
    )
    parser.add_argument("--force", action="store_true", help="rerun completed jobs")
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="stop at the first failed job",
    )
    parser.add_argument("--faster-device", default="cuda")
    parser.add_argument("--faster-compute-type", default="float16")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        choices=DATASETS,
        help="dataset specs to evaluate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="evaluate only the first N samples for each model/dataset job",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    failures: list[tuple[Job, int]] = []
    for job in build_jobs(
        tuple(args.datasets),
        args.faster_device,
        args.faster_compute_type,
        args.limit,
    ):
        returncode = run_job(job, force=args.force, dry_run=args.dry_run)
        if returncode != 0:
            failures.append((job, returncode))
            print(f"FAIL {job.name}: exit code {returncode}", flush=True)
            if args.stop_on_error:
                break

    if failures:
        print("\nFailed jobs:")
        for job, returncode in failures:
            print(f"- {job.name}: exit code {returncode}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
