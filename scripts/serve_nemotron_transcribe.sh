#!/usr/bin/env bash
# Start a vLLM server for NVIDIA Nemotron 3.5 Transcribe, following the
# Hugging Face model card quick starts.
#
# Usage:
#   scripts/serve_nemotron_transcribe.sh [--model <hf-id>] [--spec none|mtp|dflash2] [--port 8000]
#
# Run it inside the NeMo Speech + vLLM 0.32.0 environment (see README).
# --spec none     plain autoregressive decoding (default, the baseline)
# --spec mtp      the model's built-in MTP heads (card quick start default)
# --spec dflash2  the NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2 draft model
#                 (BF16 main model only; needs the vLLM 0.32.0 patch from the
#                 DFlash2 model card)
set -euo pipefail

MODEL="nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
SPEC="none"
PORT="8000"
DFLASH2_MODEL="nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --spec) SPEC="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

export VLLM_PLUGINS=nemo_speechlm
export VLLM_MAX_AUDIO_DECODE_DURATION_S=3600
export VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=180

args=(
  "$MODEL"
  --served-model-name "$MODEL"
  --port "$PORT"
  --trust-remote-code
  --max-model-len 65536
  --max-num-batched-tokens 4096
  --enable-chunked-prefill --no-enable-prefix-caching
  --use-replayssm --mamba-backend flashinfer --mamba-cache-mode none
  --mm-processor-cache-gb 0 --no-async-scheduling
  --reasoning-parser nemotron_v3
  --trust-request-mm-kwargs
)

case "$MODEL" in
  *-FP8)
    export VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel
    args+=(--kv-cache-dtype auto --gpu-memory-utilization 0.85 --max-num-seqs 64
           --limit-mm-per-prompt '{"audio":{"count":1,"length":57600000}}')
    ;;
  *-NVFP4)
    export VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel
    args+=(--kv-cache-dtype fp8 --gpu-memory-utilization 0.85 --max-num-seqs 64
           --limit-mm-per-prompt '{"audio":{"count":1,"length":57600000}}')
    ;;
  *)
    args+=(--dtype bfloat16 --gpu-memory-utilization 0.94 --max-num-seqs 192
           --limit-mm-per-prompt '{"audio":{"count":6,"length":120000000}}')
    ;;
esac

case "$SPEC" in
  none|mtp)
    # CTC word-timestamp middleware requires the V1 model runner.
    export VLLM_USE_V2_MODEL_RUNNER=0
    args+=(--middleware nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware)
    if [[ "$SPEC" == "mtp" ]]; then
      args+=(--speculative-config '{"method":"mtp","num_speculative_tokens":8}')
    fi
    ;;
  dflash2)
    if [[ "$MODEL" != "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B" ]]; then
      echo "DFlash2 is only supported with the BF16 main model" >&2
      exit 2
    fi
    # DFlash2 only runs on vLLM's V2 model runner and has no word timestamps.
    unset VLLM_USE_V2_MODEL_RUNNER
    args+=(
      --speculative-config "{\"method\":\"dflash\",\"num_speculative_tokens\":7,\"model\":\"$DFLASH2_MODEL\"}"
      --hf-overrides '{"ctc_timestamps":{"enabled":false}}'
    )
    ;;
  *)
    echo "--spec must be one of: none, mtp, dflash2" >&2
    exit 2
    ;;
esac

set -x
exec vllm serve "${args[@]}"
