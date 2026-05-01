#!/usr/bin/env bash
# Run ProteinMPNN on Apple Silicon (M-series) with MPS GPU acceleration.
# Forwards all arguments to protein_mpnn_run.py.

set -euo pipefail

# Allow CPU fallback for any op MPS doesn't implement (avoids mid-run crashes).
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Unlock full unified memory pool (default caps at ~75%).
# Set to 0.0 to remove the cap entirely; 0.9 if you want a safety margin.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.0}"

# Optional: control low watermark (when allocator starts reclaiming).
export PYTORCH_MPS_LOW_WATERMARK_RATIO="${PYTORCH_MPS_LOW_WATERMARK_RATIO:-0.0}"

exec python "$(dirname "$0")/protein_mpnn_run.py" "$@"
