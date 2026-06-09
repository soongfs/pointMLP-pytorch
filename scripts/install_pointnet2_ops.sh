#!/usr/bin/env bash
set -euo pipefail

# Build/install the local CUDA extension after `uv sync` has installed torch.
# Run this on a CUDA machine, e.g. cistn01. CUDA_HOME can be set by the user if
# CUDA is installed in a non-standard location.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

uv pip install setuptools wheel
uv pip install --no-build-isolation ./pointnet2_ops_lib

uv run python - <<'PY'
from pointnet2_ops import pointnet2_utils
print("pointnet2_ops import ok")
PY
