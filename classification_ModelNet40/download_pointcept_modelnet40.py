"""Download Pointcept ModelNet40 normal-resampled txt dataset from Hugging Face.

This downloads the same layout used by the BUPT course examples:

    modelnet40_normal_resampled/<class>/<sample_id>.txt

Example:
    cd classification_ModelNet40
    HF_ENDPOINT=https://hf-mirror.com python download_pointcept_modelnet40.py \
      --output data/modelnet40_normal_resampled
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("download Pointcept ModelNet40 normal-resampled txt data")
    parser.add_argument(
        "--repo_id",
        default="Pointcept/modelnet40_normal_resampled-compressed",
        help="Hugging Face dataset repo id",
    )
    parser.add_argument(
        "--output",
        default="data/modelnet40_normal_resampled",
        help="local output directory",
    )
    parser.add_argument(
        "--hf_endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
        help="HF endpoint; defaults to HF_ENDPOINT or https://hf-mirror.com",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["HF_ENDPOINT"] = args.hf_endpoint

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from exc

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    print(f"HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
    print(f"Downloading {args.repo_id} to {output}")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(output),
        allow_patterns=["*.txt", "*.tar.gz", "*.zip"],
        max_workers=1,
    )
    print(f"Done: {output}")


if __name__ == "__main__":
    main()
