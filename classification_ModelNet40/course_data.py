"""Dataset helpers for BUPT ML course ModelNet40 prediction.

The course examples and Pointcept/modelnet40_normal_resampled-compressed use
ModelNet40's normal-resampled text layout:

    modelnet40_normal_resampled/<class>/<sample_id>.txt

Each file contains 10,000 comma-separated rows with six values:
x, y, z, nx, ny, nz.  The official PointMLP ModelNet40 model consumes xyz only,
so this module slices to the first three channels by default.
"""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

_METADATA_NAMES = {
    "filelist.txt",
    "modelnet40_shape_names.txt",
    "modelnet40_train.txt",
    "modelnet40_test.txt",
}
_SUPPORTED_SUFFIXES = {".txt", ".csv", ".npy"}


@dataclass(frozen=True)
class PointCloudSample:
    """A discovered sample in either a directory tree or a zip archive."""

    sample_id: str
    path: str
    zip_member: Optional[str] = None


def _is_supported_sample(path: str) -> bool:
    name = os.path.basename(path)
    suffix = os.path.splitext(name)[1].lower()
    return suffix in _SUPPORTED_SUFFIXES and name not in _METADATA_NAMES


def _sample_id_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _discover_dir(root: Path) -> List[PointCloudSample]:
    samples: List[PointCloudSample] = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            full_path = Path(dirpath) / filename
            if _is_supported_sample(str(full_path)):
                samples.append(
                    PointCloudSample(
                        sample_id=_sample_id_from_path(filename),
                        path=str(full_path),
                    )
                )
    return sorted(samples, key=lambda item: item.sample_id)


def _discover_zip(zip_path: Path) -> List[PointCloudSample]:
    samples: List[PointCloudSample] = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.endswith("/") or not _is_supported_sample(member):
                continue
            samples.append(
                PointCloudSample(
                    sample_id=_sample_id_from_path(member),
                    path=str(zip_path),
                    zip_member=member,
                )
            )
    return sorted(samples, key=lambda item: item.sample_id)


def discover_samples(root: str | os.PathLike[str]) -> List[PointCloudSample]:
    """Discover point-cloud samples under a directory or inside a .zip file."""
    root_path = Path(root)
    if root_path.is_file() and root_path.suffix.lower() == ".zip":
        samples = _discover_zip(root_path)
    elif root_path.is_dir():
        samples = _discover_dir(root_path)
    else:
        raise FileNotFoundError(f"test data path does not exist or is unsupported: {root}")

    if not samples:
        raise RuntimeError(f"no point-cloud samples found under: {root}")
    return samples


def _load_text_from_bytes(raw: bytes) -> np.ndarray:
    text = raw.decode("utf-8").strip()
    if not text:
        raise ValueError("empty point-cloud text file")
    delimiter = "," if "," in text.splitlines()[0] else None
    from io import StringIO

    return np.loadtxt(StringIO(text), delimiter=delimiter, dtype=np.float32)


def _load_array(sample: PointCloudSample) -> np.ndarray:
    suffix = os.path.splitext(sample.zip_member or sample.path)[1].lower()
    if sample.zip_member is not None:
        with zipfile.ZipFile(sample.path) as zf:
            raw = zf.read(sample.zip_member)
        if suffix == ".npy":
            from io import BytesIO

            return np.load(BytesIO(raw)).astype(np.float32)
        return _load_text_from_bytes(raw)

    if suffix == ".npy":
        return np.load(sample.path).astype(np.float32)
    delimiter = "," if suffix == ".csv" else None
    try:
        return np.loadtxt(sample.path, delimiter=delimiter, dtype=np.float32)
    except ValueError:
        # ModelNet txt files are comma-separated even when the suffix is .txt.
        return np.loadtxt(sample.path, delimiter=",", dtype=np.float32)


def normalize_unit_sphere(points: np.ndarray) -> np.ndarray:
    """Center xyz and scale to unit sphere; normals are left untouched if present."""
    out = points.copy()
    xyz = out[:, :3]
    centroid = xyz.mean(axis=0, keepdims=True)
    xyz = xyz - centroid
    radius = np.sqrt((xyz ** 2).sum(axis=1)).max()
    if radius > 0:
        xyz = xyz / radius
    out[:, :3] = xyz
    return out


def deterministic_resample(points: np.ndarray, num_points: int) -> np.ndarray:
    """Return exactly num_points rows without introducing runtime randomness."""
    if points.shape[0] == num_points:
        return points
    if points.shape[0] > num_points:
        idx = np.linspace(0, points.shape[0] - 1, num_points, dtype=np.int64)
        return points[idx]
    repeats = int(np.ceil(num_points / points.shape[0]))
    tiled = np.tile(points, (repeats, 1))
    return tiled[:num_points]


class CourseTestDataset(Dataset):
    """Unlabeled course/onsite ModelNet40 test set.

    Returns (sample_id, points), where points has shape (num_points, 3) by default
    or (num_points, 6) when use_normals=True.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        num_points: int = 1024,
        use_normals: bool = False,
        normalize: bool = False,
    ) -> None:
        self.root = str(root)
        self.num_points = int(num_points)
        self.use_normals = bool(use_normals)
        self.normalize = bool(normalize)
        self.samples = discover_samples(root)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor]:
        sample = self.samples[index]
        points = _load_array(sample)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(
                f"sample {sample.sample_id} must have shape (N, C>=3), got {points.shape}"
            )
        if self.normalize:
            points = normalize_unit_sphere(points)
        points = deterministic_resample(points, self.num_points)
        channels = 6 if self.use_normals and points.shape[1] >= 6 else 3
        points = points[:, :channels].astype(np.float32, copy=False)
        return sample.sample_id, torch.from_numpy(points)
