"""Labeled ModelNet40 txt dataset for Pointcept/BUPT normal-resampled layout."""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from course_data import deterministic_resample, normalize_unit_sphere
from modelnet40_classes import CLASS_TO_IDX, MODELNET40_CLASSES

_METADATA_NAMES = {
    "filelist.txt",
    "modelnet40_shape_names.txt",
    "modelnet40_train.txt",
    "modelnet40_test.txt",
}


@dataclass(frozen=True)
class LabeledSample:
    path: str
    label: int
    zip_member: Optional[str] = None


def _find_dataset_root(root: str | os.PathLike[str]) -> Path:
    """Accept either the normal_resampled directory or a parent containing it."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"data root does not exist: {root}")
    if root_path.name == "modelnet40_normal_resampled":
        return root_path
    nested = root_path / "modelnet40_normal_resampled"
    if nested.is_dir():
        return nested
    return root_path


def discover_labeled_samples(root: str | os.PathLike[str]) -> List[LabeledSample]:
    """Return sorted labeled samples under class-name directories or in a zip."""
    root_path = Path(root)
    samples: List[LabeledSample] = []
    if root_path.is_file() and root_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(root_path) as zf:
            for member in sorted(zf.namelist()):
                if member.endswith("/") or not member.endswith(".txt"):
                    continue
                name = os.path.basename(member)
                if name in _METADATA_NAMES:
                    continue
                parts = member.split("/")
                for class_name in MODELNET40_CLASSES:
                    if class_name in parts:
                        samples.append(LabeledSample(str(root_path), CLASS_TO_IDX[class_name], member))
                        break
    else:
        dataset_root = _find_dataset_root(root)
        for class_name in MODELNET40_CLASSES:
            class_dir = dataset_root / class_name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.glob("*.txt")):
                if path.name in _METADATA_NAMES:
                    continue
                samples.append(LabeledSample(str(path), CLASS_TO_IDX[class_name]))
    if not samples:
        raise RuntimeError(f"no labeled ModelNet40 txt samples found under: {root}")
    return samples


def split_samples(
    samples: List[LabeledSample],
    split: str,
    split_ratio: float = 0.8,
) -> List[LabeledSample]:
    """Deterministic per-class split to keep class balance."""
    split = split.lower()
    if split in {"all", "full"}:
        return list(samples)
    if split not in {"train", "test", "val", "valid", "validation"}:
        raise ValueError("split must be one of train/test/val/all")
    if not 0.0 < split_ratio < 1.0:
        raise ValueError("split_ratio must be in (0, 1)")

    grouped = {idx: [] for idx in range(len(MODELNET40_CLASSES))}
    for sample in samples:
        grouped[sample.label].append(sample)

    selected: List[LabeledSample] = []
    for label in range(len(MODELNET40_CLASSES)):
        items = sorted(grouped[label], key=lambda item: item.zip_member or item.path)
        if not items:
            continue
        cutoff = max(1, int(len(items) * split_ratio))
        if split == "train":
            selected.extend(items[:cutoff])
        else:
            selected.extend(items[cutoff:])
    return selected


def load_txt_points(sample: LabeledSample) -> np.ndarray:
    if sample.zip_member is not None:
        with zipfile.ZipFile(sample.path) as zf:
            raw = zf.read(sample.zip_member).decode("utf-8")
        from io import StringIO

        try:
            points = np.loadtxt(StringIO(raw), delimiter=",", dtype=np.float32)
        except ValueError:
            points = np.loadtxt(StringIO(raw), dtype=np.float32)
    else:
        try:
            points = np.loadtxt(sample.path, delimiter=",", dtype=np.float32)
        except ValueError:
            points = np.loadtxt(sample.path, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        display_path = sample.zip_member or sample.path
        raise ValueError(f"sample {display_path} must have shape (N, C>=3), got {points.shape}")
    return points


def random_resample(points: np.ndarray, num_points: int) -> np.ndarray:
    if points.shape[0] == num_points:
        return points
    if points.shape[0] > num_points:
        idx = np.random.choice(points.shape[0], num_points, replace=False)
        return points[idx]
    idx = np.random.choice(points.shape[0], num_points, replace=True)
    return points[idx]


def translate_pointcloud_xyz(points: np.ndarray) -> np.ndarray:
    """Official PointMLP-style xyz scale/translate augmentation."""
    out = points.copy()
    scale = np.random.uniform(low=2.0 / 3.0, high=3.0 / 2.0, size=[3]).astype(np.float32)
    shift = np.random.uniform(low=-0.2, high=0.2, size=[3]).astype(np.float32)
    out[:, :3] = out[:, :3] * scale + shift
    return out.astype(np.float32, copy=False)


class CourseModelNet40(Dataset):
    """Labeled ModelNet40 normal-resampled txt dataset.

    The default returns xyz-only tensors with shape (num_points, 3), matching the
    official PointMLP ModelNet40 model. Set use_normals=True only for models that
    explicitly support six input channels.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        split: str = "train",
        num_points: int = 1024,
        split_ratio: float = 0.8,
        use_normals: bool = False,
        normalize: bool = False,
        augment: bool | None = None,
    ) -> None:
        self.root = str(root)
        self.dataset_root = str(_find_dataset_root(root))
        self.split = split.lower()
        self.num_points = int(num_points)
        self.split_ratio = float(split_ratio)
        self.use_normals = bool(use_normals)
        self.normalize = bool(normalize)
        self.augment = (self.split == "train") if augment is None else bool(augment)
        all_samples = discover_labeled_samples(root)
        self.samples = split_samples(all_samples, self.split, self.split_ratio)
        if not self.samples:
            raise RuntimeError(f"empty {split} split under: {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        points = load_txt_points(sample)
        if self.normalize:
            points = normalize_unit_sphere(points)
        points = random_resample(points, self.num_points) if self.augment else deterministic_resample(points, self.num_points)
        if self.augment:
            points = translate_pointcloud_xyz(points)
            np.random.shuffle(points)
        channels = 6 if self.use_normals and points.shape[1] >= 6 else 3
        points = points[:, :channels].astype(np.float32, copy=False)
        return torch.from_numpy(points), torch.tensor([sample.label], dtype=torch.int64)
