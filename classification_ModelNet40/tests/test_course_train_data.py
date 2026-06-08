"""Smoke tests for labeled txt ModelNet40 training dataset.

Run from repository root:
    python3 classification_ModelNet40/tests/test_course_train_data.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from course_train_data import CourseModelNet40, discover_labeled_samples
from modelnet40_classes import CLASS_TO_IDX


def write_txt(path: Path, rows: int) -> None:
    arr = np.arange(rows * 6, dtype=np.float32).reshape(rows, 6) / 100.0
    np.savetxt(path, arr, delimiter=",", fmt="%.6f")


def make_fixture() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="course-train-"))
    root = tmp / "modelnet40_normal_resampled"
    for class_name in ["airplane", "chair"]:
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for i in range(5):
            write_txt(class_dir / f"{class_name}_{i:04d}.txt", rows=20 + i)
    (root / "modelnet40_shape_names.txt").write_text("airplane\nchair\n", encoding="utf-8")
    return tmp


def test_labeled_dataset_split_and_shapes() -> None:
    tmp = make_fixture()
    try:
        samples = discover_labeled_samples(tmp)
        assert len(samples) == 10

        train = CourseModelNet40(tmp, split="train", num_points=16, split_ratio=0.8, augment=False)
        test = CourseModelNet40(tmp, split="test", num_points=16, split_ratio=0.8)
        assert len(train) == 8
        assert len(test) == 2

        points, label = train[0]
        assert tuple(points.shape) == (16, 3)
        assert tuple(label.shape) == (1,)
        assert int(label.item()) == CLASS_TO_IDX["airplane"]

        normals = CourseModelNet40(tmp, split="all", num_points=16, use_normals=True, augment=False)
        points6, _ = normals[0]
        assert tuple(points6.shape) == (16, 6)
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_labeled_dataset_split_and_shapes()
    print("course train dataset smoke tests passed")
