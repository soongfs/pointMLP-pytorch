"""Smoke tests for course dataset/prediction helpers.

Run from repository root:
    python3 classification_ModelNet40/tests/test_course_prediction.py
"""

from __future__ import annotations

import csv
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from course_data import CourseTestDataset
from predict_course import collate_batch, predict, write_csv


class ConstantModel(torch.nn.Module):
    def forward(self, x):
        logits = torch.zeros(x.shape[0], 40, device=x.device)
        logits[:, 8] = 1.0  # chair
        return logits


def write_txt(path: Path, rows: int, delimiter: str = ",") -> None:
    arr = np.arange(rows * 6, dtype=np.float32).reshape(rows, 6) / 100.0
    np.savetxt(path, arr, delimiter=delimiter, fmt="%.6f")


def test_dataset_reads_directory_and_predicts_csv() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="course-predict-"))
    try:
        data_dir = tmp / "modelnet40_normal_resampled" / "chair"
        data_dir.mkdir(parents=True)
        write_txt(data_dir / "chair_0001.txt", rows=10, delimiter=",")
        np.save(data_dir / "chair_0002.npy", np.ones((5, 6), dtype=np.float32))
        (tmp / "modelnet40_normal_resampled" / "modelnet40_shape_names.txt").write_text("chair\n", encoding="utf-8")

        dataset = CourseTestDataset(tmp, num_points=16)
        assert len(dataset) == 2
        sample_id, points = dataset[0]
        assert sample_id == "chair_0001"
        assert tuple(points.shape) == (16, 3)

        loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collate_batch)
        rows = predict(ConstantModel(), loader, torch.device("cpu"), num_votes=1)
        assert rows == [("chair_0001", "chair"), ("chair_0002", "chair")]

        output = tmp / "result.csv"
        write_csv(str(output), rows)
        with output.open(newline="", encoding="utf-8") as f:
            assert list(csv.reader(f)) == [["chair_0001", "chair"], ["chair_0002", "chair"]]
    finally:
        shutil.rmtree(tmp)


def test_dataset_reads_zip() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="course-predict-zip-"))
    try:
        txt_path = tmp / "airplane_0001.txt"
        write_txt(txt_path, rows=20, delimiter=",")
        zip_path = tmp / "sample.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(txt_path, "modelnet40_test_data/airplane_0001.txt")
            zf.writestr("modelnet40_test_data/filelist.txt", "ignored.txt\n")

        dataset = CourseTestDataset(zip_path, num_points=8, use_normals=True)
        assert len(dataset) == 1
        sample_id, points = dataset[0]
        assert sample_id == "airplane_0001"
        assert tuple(points.shape) == (8, 6)
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_dataset_reads_directory_and_predicts_csv()
    test_dataset_reads_zip()
    print("course prediction smoke tests passed")
