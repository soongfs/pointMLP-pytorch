"""Evaluate a PointMLP checkpoint on labeled ModelNet40 txt data.

This is for local validation of voting strategies before onsite submission.
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset

from course_data import deterministic_resample, normalize_unit_sphere
from course_train_data import (
    CourseModelNet40,
    LabeledSample,
    discover_labeled_samples,
    load_txt_points,
    split_samples,
    split_samples_by_file,
)
from predict_course import build_model, load_checkpoint_state, random_resample_tensor


class LabeledEvalDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "test",
        num_points: int = 1024,
        split_ratio: float = 0.8,
        use_normals: bool = False,
        normalize: bool = False,
        resample: bool = True,
    ) -> None:
        base = CourseModelNet40(
            root,
            split="all",
            num_points=num_points,
            split_ratio=split_ratio,
            use_normals=use_normals,
            normalize=normalize,
            augment=False,
        )
        all_samples = discover_labeled_samples(root)
        split_by_file = split_samples_by_file(all_samples, __import__("pathlib").Path(base.dataset_root), split)
        self.samples = split_by_file if split_by_file is not None else split_samples(all_samples, split, split_ratio)
        self.num_points = int(num_points)
        self.use_normals = bool(use_normals)
        self.normalize = bool(normalize)
        self.resample = bool(resample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample: LabeledSample = self.samples[index]
        points = load_txt_points(sample)
        if self.normalize:
            points = normalize_unit_sphere(points)
        if self.resample:
            points = deterministic_resample(points, self.num_points)
        channels = 6 if self.use_normals and points.shape[1] >= 6 else 3
        points = points[:, :channels].astype(np.float32, copy=False)
        return torch.from_numpy(points), torch.tensor(sample.label, dtype=torch.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate course PointMLP checkpoint")
    parser.add_argument("--model", default="pointMLP")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split", default="test", choices=["train", "test", "val", "all"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--num_votes", type=int, default=1)
    parser.add_argument("--vote_sampling", choices=["deterministic", "random"], default="deterministic")
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--use_normals", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def collate(batch):
    points = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.stack([item[1] for item in batch], dim=0)
    return points, labels


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset = LabeledEvalDataset(
        args.data_root,
        split=args.split,
        num_points=args.num_points,
        split_ratio=args.split_ratio,
        use_normals=args.use_normals,
        normalize=args.normalize,
        resample=args.vote_sampling != "random",
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    print(f"Loaded {len(dataset)} labeled samples from {args.data_root} split={args.split}")

    model = build_model(args.model, device)
    model.load_state_dict(load_checkpoint_state(args.checkpoint), strict=True)
    model.eval()

    votes = max(1, int(args.num_votes))
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for points, labels in loader:
            points = points.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            probs_sum = torch.zeros(points.shape[0], 40, device=device)
            for _ in range(votes):
                vote_points = points
                if args.vote_sampling == "random":
                    vote_points = random_resample_tensor(points, args.num_points)
                logits = model(vote_points.permute(0, 2, 1).contiguous())
                probs_sum += F.softmax(logits, dim=1)
            preds = (probs_sum / votes).argmax(dim=1)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())

    inst_acc = 100.0 * metrics.accuracy_score(y_true, y_pred)
    class_acc = 100.0 * metrics.balanced_accuracy_score(y_true, y_pred)
    print(f"Inst Acc: {inst_acc:.3f}%")
    print(f"Class Acc: {class_acc:.3f}%")


if __name__ == "__main__":
    main()
