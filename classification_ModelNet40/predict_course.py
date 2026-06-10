"""Predict BUPT ML course ModelNet40 labels and write submission CSV.

Example:
    cd classification_ModelNet40
    python predict_course.py \
      --model pointMLP \
      --checkpoint checkpoints/pointMLP-official/best_checkpoint.pth \
      --test_dir /path/to/modelnet40_test_data \
      --output result.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from course_data import CourseTestDataset, deterministic_resample
from modelnet40_classes import class_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("BUPT ModelNet40 course prediction")
    parser.add_argument("--model", default="pointMLP", help="model factory name in models/__init__.py")
    parser.add_argument("--checkpoint", required=True, help="path to .pth checkpoint")
    parser.add_argument("--test_dir", required=True, help="directory or .zip containing onsite samples")
    parser.add_argument("--output", required=True, help="submission CSV path")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--num_votes", type=int, default=1, help="average predictions across point resampling votes")
    parser.add_argument(
        "--vote_sampling",
        choices=["deterministic", "random"],
        default="deterministic",
        help="deterministic reuses the dataset sample; random resamples points per vote",
    )
    parser.add_argument("--use_normals", action="store_true", help="pass xyz+normal channels if the model supports 6D input")
    parser.add_argument("--normalize", action="store_true", help="center xyz and scale each sample to unit sphere")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--header", action="store_true", help="write CSV header: id,pred_label")
    parser.add_argument("--dry_run", action="store_true", help="load data and model shape only; do not require checkpoint")
    return parser.parse_args()


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return OrderedDict((key.removeprefix("module."), value) for key, value in state_dict.items())


def load_checkpoint_state(path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("net", "state_dict", "model", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return strip_module_prefix(checkpoint[key])
    if isinstance(checkpoint, dict):
        return strip_module_prefix(checkpoint)
    raise TypeError(f"unsupported checkpoint format: {type(checkpoint)!r}")


def build_model(model_name: str, device: torch.device) -> torch.nn.Module:
    import models as models

    if model_name not in models.__dict__ or not callable(models.__dict__[model_name]):
        available = sorted(name for name, value in models.__dict__.items() if callable(value) and not name.startswith("_"))
        raise KeyError(f"unknown model {model_name!r}; available: {available}")
    model = models.__dict__[model_name]().to(device)
    model.eval()
    return model


def collate_batch(batch: List[Tuple[str, torch.Tensor]]) -> Tuple[List[str], torch.Tensor]:
    sample_ids = [item[0] for item in batch]
    points = torch.stack([item[1] for item in batch], dim=0)
    return sample_ids, points


def random_resample_tensor(points: torch.Tensor, num_points: int) -> torch.Tensor:
    """Randomly resample a batch of B,N,C tensors to B,num_points,C."""
    batch_size, point_count, _ = points.shape
    if point_count == num_points:
        return points
    samples = []
    for batch_idx in range(batch_size):
        if point_count > num_points:
            idx = torch.randperm(point_count, device=points.device)[:num_points]
        else:
            idx = torch.randint(0, point_count, (num_points,), device=points.device)
        samples.append(points[batch_idx, idx])
    return torch.stack(samples, dim=0)


def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_votes: int,
    vote_sampling: str = "deterministic",
    num_points: int = 1024,
) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []
    votes = max(1, int(num_votes))
    with torch.no_grad():
        for sample_ids, points in loader:
            points = points.to(device, non_blocking=True)
            probs_sum = torch.zeros(points.shape[0], 40, device=device)
            for vote_idx in range(votes):
                vote_points = points
                if vote_sampling == "random":
                    vote_points = random_resample_tensor(points, num_points)
                elif vote_idx == 0 and points.shape[1] != num_points:
                    # Defensive path for custom collators; CourseTestDataset already resamples.
                    vote_points = torch.as_tensor(
                        deterministic_resample(points.cpu().numpy(), num_points),
                        device=device,
                        dtype=points.dtype,
                    )
                # PointMLP expects (B, C, N). Official ModelNet40 PointMLP consumes xyz.
                logits = model(vote_points.permute(0, 2, 1).contiguous())
                probs_sum = probs_sum + F.softmax(logits, dim=1)
            probs = probs_sum / votes
            pred_indices = probs.argmax(dim=1).detach().cpu().tolist()
            results.extend((sample_id, class_name(label_idx)) for sample_id, label_idx in zip(sample_ids, pred_indices))
    return results


def write_csv(path: str, rows: Iterable[Tuple[str, str]], header: bool = False) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(["id", "pred_label"])
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset = CourseTestDataset(
        args.test_dir,
        num_points=args.num_points,
        use_normals=args.use_normals,
        normalize=args.normalize,
        resample=args.vote_sampling != "random",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )
    print(f"Loaded {len(dataset)} samples from {args.test_dir}")

    model = build_model(args.model, device)
    if not args.dry_run:
        state_dict = load_checkpoint_state(args.checkpoint)
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("Dry run: checkpoint loading skipped")

    rows = predict(model, loader, device, args.num_votes, args.vote_sampling, args.num_points)
    write_csv(args.output, rows, header=args.header)
    print(f"Wrote {len(rows)} predictions to {args.output}")
    if rows:
        print("First prediction: %s,%s" % rows[0])


if __name__ == "__main__":
    main()
