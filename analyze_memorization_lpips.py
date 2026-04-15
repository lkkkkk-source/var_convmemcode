from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    import lpips  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The 'lpips' package is required for this script. Install it with: pip install lpips"
    ) from exc


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


@dataclass
class Summary:
    mode: str
    query_count: int
    reference_count: int
    duplicate_threshold: float
    duplicate_count: int
    duplicate_ratio: float
    mean_nn_lpips: float
    median_nn_lpips: float
    min_nn_lpips: float
    max_nn_lpips: float


class ImagePathDataset(Dataset):
    def __init__(self, paths: Sequence[Path], image_size: int):
        self.paths = list(paths)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        tensor = tensor * 2.0 - 1.0  # [0,1] -> [-1,1]
        return tensor, str(path)


def list_images(root: Path) -> List[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def subsample(paths: Sequence[Path], count: int, seed: int) -> List[Path]:
    paths = list(paths)
    if count <= 0 or count >= len(paths):
        return paths
    rng = random.Random(seed)
    chosen = paths[:]
    rng.shuffle(chosen)
    return sorted(chosen[:count])


def extract_embeddings(
    paths: Sequence[Path],
    image_size: int,
    batch_size: int,
    workers: int,
    device: str,
    loss_fn,
) -> tuple[torch.Tensor, List[str]]:
    dataset = ImagePathDataset(paths, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    feats: List[torch.Tensor] = []
    ordered_paths: List[str] = []

    with torch.no_grad():
        for batch, batch_paths in loader:
            batch = batch.to(device, non_blocking=True)
            emb = loss_fn.net.forward(batch)
            emb = F.normalize(emb.flatten(1), dim=1)
            feats.append(emb.cpu())
            ordered_paths.extend(batch_paths)

    return torch.cat(feats, dim=0), ordered_paths


def nearest_lpips(
    q_feats: torch.Tensor,
    r_feats: torch.Tensor,
    same_source: bool,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    nn_vals: List[torch.Tensor] = []
    nn_idx: List[torch.Tensor] = []

    for start in range(0, q_feats.size(0), chunk_size):
        q = q_feats[start : start + chunk_size]
        sims = q @ r_feats.T
        dists = ((1.0 - sims).clamp(min=0.0)) / 2.0
        if same_source:
            for row_idx in range(q.size(0)):
                global_idx = start + row_idx
                if global_idx < dists.size(1):
                    dists[row_idx, global_idx] = float("inf")
        vals, idx = torch.min(dists, dim=1)
        nn_vals.append(vals.cpu())
        nn_idx.append(idx.cpu())

    return torch.cat(nn_vals).numpy(), torch.cat(nn_idx).numpy()


def summarize(mode: str, nn_vals: np.ndarray, threshold: float, qn: int, rn: int) -> Summary:
    dup_count = int((nn_vals < threshold).sum())
    return Summary(
        mode=mode,
        query_count=qn,
        reference_count=rn,
        duplicate_threshold=threshold,
        duplicate_count=dup_count,
        duplicate_ratio=dup_count / max(qn, 1),
        mean_nn_lpips=float(nn_vals.mean()),
        median_nn_lpips=float(np.median(nn_vals)),
        min_nn_lpips=float(nn_vals.min()),
        max_nn_lpips=float(nn_vals.max()),
    )


def write_summary(path: Path, summaries: Sequence[Summary]) -> None:
    payload = {item.mode: asdict(item) for item in summaries}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_pairs_csv(path: Path, query_paths: Sequence[str], ref_paths: Sequence[str], nn_vals: np.ndarray, nn_idx: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_path", "nearest_ref_path", "nn_lpips"])
        for qp, idx, val in zip(query_paths, nn_idx, nn_vals):
            writer.writerow([qp, ref_paths[int(idx)], f"{float(val):.6f}"])


def plot_threshold_sweep(path: Path, thresholds: Sequence[float], curves: dict[str, np.ndarray]) -> None:
    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 320,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.16,
            "grid.linewidth": 0.6,
        }
    )
    fig, ax = plt.subplots(1, 1, figsize=(5.4, 3.8))
    colors = {"T-T": "#1f4e79", "G-T": "#b22222", "R-T": "#2e8b57"}
    for label, vals in curves.items():
        ax.plot(thresholds, vals * 100.0, marker="o", linewidth=2.2, markersize=5.2, label=label, color=colors.get(label, None))
    ax.set_xlabel("LPIPS threshold")
    ax.set_ylabel("Duplicate ratio (%)")
    ax.set_title("Near-duplicate sensitivity analysis", fontweight="semibold")
    ax.legend(loc="best")
    ax.grid(True, axis="y")
    fig.tight_layout(pad=0.8)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def ratios_for_thresholds(nn_vals: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    vals = []
    for thr in thresholds:
        vals.append(float((nn_vals < thr).mean()))
    return np.asarray(vals, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="Analyze memorization with LPIPS nearest neighbors.")
    parser.add_argument("--train_root", type=str, required=True, help="Training image root.")
    parser.add_argument("--generated_root", type=str, required=True, help="Generated image root.")
    parser.add_argument("--heldout_root", type=str, default="", help="Optional held-out real image root for R-T analysis.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sample_count", type=int, default=5000, help="Subsample size per mode; <=0 means all.")
    parser.add_argument("--duplicate_threshold", type=float, default=0.1)
    parser.add_argument("--thresholds", type=str, default="0.05,0.075,0.1,0.125,0.15")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=512)
    args = parser.parse_args()

    train_root = Path(args.train_root)
    gen_root = Path(args.generated_root)
    heldout_root = Path(args.heldout_root) if args.heldout_root else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_paths = subsample(list_images(train_root), args.sample_count, args.seed)
    gen_paths = subsample(list_images(gen_root), args.sample_count, args.seed + 1)
    heldout_paths = subsample(list_images(heldout_root), args.sample_count, args.seed + 2) if heldout_root else []

    if not train_paths:
        raise RuntimeError(f"No training images found in: {train_root}")
    if not gen_paths:
        raise RuntimeError(f"No generated images found in: {gen_root}")

    print(f"[LPIPS-NN] train images: {len(train_paths)}")
    print(f"[LPIPS-NN] generated images: {len(gen_paths)}")
    if heldout_paths:
        print(f"[LPIPS-NN] held-out images: {len(heldout_paths)}")

    loss_fn = lpips.LPIPS(net="alex").to(device).eval()

    train_feats, train_order = extract_embeddings(train_paths, args.image_size, args.batch_size, args.workers, device, loss_fn)
    gen_feats, gen_order = extract_embeddings(gen_paths, args.image_size, args.batch_size, args.workers, device, loss_fn)
    heldout_feats, heldout_order = (
        extract_embeddings(heldout_paths, args.image_size, args.batch_size, args.workers, device, loss_fn)
        if heldout_paths
        else (None, [])
    )

    tt_vals, tt_idx = nearest_lpips(train_feats, train_feats, same_source=True, chunk_size=args.chunk_size)
    gt_vals, gt_idx = nearest_lpips(gen_feats, train_feats, same_source=False, chunk_size=args.chunk_size)

    summaries = [
        summarize("T-T", tt_vals, args.duplicate_threshold, len(train_order), len(train_order)),
        summarize("G-T", gt_vals, args.duplicate_threshold, len(gen_order), len(train_order)),
    ]

    write_pairs_csv(out_dir / "tt_pairs.csv", train_order, train_order, tt_vals, tt_idx)
    write_pairs_csv(out_dir / "gt_pairs.csv", gen_order, train_order, gt_vals, gt_idx)

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    curves = {
        "T-T": ratios_for_thresholds(tt_vals, thresholds),
        "G-T": ratios_for_thresholds(gt_vals, thresholds),
    }

    if heldout_feats is not None:
        rt_vals, rt_idx = nearest_lpips(heldout_feats, train_feats, same_source=False, chunk_size=args.chunk_size)
        summaries.append(summarize("R-T", rt_vals, args.duplicate_threshold, len(heldout_order), len(train_order)))
        write_pairs_csv(out_dir / "rt_pairs.csv", heldout_order, train_order, rt_vals, rt_idx)
        curves["R-T"] = ratios_for_thresholds(rt_vals, thresholds)

    write_summary(out_dir / "summary.json", summaries)
    plot_threshold_sweep(out_dir / "threshold_sweep.png", thresholds, curves)

    for item in summaries:
        print(f"[{item.mode}] duplicates={item.duplicate_count}/{item.query_count} ({item.duplicate_ratio * 100:.2f}%), mean_nn_lpips={item.mean_nn_lpips:.4f}")
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
