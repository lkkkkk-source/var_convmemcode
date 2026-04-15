from __future__ import annotations

import argparse
import math
import os
import os.path as osp
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def apply_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def load_var_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "trainer" in ckpt and "var_wo_ddp" in ckpt["trainer"]:
        return ckpt["trainer"]["var_wo_ddp"]
    if "var_wo_ddp" in ckpt:
        return ckpt["var_wo_ddp"]
    raise KeyError(f"Cannot find VAR state dict in checkpoint: {ckpt_path}")


def pca_2d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, vh = np.linalg.svd(x, full_matrices=False)
    return x @ vh[:2].T


def gather_memory_tensors(sd: Dict[str, torch.Tensor], layer_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix = f"blocks.{layer_idx}.attn.knitting_memory"

    shared_per_scale: List[np.ndarray] = []
    for scale_idx in range(10):
        key = f"{prefix}.shared_memory.scale_{scale_idx}"
        mem = sd[key].float().numpy()  # [patterns, mem_size, C]
        shared_per_scale.append(mem.reshape(-1, mem.shape[-1]))

    shared = np.stack(shared_per_scale, axis=0)  # [num_scales, slots, C]

    cat_A = sd[f"{prefix}.cat_A"].float().numpy()  # [num_classes, num_scales, rank]
    cat_B = sd[f"{prefix}.cat_B"].float().numpy()  # [num_scales, rank, slots*C]
    return shared, cat_A, cat_B


def plot_shared_memory_pca(shared: np.ndarray, layer_idx: int, out_dir: str) -> None:
    num_scales, slots, dim = shared.shape
    patterns = 4
    mem_size = max(slots // patterns, 1)

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.flatten()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for scale_idx in range(num_scales):
        ax = axes[scale_idx]
        pts = pca_2d(shared[scale_idx])
        for slot_idx in range(slots):
            pattern_id = slot_idx // mem_size
            ax.scatter(
                pts[slot_idx, 0],
                pts[slot_idx, 1],
                color=colors[pattern_id % len(colors)],
                s=30,
                alpha=0.85,
            )
            ax.text(pts[slot_idx, 0], pts[slot_idx, 1], str(slot_idx), fontsize=7)
        ax.set_title(f"layer{layer_idx} scale {scale_idx}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    fig.suptitle(f"Shared memory PCA — layer {layer_idx}", fontsize=14)
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, f"layer{layer_idx}_shared_memory_pca.png"), bbox_inches="tight")
    plt.close(fig)


def plot_selected_scale_pca(shared: np.ndarray, layer_idx: int, out_dir: str, scales: List[int]) -> None:
    slots = shared.shape[1]
    patterns = 4
    mem_size = max(slots // patterns, 1)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, axes = plt.subplots(1, len(scales), figsize=(4.8 * len(scales), 4.6))
    if len(scales) == 1:
        axes = [axes]

    for ax, scale_idx in zip(axes, scales):
        pts = pca_2d(shared[scale_idx])
        for slot_idx in range(slots):
            pattern_id = slot_idx // mem_size
            ax.scatter(
                pts[slot_idx, 0],
                pts[slot_idx, 1],
                color=colors[pattern_id % len(colors)],
                s=60,
                alpha=0.9,
                edgecolor="black",
                linewidth=0.3,
            )
            ax.text(pts[slot_idx, 0], pts[slot_idx, 1], str(slot_idx), fontsize=8)
        ax.set_title(f"Layer {layer_idx}, scale {scale_idx}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    fig.suptitle(f"Selected shared memory PCA views — layer {layer_idx}", fontsize=14)
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, f"layer{layer_idx}_shared_memory_pca_selected.png"), bbox_inches="tight")
    plt.close(fig)


def plot_shared_norm_heatmap(shared: np.ndarray, layer_idx: int, out_dir: str) -> None:
    norms = np.linalg.norm(shared, axis=-1)  # [num_scales, slots]
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(norms, aspect="auto", cmap="viridis")
    ax.set_title(f"Shared memory slot norms — layer {layer_idx}")
    ax.set_xlabel("slot id")
    ax.set_ylabel("scale id")
    ax.set_xticks(range(norms.shape[1]))
    ax.set_yticks(range(norms.shape[0]))
    fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, f"layer{layer_idx}_shared_norm_heatmap.png"), bbox_inches="tight")
    plt.close(fig)


def plot_shared_norm_bar(shared: np.ndarray, layer_idx: int, out_dir: str) -> None:
    norms = np.linalg.norm(shared, axis=-1)
    mean_norm = norms.mean(axis=1)
    std_norm = norms.std(axis=1)
    x = np.arange(len(mean_norm))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x, mean_norm, yerr=std_norm, color="#4c78a8", alpha=0.85, capsize=4)
    ax.set_title(f"Shared memory norm by scale — layer {layer_idx}")
    ax.set_xlabel("scale id")
    ax.set_ylabel("mean slot norm")
    ax.set_xticks(x)
    for xi, yi in zip(x, mean_norm):
        ax.text(xi, yi + 0.01, f"{yi:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, f"layer{layer_idx}_shared_norm_bar.png"), bbox_inches="tight")
    plt.close(fig)


def plot_class_residual_heatmap(shared: np.ndarray, cat_A: np.ndarray, cat_B: np.ndarray, layer_idx: int, out_dir: str) -> None:
    num_classes, num_scales, rank = cat_A.shape
    residual_strength = np.zeros((num_classes, num_scales), dtype=np.float64)
    residual_ratio = np.zeros((num_classes, num_scales), dtype=np.float64)

    for cls in range(num_classes):
        for scale in range(num_scales):
            delta = (cat_A[cls, scale] @ cat_B[scale]).reshape(shared.shape[1], shared.shape[2])
            residual_strength[cls, scale] = np.linalg.norm(delta)
            shared_norm = np.linalg.norm(shared[scale]) + 1e-8
            residual_ratio[cls, scale] = residual_strength[cls, scale] / shared_norm

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, vals, title, fname in [
        (axes[0], residual_strength, "Residual strength ||A@B||", f"layer{layer_idx}_class_residual_strength.png"),
        (axes[1], residual_ratio, "Residual / shared norm ratio", f"layer{layer_idx}_class_residual_ratio.png"),
    ]:
        im = ax.imshow(vals, aspect="auto", cmap="magma")
        ax.set_title(f"layer {layer_idx} — {title}")
        ax.set_xlabel("scale id")
        ax.set_ylabel("class id")
        ax.set_xticks(range(vals.shape[1]))
        ax.set_yticks(range(vals.shape[0]))
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, f"layer{layer_idx}_class_residual_summary.png"), bbox_inches="tight")
    plt.close(fig)


def plot_class_residual_ratio_annotated(shared: np.ndarray, cat_A: np.ndarray, cat_B: np.ndarray, layer_idx: int, out_dir: str) -> None:
    num_classes, num_scales, _ = cat_A.shape
    residual_ratio = np.zeros((num_classes, num_scales), dtype=np.float64)
    for cls in range(num_classes):
        for scale in range(num_scales):
            delta = (cat_A[cls, scale] @ cat_B[scale]).reshape(shared.shape[1], shared.shape[2])
            residual_ratio[cls, scale] = np.linalg.norm(delta) / (np.linalg.norm(shared[scale]) + 1e-8)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(residual_ratio, aspect="auto", cmap="magma")
    ax.set_title(f"Class residual / shared ratio — layer {layer_idx}")
    ax.set_xlabel("scale id")
    ax.set_ylabel("class id")
    ax.set_xticks(range(num_scales))
    ax.set_yticks(range(num_classes))
    for i in range(num_classes):
        for j in range(num_scales):
            ax.text(j, i, f"{residual_ratio[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, f"layer{layer_idx}_class_residual_ratio_annotated.png"), bbox_inches="tight")
    plt.close(fig)


def plot_class_residual_bar(shared: np.ndarray, cat_A: np.ndarray, cat_B: np.ndarray, layer_idx: int, out_dir: str) -> None:
    num_classes, num_scales, _ = cat_A.shape
    class_means = []
    for cls in range(num_classes):
        ratios = []
        for scale in range(num_scales):
            delta = (cat_A[cls, scale] @ cat_B[scale]).reshape(shared.shape[1], shared.shape[2])
            ratios.append(np.linalg.norm(delta) / (np.linalg.norm(shared[scale]) + 1e-8))
        class_means.append(np.mean(ratios))

    x = np.arange(num_classes)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, class_means, color="#e45756", alpha=0.85)
    ax.set_title(f"Mean class residual ratio by class — layer {layer_idx}")
    ax.set_xlabel("class id")
    ax.set_ylabel("mean residual / shared ratio")
    ax.set_xticks(x)
    for xi, yi in zip(x, class_means):
        ax.text(xi, yi + 0.01, f"{yi:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, f"layer{layer_idx}_class_residual_bar.png"), bbox_inches="tight")
    plt.close(fig)


def save_text_summary(shared: np.ndarray, cat_A: np.ndarray, cat_B: np.ndarray, layer_idx: int, out_dir: str) -> None:
    lines = [f"# Memory summary for layer {layer_idx}"]
    lines.append("")
    lines.append(f"shared tensor shape: {tuple(shared.shape)}  # [num_scales, slots, dim]")
    lines.append(f"cat_A shape: {tuple(cat_A.shape)}")
    lines.append(f"cat_B shape: {tuple(cat_B.shape)}")
    lines.append("")

    shared_norms = np.linalg.norm(shared, axis=-1)
    lines.append("## Shared memory norm statistics")
    for scale in range(shared.shape[0]):
        vals = shared_norms[scale]
        lines.append(
            f"scale_{scale}: mean={vals.mean():.4f}, std={vals.std():.4f}, min={vals.min():.4f}, max={vals.max():.4f}"
        )

    lines.append("")
    lines.append("## Class residual ratio statistics")
    for cls in range(cat_A.shape[0]):
        ratios = []
        for scale in range(cat_A.shape[1]):
            delta = (cat_A[cls, scale] @ cat_B[scale]).reshape(shared.shape[1], shared.shape[2])
            ratio = np.linalg.norm(delta) / (np.linalg.norm(shared[scale]) + 1e-8)
            ratios.append(ratio)
        ratios = np.array(ratios)
        lines.append(
            f"class_{cls}: mean_ratio={ratios.mean():.4f}, min_ratio={ratios.min():.4f}, max_ratio={ratios.max():.4f}"
        )

    with open(osp.join(out_dir, f"layer{layer_idx}_memory_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize learned class-aware memory from a checkpoint")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--out_dir", type=str, default="./memory_viz", help="Output directory")
    parser.add_argument("--layers", type=str, default="8_12", help="Memory layers to visualize, e.g. 8_12")
    args = parser.parse_args()

    apply_style()
    os.makedirs(args.out_dir, exist_ok=True)
    sd = load_var_state_dict(args.ckpt)
    layers = [int(x) for x in args.layers.replace("-", "_").split("_") if x]

    for layer_idx in layers:
        layer_out = osp.join(args.out_dir, f"layer{layer_idx}")
        os.makedirs(layer_out, exist_ok=True)
        shared, cat_A, cat_B = gather_memory_tensors(sd, layer_idx)
        plot_shared_memory_pca(shared, layer_idx, layer_out)
        plot_selected_scale_pca(shared, layer_idx, layer_out, scales=[0, 4, 8, 9])
        plot_shared_norm_heatmap(shared, layer_idx, layer_out)
        plot_shared_norm_bar(shared, layer_idx, layer_out)
        plot_class_residual_heatmap(shared, cat_A, cat_B, layer_idx, layer_out)
        plot_class_residual_ratio_annotated(shared, cat_A, cat_B, layer_idx, layer_out)
        plot_class_residual_bar(shared, cat_A, cat_B, layer_idx, layer_out)
        save_text_summary(shared, cat_A, cat_B, layer_idx, layer_out)
        print(f"Saved visualizations for layer {layer_idx} -> {layer_out}")

    print(f"Done. Output dir: {args.out_dir}")


if __name__ == "__main__":
    main()
