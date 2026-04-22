from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "TMM" / "img" / "parameter_analysis"


def apply_style():
    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 320,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "axes.labelweight": "medium",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "grid.alpha": 0.16,
            "grid.linewidth": 0.6,
        }
    )


def _metric_text(val):
    return f"{val:.4f}" if val < 1 else f"{val:.2f}"


def _plot_single_metric(x, vals, labels, title, xlabel, ylabel, color, filename, rotate=0, annotate=True):
    fig, ax = plt.subplots(1, 1, figsize=(5.4, 4.0))
    best_idx = min(range(len(vals)), key=lambda i: vals[i])
    ax.plot(x, vals, marker="o", linewidth=2.3, markersize=5.5, color=color, zorder=2)
    ax.scatter([x[best_idx]], [vals[best_idx]], s=80, color="#d4af37", edgecolors="black", linewidths=0.8, zorder=3)
    ax.set_title(title, pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    if annotate:
        pass
    fig.tight_layout(pad=1.0)
    fig.savefig(OUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def plot_two_metrics(x, fid, kid, labels, title, xlabel, filename, rotate=0, annotate=True):
    stem = Path(filename).stem
    suffix = Path(filename).suffix or ".png"
    _plot_single_metric(
        x,
        fid,
        labels,
        f"{title} (FID)",
        xlabel,
        "FID ↓",
        "#1f4e79",
        f"{stem}_fid{suffix}",
        rotate=rotate,
        annotate=annotate,
    )
    _plot_single_metric(
        x,
        kid,
        labels,
        f"{title} (KID)",
        xlabel,
        "KID ↓",
        "#b22222",
        f"{stem}_kid{suffix}",
        rotate=rotate,
        annotate=annotate,
    )


def plot_seed_stability(seeds, fid, kid, filename):
    stem = Path(filename).stem
    suffix = Path(filename).suffix or ".png"
    for vals, ylabel, color, out_name in [
        (fid, "FID ↓", "#1f4e79", f"{stem}_fid{suffix}"),
        (kid, "KID ↓", "#b22222", f"{stem}_kid{suffix}"),
    ]:
        mean_val = sum(vals) / len(vals)
        best_idx = min(range(len(vals)), key=lambda i: vals[i])
        fig, ax = plt.subplots(1, 1, figsize=(5.4, 4.0))
        ax.plot(seeds, vals, marker="o", linewidth=2.3, markersize=5.5, color=color, zorder=2)
        ax.scatter([seeds[best_idx]], [vals[best_idx]], s=80, color="#d4af37", edgecolors="black", linewidths=0.8, zorder=3)
        ax.axhline(mean_val, linestyle="--", color="gray", linewidth=1.1, label=f"mean={mean_val:.4f}")
        ax.fill_between(seeds, [min(vals)] * len(seeds), [max(vals)] * len(seeds), color=color, alpha=0.08)
        ax.set_xlabel("Random seed")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Multi-seed stability ({ylabel})", pad=8)
        ax.set_xticks(seeds)
        ax.legend(loc="best")
        fig.tight_layout(pad=1.0)
        fig.savefig(OUT_DIR / out_name, bbox_inches="tight")
        plt.close(fig)


def plot_cfg_tradeoff(cfg_scales, fid, kid, lpips_proxy, selected_cfg, filename):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    cmap = plt.get_cmap("viridis")
    scale_min = min(cfg_scales)
    scale_range = max(cfg_scales) - scale_min or 1.0
    colors = [cmap((cfg - scale_min) / scale_range) for cfg in cfg_scales]

    panels = [
        (axes[0], fid, "FID ↓", "Quality–perceptual trade-off (FID)"),
        (axes[1], kid, "KID ↓", "Quality–perceptual trade-off (KID)"),
    ]

    for ax, quality_vals, ylabel, title in panels:
        ax.plot(lpips_proxy, quality_vals, color="#9aa0a6", linewidth=1.2, alpha=0.7, zorder=1)

        for cfg, x, y, color in zip(cfg_scales, lpips_proxy, quality_vals, colors):
            is_selected = abs(cfg - selected_cfg) < 1e-8
            ax.scatter(
                [x],
                [y],
                s=120 if is_selected else 64,
                color=color,
                edgecolors="black",
                linewidths=1.1 if is_selected else 0.6,
                marker="*" if is_selected else "o",
                zorder=3,
            )
            ax.annotate(
                f"{cfg:g}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 8 if is_selected else 6),
                ha="center",
                fontsize=8.5,
            )

        ax.set_title(title, pad=8)
        ax.set_xlabel("LPIPS proxy ↑")
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.grid(True, alpha=0.18)
        ax.text(
            0.03,
            0.04,
            "← lower LPIPS proxy is better\n↓ lower quality metric is better",
            transform=ax.transAxes,
            fontsize=8.2,
            color="#4d4d4d",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#d0d0d0", alpha=0.9),
        )

    fig.suptitle("CFG trade-off with perceptual-distance proxy", y=1.02, fontsize=12, fontweight="semibold")
    fig.text(
        0.5,
        -0.01,
        "LPIPS proxy here is computed against real-image samples from the evaluation summary, and can be replaced by pairwise LPIPS later.",
        ha="center",
        fontsize=8.2,
        color="#555555",
    )
    fig.tight_layout(pad=1.0)
    fig.savefig(OUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()

    # 1) Component ablation
    component_labels = ["Baseline", "Texture only", "Memory only", "Full (v2.2)"]
    component_x = list(range(len(component_labels)))
    component_fid = [47.54, 47.63, 46.61, 42.6843]
    component_kid = [0.00931, 0.01109, 0.01027, 0.008637]
    plot_two_metrics(
        component_x,
        component_fid,
        component_kid,
        component_labels,
        "Component ablation under the locked protocol",
        "Model variant",
        "01_component_ablation.png",
        rotate=12,
    )

    # 2) CFG sweep (v1.5 line)
    cfg_labels = [2.0, 2.5, 3.5, 4.0, 4.5, 5.0, 5.5]
    cfg_fid = [47.43, 46.81, 45.52, 45.31, 44.81, 44.43, 44.26]
    cfg_kid = [0.01196, 0.01146, 0.01034, 0.00995, 0.00948, 0.00900, 0.00852]
    cfg_lpips_proxy = [0.7865, 0.7861, 0.7874, 0.7877, 0.7888, 0.7900, 0.7871]
    plot_two_metrics(
        cfg_labels,
        cfg_fid,
        cfg_kid,
        None,
        "CFG sensitivity",
        "CFG scale",
        "02_cfg_sweep.png",
        annotate=False,
    )
    plot_cfg_tradeoff(
        cfg_labels,
        cfg_fid,
        cfg_kid,
        cfg_lpips_proxy,
        selected_cfg=4.5,
        filename="02_cfg_tradeoff.png",
    )

    # 3) Class-aware rank sweep
    rank_labels = [1, 2, 3, 4]
    rank_fid = [45.1522, 42.6843, 45.3039, 46.7636]
    rank_kid = [0.010480, 0.008637, 0.010132, 0.010353]
    plot_two_metrics(
        rank_labels,
        rank_fid,
        rank_kid,
        None,
        "Category-aware rank sensitivity",
        "Residual rank",
        "03_rank_sweep.png",
        annotate=False,
    )

    # 4) Mainline evolution
    evolution_labels = ["Baseline", "Cyclic shift", "Rank-2 memory", "Mem@8,12", "Learned prior"]
    evolution_x = list(range(len(evolution_labels)))
    evolution_fid = [44.81, 44.18, 43.9799, 42.6843, 40.9603]
    evolution_kid = [0.00948, 0.00890, 0.009177, 0.008637, 0.007514]
    plot_two_metrics(
        evolution_x,
        evolution_fid,
        evolution_kid,
        evolution_labels,
        "Mainline evolution",
        "Configuration",
        "04_mainline_evolution.png",
        rotate=8,
        annotate=False,
    )

    # 5) Texture branch variant analysis
    tex_variant_labels = ["Full texture", "Axial-only", "KV-only"]
    tex_variant_x = list(range(len(tex_variant_labels)))
    tex_variant_fid = [42.6843, 44.4989, 45.7310]
    tex_variant_kid = [0.008637, 0.009104, 0.010647]
    plot_two_metrics(
        tex_variant_x,
        tex_variant_fid,
        tex_variant_kid,
        tex_variant_labels,
        "Texture branch analysis around the v2.2 mainline",
        "Texture formulation",
        "05_texture_variants.png",
        rotate=8,
    )

    # 6) Sampling-time local prior enhancement
    prior_labels = ["ATE + CSR", "+ heuristic prior", "+ learned prior"]
    prior_x = list(range(len(prior_labels)))
    prior_fid = [42.6843, 40.7663, 40.9603]
    prior_kid = [0.008637, 0.007513, 0.007514]
    plot_two_metrics(
        prior_x,
        prior_fid,
        prior_kid,
        prior_labels,
        "Sampling-time local prior enhancement",
        "Inference-time variant",
        "06_local_prior_enhancement.png",
        rotate=8,
        )

    # 7) Multi-seed stability
    seeds = [0, 1, 2]
    seed_fid = [42.6843, 42.6645, 42.1888]
    seed_kid = [0.008637, 0.008467, 0.008425]
    plot_seed_stability(seeds, seed_fid, seed_kid, "07_multiseed_stability.png")

    print(f"Saved parameter analysis plots to: {OUT_DIR}")


if __name__ == "__main__":
    main()
