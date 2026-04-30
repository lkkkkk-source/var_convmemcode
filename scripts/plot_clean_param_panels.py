from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "TMM" / "img" / "parameter_analysis_clean"
FID_COLOR = "#1f4e79"
KID_COLOR = "#b22222"


def apply_style():
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 320,
            "font.size": 11,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "font.family": "DejaVu Serif",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.linewidth": 1.0,
            "grid.alpha": 0.0,
        }
    )


def plot_numeric_line(x, y, labels, color, marker, out_name, rotate=0, xlabel=None, ylabel=None):
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(
        x,
        y,
        color=color,
        marker=marker,
        linewidth=1.3,
        markersize=7.8,
        markerfacecolor=color,
        markeredgewidth=0.0,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(False)

    y_min = min(y)
    y_max = max(y)
    pad = (y_max - y_min) * 0.12 if y_max > y_min else max(abs(y_max) * 0.02, 0.01)
    ax.set_ylim(y_min - pad * 0.4, y_max + pad)

    fig.tight_layout(pad=0.8)
    fig.savefig(OUT_DIR / out_name, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()

    rank_x = [1, 2, 3, 4]
    rank_fid = [45.1522, 42.6843, 45.3039, 46.7636]
    rank_kid = [0.010480, 0.008637, 0.010132, 0.010353]
    rank_labels = [f"r={v}" for v in rank_x]
    plot_numeric_line(rank_x, rank_fid, rank_labels, FID_COLOR, "^", "rank_fid_clean.png", ylabel="FID ↓")
    plot_numeric_line(rank_x, rank_kid, rank_labels, KID_COLOR, "^", "rank_kid_clean.png", ylabel="KID ↓")

    prior_x = [0, 1, 2]
    prior_labels = ["ATE + CSR", "+ heuristic prior", "+ learned prior"]
    prior_fid = [42.6843, 40.7663, 40.9603]
    prior_kid = [0.008637, 0.007513, 0.007514]
    plot_numeric_line(prior_x, prior_fid, prior_labels, FID_COLOR, "d", "local_prior_fid_clean.png", rotate=8, ylabel="FID ↓")
    plot_numeric_line(prior_x, prior_kid, prior_labels, KID_COLOR, "d", "local_prior_kid_clean.png", rotate=8, ylabel="KID ↓")

    print(f"Saved clean parameter plots to: {OUT_DIR}")


if __name__ == "__main__":
    main()
