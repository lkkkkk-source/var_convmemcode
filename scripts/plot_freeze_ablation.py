from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "TMM" / "img" / "parameter_analysis" / "08_freeze_strategy.png"


def main():
    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 320,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "grid.alpha": 0.16,
            "grid.linewidth": 0.6,
        }
    )

    labels = ["freeze 0--7", "freeze 0--9", "freeze 0--10", "freeze 0--11"]
    x = list(range(len(labels)))
    fid = [48.1733, 45.9512, 46.9420, 44.8140]

    fig, ax = plt.subplots(1, 1, figsize=(5.4, 3.8))
    best_idx = min(range(len(fid)), key=lambda i: fid[i])

    ax.plot(x, fid, marker="o", linewidth=2.4, markersize=5.8, color="#1f4e79")
    ax.scatter([x[best_idx]], [fid[best_idx]], s=80, color="#d4af37", edgecolors="black", linewidths=0.8, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Frozen backbone layers")
    ax.set_ylabel("FID ↓")
    ax.set_title("Freeze-strategy sensitivity", pad=8, fontweight="semibold")
    ax.grid(True, axis="y")

    fig.tight_layout(pad=0.8)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved freeze ablation plot to: {OUT_PATH}")


if __name__ == "__main__":
    main()
