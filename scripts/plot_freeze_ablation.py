from plot_clean_param_panels import OUT_DIR, FID_COLOR, KID_COLOR, apply_style, plot_numeric_line


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()

    labels = ["freeze 0-7", "freeze 0-9", "freeze 0-10", "freeze 0-11"]
    x = list(range(len(labels)))
    fid = [48.1733, 45.9512, 46.9420, 44.8140]
    kid = [0.012261, 0.009708, 0.008023, 0.009480]

    plot_numeric_line(
        x,
        fid,
        labels,
        FID_COLOR,
        "o",
        "freeze_strategy_fid_clean.png",
        xlabel="Frozen backbone layers",
        ylabel="FID ↓",
    )
    plot_numeric_line(
        x,
        kid,
        labels,
        KID_COLOR,
        "o",
        "freeze_strategy_kid_clean.png",
        xlabel="Frozen backbone layers",
        ylabel="KID ↓",
    )

    print(f"Saved clean freeze ablation plots to: {OUT_DIR}")


if __name__ == "__main__":
    main()
