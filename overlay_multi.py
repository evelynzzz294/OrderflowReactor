import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE = r"C:\Users\evezh\Downloads\OrderFlowReactor"
DAYS = ["20260427", "20260505", "20260513", "20260521"]
PLOT_PATH = os.path.join(BASE, "overlay_multiday.png")
N_BINS = 20


def theory_arctan(I):
    I = np.clip(I, 1e-9, 1 - 1e-9)
    return 1 - (2 / np.pi) * np.arctan((1 - I) / I)


def day_curve(date, edges):
    df = pd.read_parquet(os.path.join(BASE, f"MNQ_labeled_{date}.parquet"),
                         columns=["imbalance", "label"])
    df["is_up"] = (df["label"] == 1).astype(int)
    df["bin"] = pd.cut(df["imbalance"], bins=edges, include_lowest=True)
    g = df.groupby("bin", observed=True)["is_up"]
    c = pd.DataFrame({"p_up": g.mean(), "count": g.size()}).reset_index()
    c["mid"] = c["bin"].apply(lambda b: b.mid)
    return c


def main():
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    curves = {d: day_curve(d, edges) for d in DAYS}

    # stack per-day p_up into a matrix for mean/band
    mids = curves[DAYS[0]]["mid"].to_numpy()
    mat = np.vstack([curves[d].set_index("mid").reindex(mids)["p_up"].to_numpy() for d in DAYS])
    mean_curve = np.nanmean(mat, axis=0)
    std_curve = np.nanstd(mat, axis=0)

    xs = np.linspace(0.01, 0.99, 200)

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))

    # --- Plot 1: per-day curves + theory ---
    for d in DAYS:
        c = curves[d]
        ax[0].plot(c["mid"], c["p_up"], marker="o", ms=3, alpha=0.7, label=d)
    ax[0].plot(xs, theory_arctan(xs), color="black", lw=2, ls="-", label="arctan theory")
    ax[0].axhline(0.5, color="gray", ls="--", lw=1)
    ax[0].set_title("Per-day calibration curves vs. theory")
    ax[0].set_xlabel("imbalance"); ax[0].set_ylabel("P(up)")
    ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.2)

    # --- Plot 2: mean + band + theory ---
    ax[1].fill_between(mids, mean_curve - std_curve, mean_curve + std_curve,
                       alpha=0.25, color="#534AB7", label="±1 std across days")
    ax[1].plot(mids, mean_curve, color="#534AB7", marker="o", ms=3, label="mean empirical")
    ax[1].plot(xs, theory_arctan(xs), color="black", lw=2, label="arctan theory")
    ax[1].axhline(0.5, color="gray", ls="--", lw=1)
    ax[1].set_title("Mean curve + day-to-day band vs. theory")
    ax[1].set_xlabel("imbalance"); ax[1].set_ylabel("P(up)")
    ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.2)

    plt.tight_layout(); plt.savefig(PLOT_PATH, dpi=130)
    print(f"saved -> {PLOT_PATH}")
  
    # the gap table, averaged across days
    print(f"\n{'imbalance':>10} {'mean P(up)':>11} {'std':>7} {'theory':>8} {'gap':>8}")
    for i, m in enumerate(mids):
        th = theory_arctan(m)
        print(f"{m:>10.3f} {mean_curve[i]:>11.3f} {std_curve[i]:>7.3f} "
              f"{th:>8.3f} {mean_curve[i]-th:>+8.3f}")


if __name__ == "__main__":
    main()