import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LABELED_PATH = r"C:\Users\evezh\Downloads\OrderFlowReactor\MNQ_labeled_20260521.parquet"
PLOT_PATH    = r"C:\Users\evezh\Downloads\OrderFlowReactor\calibration_20260521.png"
N_BINS = 20


def calibrate(labeled_path, plot_path, n_bins=N_BINS):
    df = pd.read_parquet(labeled_path, columns=["imbalance", "label"])
    df["is_up"] = (df["label"] == 1).astype(int)   

    # bin imbalance into n_bins equal-width buckets across [0, 1]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    df["bin"] = pd.cut(df["imbalance"], bins=edges, include_lowest=True)

    # per bin: fraction that went up, and how many samples
    g = df.groupby("bin", observed=True)["is_up"]
    curve = pd.DataFrame({
        "p_up": g.mean(),
        "count": g.size(),
    }).reset_index()
    curve["imb_mid"] = curve["bin"].apply(lambda b: b.mid)  

    # print the table
    print(f"{'imbalance':>12} {'P(up)':>8} {'count':>12}")
    for _, r in curve.iterrows():
        print(f"{r['imb_mid']:>12.3f} {r['p_up']:>8.3f} {int(r['count']):>12,}")

    # plot
    plt.figure(figsize=(7, 6))
    plt.scatter(curve["imb_mid"], curve["p_up"], s=curve["count"] / curve["count"].max() * 200,
                color="#534AB7", zorder=3, label="empirical P(up | imbalance)")
    plt.plot(curve["imb_mid"], curve["p_up"], color="#534AB7", alpha=0.4, zorder=2)
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="0.5 (no signal)")
    plt.plot([0, 1], [0, 1], color="#E24B4A", linestyle=":", linewidth=1, label="naive y = x")
    plt.xlabel("queue imbalance  q_b / (q_b + q_a)")
    plt.ylabel("P(mid ticks up before down)")
    plt.title("MNQ calibration: imbalance vs. next-tick direction (1 day)")
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.legend(); plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=130)
    print(f"\nsaved plot -> {plot_path}")


if __name__ == "__main__":
    calibrate(LABELED_PATH, plot_path=PLOT_PATH)