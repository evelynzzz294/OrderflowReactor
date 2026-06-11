import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LABELED_PATH = r"C:\Users\evezh\Downloads\OrderFlowReactor\MNQ_labeled_20260521.parquet"
PLOT_PATH    = r"C:\Users\evezh\Downloads\OrderFlowReactor\overlay_20260521.png"
N_BINS = 20


def theory_arctan(I):
    """Symmetric Cont–de Larrard hitting probability: P(up) given imbalance I."""
    I = np.clip(I, 1e-9, 1 - 1e-9)
    return 1 - (2 / np.pi) * np.arctan((1 - I) / I)


def overlay(labeled_path, plot_path, n_bins=N_BINS):
    df = pd.read_parquet(labeled_path, columns=["imbalance", "label"])
    df["is_up"] = (df["label"] == 1).astype(int)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    df["bin"] = pd.cut(df["imbalance"], bins=edges, include_lowest=True)
    g = df.groupby("bin", observed=True)["is_up"]
    curve = pd.DataFrame({"p_up": g.mean(), "count": g.size()}).reset_index()
    curve["imb_mid"] = curve["bin"].apply(lambda b: b.mid)
    curve["theory"] = theory_arctan(curve["imb_mid"].astype(float))
    curve["gap"] = curve["p_up"] - curve["theory"]   # empirical minus theory

    print(f"{'imbalance':>10} {'P(up)':>8} {'theory':>8} {'gap':>8} {'count':>12}")
    for _, r in curve.iterrows():
        print(f"{r['imb_mid']:>10.3f} {r['p_up']:>8.3f} {r['theory']:>8.3f} "
              f"{r['gap']:>+8.3f} {int(r['count']):>12,}")

    # smooth theory line
    xs = np.linspace(0.01, 0.99, 200)

    plt.figure(figsize=(7, 6))
    plt.scatter(curve["imb_mid"], curve["p_up"],
                s=curve["count"] / curve["count"].max() * 200,
                color="#534AB7", zorder=4, label="empirical P(up | imbalance)")
    plt.plot(xs, theory_arctan(xs), color="#E8833A", linewidth=2, zorder=3,
             label="arctan theory (ρ = 0)")
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.plot([0, 1], [0, 1], color="#C0392B", linestyle=":", linewidth=1, label="y = x")
    plt.xlabel("queue imbalance  q_b / (q_b + q_a)")
    plt.ylabel("P(mid ticks up before down)")
    plt.title("MNQ: empirical vs. arctan hitting-probability theory (1 day)")
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.legend(); plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=130)
    print(f"\nsaved -> {plot_path}")


if __name__ == "__main__":
    overlay(LABELED_PATH, plot_path=PLOT_PATH)