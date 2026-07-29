"""
Closes the depth thread. Reads held-out predictions from phase2_brier.py, adds the
parameter-free PDE prediction, and answers two DISTINCT questions:

  Delta_mapping = B_PDE - B_L1        : held-out improvement from replacing the PDE mapping
                                        with an empirical L1 mapping (reflects too-steep calib)
  Delta_depth   = B_L1  - B_L1+depth  : extra STATE-LEVEL error depth removes WITHIN bins
  Delta_total   = B_PDE - B_L1+depth  : total held-out improvement over the PDE

Ratio Delta_depth / Delta_total = "depth's share of held-out Brier improvement over the PDE
benchmark." NOT causal, NOT a decomposition of the theoretical gap. The aggregate PDE gap is
a property of P(up|I); depth conditioning cannot change it (averaging identity).

Depth terciles are formed WITHIN each L1 bin; the PDE comparator is averaged WITHIN each
(bin, tercile) group -- so low/high-depth states are each compared to their OWN PDE average,
not a pooled one (their exact imbalance distributions can differ slightly within a bin).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = r"C:\Users\evezh\Downloads\OrderFlowReactor"
PRED = os.path.join(BASE, "phase2_predictions.parquet")
PLOT = os.path.join(BASE, "depth_vs_pde.png")
RESID_PLOT = os.path.join(BASE, "depth_pde_residuals.png")
N_L1_BINS = 20


def pde_prob(I):
    eps = 1e-12
    I = np.clip(I, eps, 1 - eps)
    return 1.0 - (2.0 / np.pi) * np.arctan((1.0 - I) / I)


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def main():
    df = pd.read_parquet(PRED)
    y = df["is_up"].to_numpy()
    df["pred_PDE"] = pde_prob(df["imbalance"].to_numpy())

    b_pde = brier(df["pred_PDE"].to_numpy(), y)
    b_l1 = brier(df["pred_L1"].to_numpy(), y)
    b_l1d = brier(df["pred_L1_depth"].to_numpy(), y)
    d_map, d_dep, d_tot = b_pde - b_l1, b_l1 - b_l1d, b_pde - b_l1d

    print("Held-out Brier (pooled across four folds):")
    print(f"  PDE={b_pde:.5f}   L1={b_l1:.5f}   L1+depth={b_l1d:.5f}")
    print(f"  Delta_mapping (PDE-L1):       {d_map:+.6f}   (improvement from replacing PDE with empirical L1 mapping)")
    print(f"  Delta_depth   (L1-L1+depth):  {d_dep:+.6f}   (within-bin sharpening from depth)")
    print(f"  Delta_total   (PDE-L1+depth): {d_tot:+.6f}")
    if d_tot != 0:
        print(f"  depth's share of held-out Brier improvement over PDE: {100*d_dep/d_tot:.1f}%")
    print("  (predictive-error accounting -- NOT causal, NOT a gap decomposition)")

    print("\nPer-fold:")
    for date, g in df.groupby("date"):
        yd = g["is_up"].to_numpy()
        bp, bl, bd = (brier(g["pred_PDE"].to_numpy(), yd),
                      brier(g["pred_L1"].to_numpy(), yd),
                      brier(g["pred_L1_depth"].to_numpy(), yd))
        print(f"  {date}: PDE={bp:.6f} L1={bl:.6f} L1+depth={bd:.6f} "
              f"mapping={bp-bl:+.6f} depth={bl-bd:+.6f}")

    # within-bin depth terciles (vectorized)
    edges = np.linspace(0.0, 1.0, N_L1_BINS + 1)
    df["l1_bin"] = np.clip(np.digitize(df["imbalance"], edges) - 1, 0, N_L1_BINS - 1)
    pct = df.groupby("l1_bin")["deep_imb"].rank(method="first", pct=True).to_numpy()
    df["d_ter_num"] = np.floor(np.minimum(pct, 1 - 1e-12) * 3).astype(np.int8)
    df["d_ter"] = df["d_ter_num"].map({0: "low", 1: "mid", 2: "high"})
    mids = (edges[:-1] + edges[1:]) / 2

    # depth-SPECIFIC empirical AND PDE curves (each tercile vs its own PDE average)
    emp, pde_c = {}, {}
    for t in ["low", "mid", "high"]:
        sub = df[df["d_ter"] == t]
        emp[t] = sub.groupby("l1_bin")["is_up"].mean().reindex(range(N_L1_BINS))
        pde_c[t] = sub.groupby("l1_bin")["pred_PDE"].mean().reindex(range(N_L1_BINS))

    # residual sign analysis (each tercile vs its OWN PDE)
    print("\nWithin-bin residual (empirical - own-PDE) by depth tercile -- sign check:")
    same, total = 0, 0
    counts = df.groupby(["l1_bin", "d_ter"]).size().unstack(fill_value=0)
    for b in range(N_L1_BINS):
        if b not in counts.index or counts.loc[b, ["low", "high"]].min() < 1000:
            continue
        rl = emp["low"][b] - pde_c["low"][b]
        rh = emp["high"][b] - pde_c["high"][b]
        if np.isnan(rl) or np.isnan(rh):
            continue
        total += 1
        tag = "same-sign" if np.sign(rl) == np.sign(rh) else "OPPOSITE"
        if tag == "same-sign": same += 1
        print(f"  bin {mids[b]:.3f}: low_resid={rl:+.3f}  high_resid={rh:+.3f}  -> {tag}")
    if total:
        print(f"\n  {same}/{total} populated bins: low & high depth residuals share the same sign")
        print("   most/all same sign  -> depth separates probabilities within the bin, but does")
        print("                          not identify groups on opposite sides of the PDE")
        print("   many opposite sign  -> depth-conditioned states straddle the PDE, revealing")
        print("                          within-bin heterogeneity hidden by the pooled curve")

    # plot 1: probability curves
    xs = np.linspace(0.01, 0.99, 200)
    plt.figure(figsize=(8, 6))
    plt.plot(xs, pde_prob(xs), color="black", lw=1.5, label="PDE (theory)")
    for t, c in [("low", "#1D9E75"), ("mid", "#E8833A"), ("high", "#534AB7")]:
        plt.plot(mids, emp[t], marker="o", ms=3, color=c, label=f"empirical, {t} deep-imb")
    plt.axhline(0.5, color="#eee", ls=":", lw=1)
    plt.xlabel("L1 imbalance"); plt.ylabel("P(up)")
    plt.title("Depth-conditioned empirical curves vs. PDE (within-bin terciles)")
    plt.xlim(0, 1); plt.ylim(0, 1); plt.legend(fontsize=8); plt.grid(alpha=0.2)
    plt.tight_layout(); plt.savefig(PLOT, dpi=130)

    # plot 2: residual (empirical - own PDE) -- the direct answer
    plt.figure(figsize=(8, 5))
    for t, c in [("low", "#1D9E75"), ("mid", "#E8833A"), ("high", "#534AB7")]:
        plt.plot(mids, emp[t] - pde_c[t], marker="o", ms=3, color=c, label=f"{t} deep-imb")
    plt.axhline(0.0, color="black", lw=1)
    plt.xlabel("L1 imbalance"); plt.ylabel("empirical P(up) - PDE P(up)")
    plt.title("PDE residual by depth tercile at ~fixed L1")
    plt.legend(fontsize=8); plt.grid(alpha=0.2)
    plt.tight_layout(); plt.savefig(RESID_PLOT, dpi=130)

    print(f"\nsaved -> {PLOT}")
    print(f"saved -> {RESID_PLOT}")


if __name__ == "__main__":
    main()
