"""
Does total book VOLUME (thickness) condition the imbalance signal -- tested at increasing
depth ranges: level 1, levels 1-2, levels 1-5.

L1 alone is floored near zero and right-skewed. Summing more levels gives a fuller, smoother
measure of total resting liquidity. Testing L1 / L1-2 / L1-5 shows whether thickness starts
to matter as you include more depth, rather than assuming "deeper is better."

Method (same as the depth test): within each fixed L1-imbalance bin, split states by total
volume into three sigma buckets -- THIN (< mean-1sd), NORMAL (within +/-1sd), THICK (> mean+1sd)
-- and compare P(up). A consistent, clean (low imb-gap) thick-minus-thin spread = thickness
carries information beyond the imbalance ratio.
"""

import os
import numpy as np
import pandas as pd

BASE = r"C:\Users\evezh\Downloads\OrderFlowReactor"
DATE = "20260521"
N_L1_BINS = 20
DEPTH_RANGES = {"L1": [1], "L1-2": [1, 2], "L1-5": [1, 2, 3, 4, 5]}


def main():
    # load everything needed: L1-5 sizes both sides + label inputs
    lv = list(range(1, 6))
    cols = ["seq", "mid", "imbalance"] + \
           [f"bid_sz_{i}" for i in lv] + [f"ask_sz_{i}" for i in lv]
    df = pd.read_parquet(os.path.join(BASE, f"MNQ_book_states_{DATE}.parquet"), columns=cols)
    df = df.sort_values("seq").reset_index(drop=True)

    is_move = df["mid"] != df["mid"].shift(1)
    df["seg"] = is_move.cumsum()
    moves = df[is_move].reset_index(drop=True)
    moves["next_mid"] = moves["mid"].shift(-1)
    seg_next = moves["next_mid"].copy(); seg_next.index = range(1, len(moves) + 1)
    df["nm"] = df["seg"].map(seg_next)
    df = df[df["nm"].notna()].copy()
    df["is_up"] = (df["nm"] > df["mid"]).astype(int)

    edges = np.linspace(0.0, 1.0, N_L1_BINS + 1)
    df["l1_bin"] = np.clip(np.digitize(df["imbalance"], edges) - 1, 0, N_L1_BINS - 1)
    mids = (edges[:-1] + edges[1:]) / 2

    for name, levels in DEPTH_RANGES.items():
        vol = sum(df[f"bid_sz_{i}"] for i in levels) + sum(df[f"ask_sz_{i}"] for i in levels)
        df["vol"] = vol
        print(f"\n===== depth range {name}  (total size: median={vol.median():.0f} "
              f"mean={vol.mean():.1f} std={vol.std():.1f}) =====")
        print(f"{'L1 bin':>7} {'n':>9} {'P(thin)':>8} {'P(thick)':>9} {'thick-thin':>10} "
              f"{'n_thin':>8} {'n_thick':>8} {'imb gap':>8}")
        print("-" * 74)

        spreads = []
        for b, g in df.groupby("l1_bin", observed=True):
            if len(g) < 5000:
                continue
            m, s = g["vol"].mean(), g["vol"].std()
            thin = g[g["vol"] < m - s]
            thick = g[g["vol"] > m + s]
            if len(thin) < 500 or len(thick) < 500:
                continue
            p_thin, p_thick = thin["is_up"].mean(), thick["is_up"].mean()
            sp = p_thick - p_thin
            imb_gap = thick["imbalance"].mean() - thin["imbalance"].mean()
            spreads.append(sp)
            print(f"{mids[b]:>7.3f} {len(g):>9,} {p_thin:>8.3f} {p_thick:>9.3f} {sp:>+10.3f} "
                  f"{len(thin):>8,} {len(thick):>8,} {imb_gap:>+8.4f}")

        if spreads:
            spreads = np.array(spreads)
            print("-" * 74)
            print(f"  mean thick-thin: {spreads.mean():+.4f}   all same sign? "
                  f"{bool((spreads > 0).all() or (spreads < 0).all())}   "
                  f"(usable bins: {len(spreads)})")

    print("\nReading it:")
    print("  Compare the mean thick-thin across the three depth ranges:")
    print("    grows and becomes consistent as depth increases -> thickness matters, but only")
    print("       when measured deep enough (L1 alone was too noisy)")
    print("    stays ~0 / mixed at all three -> total volume adds nothing beyond the ratio,")
    print("       at any depth (strong null)")
    print("  Always check imb gap: large (>0.005) means the split is confounded by residual")
    print("     imbalance, so the spread is not a clean thickness effect.")


if __name__ == "__main__":
    main()
