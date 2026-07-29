"""
Does the deeper book (levels 2-10) carry information beyond top-of-book imbalance?

Logic:
  - The arctan theory sees only level 1.
  - Within an L1-imbalance bin, states look approximately identical to theory.
  - Those states can still differ in their levels 2-10 liquidity.
  - Split each L1 bin into states with high versus low deep-book imbalance.
  - If the high-deep-imbalance group moves up more often, while exact L1
    imbalance is similar across groups, depth appears to contain additional
    predictive information.

Output:
  - P(up) for high and low deep-imbalance states within each L1 bin
  - their probability spread
  - the remaining exact-L1 imbalance gap between the groups

This is an exploratory information test. It does not by itself show that depth
explains the theory-data residual -- that requires an out-of-sample error
comparison (L1-only vs L1+depth), built only if this first look is positive.
"""

import pandas as pd
import numpy as np

# Uses the most recent available state file containing L10 columns.
# This first diagnostic is single-day; multi-day validation comes later.
import glob, os

BASE = r"C:\Users\evezh\Downloads\OrderFlowReactor"
N_BINS = 20
DEPTH_LEVELS = range(2, 11)   # levels 2..10


def load_states():
    """Load rows that have both the label and the L10 size columns."""
    # Prefer a state file (has L10). The labeled files may not carry depth.
    candidates = sorted(glob.glob(os.path.join(BASE, "MNQ_book_states_*.parquet")))
    if not candidates:
        raise FileNotFoundError(
            "Need a state file with L10 size columns (MNQ_book_states_*.parquet). "
            "If only labeled files remain, re-run record_states.py for one day."
        )
    # a state file has bid_sz_2..10 / ask_sz_2..10 but NO label -- so we must
    # re-derive the label here from mid, same first-passage logic as label.py.
    path = candidates[-1]
    print(f"loading {os.path.basename(path)}")
    cols = ["seq", "mid", "imbalance"] + \
           [f"bid_sz_{i}" for i in DEPTH_LEVELS] + \
           [f"ask_sz_{i}" for i in DEPTH_LEVELS]
    df = pd.read_parquet(path, columns=cols)
    df = df.sort_values("seq").reset_index(drop=True)

    # first-passage label (same as label.py): next distinct mid, up=+1 down=-1
    is_move = df["mid"] != df["mid"].shift(1)
    moves = df[is_move].reset_index(drop=True)
    moves["next_mid"] = moves["mid"].shift(-1)
    df["seg"] = is_move.cumsum()
    seg_next = moves["next_mid"].copy(); seg_next.index = range(1, len(moves) + 1)
    df["next_mid"] = df["seg"].map(seg_next)
    df = df[df["next_mid"].notna()].copy()
    df["is_up"] = (df["next_mid"] > df["mid"]).astype(int)
    return df


def main():
    df = load_states()

    depth_cols = [f"bid_sz_{i}" for i in DEPTH_LEVELS] + [f"ask_sz_{i}" for i in DEPTH_LEVELS]

    # sanity: are depth levels actually populated, or silently NaN?
    na_frac = df[depth_cols].isna().mean()
    print("\nNaN fraction per depth column (should be ~0):")
    print(na_frac.sort_values(ascending=False).head())
    zero_frac = (df[depth_cols] == 0).mean().mean()
    print(f"overall fraction of depth cells == 0: {zero_frac:.3f}  "
          f"(some zeros are normal -- empty deep levels)")

    # deep-book imbalance from levels 2-10 only (exclude level 1 -- that's what we hold fixed)
    bid_deep = df[[f"bid_sz_{i}" for i in DEPTH_LEVELS]].sum(axis=1)
    ask_deep = df[[f"ask_sz_{i}" for i in DEPTH_LEVELS]].sum(axis=1)
    total_deep = bid_deep + ask_deep
    df["deep_imb"] = np.where(total_deep > 0, bid_deep / total_deep, 0.5)

    # L1 bins (hold top-of-book imbalance fixed)
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    df["bin"] = pd.cut(df["imbalance"], bins=edges, include_lowest=True)

    #   spread = P(up | high deep-imbalance) - P(up | low deep-imbalance)  within the L1 bin
    #   L1 hi / L1 lo = mean EXACT L1 imbalance in each group -- if these differ, the
    #                   spread is contaminated by residual L1, not a clean depth effect.
    print(f"\n{'L1 bin':>7} {'n':>10} {'P(up)hi':>8} {'P(up)lo':>8} {'spread':>8} "
          f"{'L1 hi':>7} {'L1 lo':>7} {'L1 gap':>8}")
    print("-" * 74)

    rows = []
    for b, g in df.groupby("bin", observed=True):
        if len(g) < 5000:
            continue
        med = g["deep_imb"].median()
        hi = g[g["deep_imb"] > med]    # HIGH deep-imbalance (more bid-side deep size)
        lo = g[g["deep_imb"] <= med]   # LOW deep-imbalance
        if len(hi) < 1000 or len(lo) < 1000:
            continue

        p_hi, p_lo = hi["is_up"].mean(), lo["is_up"].mean()
        spread = p_hi - p_lo
        l1_hi, l1_lo = hi["imbalance"].mean(), lo["imbalance"].mean()   # confound check
        l1_gap = l1_hi - l1_lo

        rows.append((b.mid, len(g), p_hi, p_lo, spread, l1_hi, l1_lo, l1_gap))
        print(f"{b.mid:>7.3f} {len(g):>10,} {p_hi:>8.3f} {p_lo:>8.3f} {spread:>+8.3f} "
              f"{l1_hi:>7.3f} {l1_lo:>7.3f} {l1_gap:>+8.4f}")

    if rows:
        spreads = np.array([r[4] for r in rows])
        l1_gaps = np.abs(np.array([r[7] for r in rows]))
        print("-" * 74)
        print(f"mean spread (high minus low deep-imbalance P(up)): {spreads.mean():+.4f}")
        print(f"spreads all same sign? {bool((spreads > 0).all() or (spreads < 0).all())}")
        print(f"max |L1 gap| between groups: {l1_gaps.max():.4f}  "
              f"(if large, the spread is contaminated by residual L1)")
        print("\nReading it:")
        print("  spread consistently > 0 AND L1 gaps tiny")
        print("     -> depth APPEARS to contain predictive information beyond L1")
        print("  spread ~ 0")
        print("     -> depth adds little once L1 is fixed; look to dynamics (clustering/timescale)")
        print("  spread > 0 BUT L1 gaps large")
        print("     -> NOT a clean depth effect; residual L1 is doing the work. Use narrower")
        print("        bins or a regression controlling continuously for exact L1.")
        print("\nNote: exploratory. A positive clean spread shows depth APPEARS informative;")
        print("it does NOT yet show depth explains the theory gap -- that needs an out-of-sample")
        print("error comparison (L1-only vs L1+depth), built only if this first look is positive.")


if __name__ == "__main__":
    main()