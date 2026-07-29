"""
Sanity check on the Stage 2 cost: is the ~1.8-tick round-trip spread real, or an artifact?

Compares the spread distribution:
  (a) across ALL states (the whole book), and
  (b) conditional on the signal FIRING (|p-0.5| above a threshold),
for one day. If (b) is wider than (a), the signal fires in wide-spread states -- a real,
interesting finding. If the spread is ~1.8 ticks EVERYWHERE, that is suspicious and points
at the reconstruction rather than the strategy.
"""

import os
import numpy as np
import pandas as pd

BASE = r"C:\Users\evezh\Downloads\OrderFlowReactor"
DATE = "20260521"
N_L1_BINS = 20
TICK = 250_000_000
L1_EDGES = np.linspace(0.0, 1.0, N_L1_BINS + 1)


def main():
    df = pd.read_parquet(os.path.join(BASE, f"MNQ_book_states_{DATE}.parquet"),
                         columns=["seq", "mid", "imbalance", "best_bid", "best_ask"]).sort_values("seq").reset_index(drop=True)
    df = df[df["best_ask"] > df["best_bid"]].copy()      # drop crossed/locked
    df["spread_ticks"] = (df["best_ask"] - df["best_bid"]) / TICK

    # empirical p(up|imbalance) on this same day (rough -- just to select signal states)
    is_move = df["mid"] != df["mid"].shift(1)
    df["seg"] = is_move.cumsum()
    moves = df[is_move].reset_index(drop=True)
    moves["next_mid"] = moves["mid"].shift(-1)
    seg_next = moves["next_mid"].copy(); seg_next.index = range(1, len(moves) + 1)
    df["nm"] = df["seg"].map(seg_next)
    df = df[df["nm"].notna()].copy()
    df["is_up"] = (df["nm"] > df["mid"]).astype(float)
    df["l1_bin"] = np.clip(np.digitize(df["imbalance"], L1_EDGES) - 1, 0, N_L1_BINS - 1)
    rate = df.groupby("l1_bin")["is_up"].transform("mean")
    df["signal_strength"] = np.abs(rate - 0.5)

    def describe(s, label):
        q = s.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        print(f"{label:>28}: mean={s.mean():.3f}  median={q[0.50]:.3f}  "
              f"p10={q[0.10]:.3f} p25={q[0.25]:.3f} p75={q[0.75]:.3f} p90={q[0.90]:.3f}")

    print(f"Spread (in ticks) distribution on {DATE}:\n")
    describe(df["spread_ticks"], "ALL states")
    describe(df.loc[df["signal_strength"] >= 0.05, "spread_ticks"], "signal fires (str>=0.05)")
    describe(df.loc[df["signal_strength"] >= 0.10, "spread_ticks"], "strong signal (str>=0.10)")

    # what fraction of the book is 1 tick wide vs wider?
    print("\nspread width composition (ALL states):")
    for w in [1, 2, 3, 4]:
        frac = np.mean(np.isclose(df["spread_ticks"], w))
        print(f"  exactly {w} tick(s): {100*frac:.1f}%")
    print(f"  > 4 ticks:        {100*np.mean(df['spread_ticks'] > 4):.1f}%")

    print("\nReading it:")
    print("  if 'signal fires' spread >> 'ALL states' spread -> signal fires in wide-spread")
    print("     states; the ~1.8-tick cost is real and the strategy genuinely can't cross it.")
    print("  if spread is ~1.8 everywhere and mostly NOT exactly 1 tick -> investigate the")
    print("     reconstruction (are best levels being dropped?) before trusting the cost.")


if __name__ == "__main__":
    main()
