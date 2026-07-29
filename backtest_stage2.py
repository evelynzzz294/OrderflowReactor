"""
Stage 2 backtest -- does the gross edge survive realistic execution cost?

Same trades as Stage 1 (N=5 and N=10, threshold sweep, non-overlapping, leave-one-day-out).
Now models the TAKER cost: to enter you cross the spread (buy at ask / sell at bid), and to
exit you cross it again. Cost is the REALIZED half-spread at t and at t+N, computed from the
actual best_bid/best_ask in the book -- not a hard-coded tick.

Spread only (no commission): the spread is the dominant execution cost; commission (~0.01
ticks) can be added later and barely changes the conclusion.

Per threshold (for N=5 and N=10):
  trades | gross ticks/trade | exec cost/trade | net ticks/trade
Break-even round-trip cost = gross ticks/trade (cost above this -> unprofitable).

This is the TAKER regime (act on the signal by crossing). A maker (posting passively) could
earn rather than pay the spread but faces adverse selection -- a separate, harder analysis
noted as future work, not claimed here.
"""

import os
import numpy as np
import pandas as pd

BASE = r"C:\Users\evezh\Downloads\OrderFlowReactor"
DATES = ["20260427", "20260505", "20260513", "20260521"]
N_L1_BINS = 20
TAUS = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]
HORIZONS = [5, 10]
TICK = 250_000_000            # full exchange tick; P&L and costs both in full ticks
L1_EDGES = np.linspace(0.0, 1.0, N_L1_BINS + 1)


def load_day(date):
    path = os.path.join(BASE, f"MNQ_book_states_{date}.parquet")
    df = pd.read_parquet(path, columns=["seq", "mid", "imbalance", "best_bid", "best_ask"])
    df = df.sort_values("seq").reset_index(drop=True)
    is_move = df["mid"] != df["mid"].shift(1)
    df["seg"] = is_move.cumsum()
    moves = df[is_move].reset_index(drop=True)
    moves["next_mid"] = moves["mid"].shift(-1)
    seg_next = moves["next_mid"].copy(); seg_next.index = range(1, len(moves) + 1)
    df["nm"] = df["seg"].map(seg_next)
    df["is_up"] = np.where(df["nm"].notna(), (df["nm"] > df["mid"]).astype(float), np.nan)
    df["l1_bin"] = np.clip(np.digitize(df["imbalance"], L1_EDGES) - 1, 0, N_L1_BINS - 1)
    df["half_spread"] = (df["best_ask"] - df["best_bid"]) / 2.0    # cost to cross one side

    # crossed (ask<bid) or locked (ask==bid) states have zero/negative spread -- transient
    # reconstruction artifacts. Count them, then drop them so cost is never understated.
    bad = df["best_ask"] <= df["best_bid"]
    n_bad = int(bad.sum())
    if n_bad:
        print(f"  {date}: {n_bad:,} crossed/locked states ({100*n_bad/len(df):.4f}%) -- dropped")
    df = df[~bad].reset_index(drop=True)
    return df


def fit_curve(train_days):
    t = pd.concat([d[["l1_bin", "is_up"]].dropna() for d in train_days], ignore_index=True)
    g = t.groupby("l1_bin")["is_up"].mean()
    rate = np.full(N_L1_BINS, t["is_up"].mean())
    for b, v in g.items():
        rate[b] = v
    return rate


def backtest_day(df, rate, tau, N):
    p = rate[df["l1_bin"].to_numpy()]
    mid = df["mid"].to_numpy()
    hs = df["half_spread"].to_numpy()
    n = len(df)
    signal = np.where(p > 0.5 + tau, 1, np.where(p < 0.5 - tau, -1, 0))

    gross, cost = [], []
    i = 0
    while i < n - N:
        s = signal[i]
        if s == 0:
            i += 1
            continue
        gross.append(s * (mid[i + N] - mid[i]) / TICK)
        # taker: cross half-spread on entry (t) and on exit (t+N)
        cost.append((hs[i] + hs[i + N]) / TICK)
        i += N
    return np.array(gross), np.array(cost)


def main():
    days = {d: load_day(d) for d in DATES}
    print("Stage 2: NET of taker spread cost (cross the spread on entry and exit).\n")

    for N in HORIZONS:
        print(f"=== Horizon N = {N} states ===")
        hdr = (f"{'long>':>6} {'trades':>10} {'gross tk/tr':>12} {'cost tk/tr':>11} "
               f"{'net tk/tr':>10} {'net total':>12} {'break-even':>11}")
        print(hdr); print("-" * len(hdr))
        for tau in TAUS:
            g_all, c_all, per_day_net = [], [], []
            for test_day in DATES:
                rate = fit_curve([days[d] for d in DATES if d != test_day])
                g, c = backtest_day(days[test_day], rate, tau, N)
                g_all.append(g); c_all.append(c)
                per_day_net.append((g - c).mean() if len(g) else np.nan)
            g = np.concatenate(g_all); c = np.concatenate(c_all)
            if len(g) == 0:
                print(f"{0.5+tau:>6.2f} {'0':>10}")
                continue
            net = g - c
            print(f"{0.5+tau:>6.2f} {len(g):>10,} {g.mean():>+12.4f} {c.mean():>11.4f} "
                  f"{net.mean():>+10.4f} {net.sum():>+12.1f} {g.mean():>11.4f}")
        print()

    print("Reading it:")
    print("  gross tk/tr  = mid-to-mid directional edge (from Stage 1).")
    print("  cost tk/tr   = realized round-trip spread crossed (half-spread at entry + exit).")
    print("  net tk/tr    = gross - cost. NEGATIVE = loses money after crossing the spread.")
    print("  break-even   = gross tk/tr: the strategy is profitable only if round-trip")
    print("                 execution cost stays BELOW this. Compare to the ~1-tick spread.")
    print("  This is the TAKER regime. A maker could earn the spread but faces adverse")
    print("  selection -- separate analysis, not claimed here.")


if __name__ == "__main__":
    main()
