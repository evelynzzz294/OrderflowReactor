"""
Trend-filtered backtest -- does avoiding against-trend trades improve the imbalance signal?

Hypothesis (yours): the imbalance signal sometimes says LONG while price is trending down
(or SHORT while trending up). Filtering to only take the signal when it AGREES with the
recent trend might remove losing trades and raise the gross edge.

The key question this answers: does trend add INDEPENDENT information, or is it redundant
with imbalance (which may already reflect recent direction)? If filtering raises gross
ticks/trade, trend helps. If it just cuts trades without improving per-trade edge, trend
is redundant -- same pattern as the depth/volume tests.

Trend = sign(mid_t - mid_{t-K}): is the mid higher or lower than K book-updates ago.
Tested at K = 20, 50, 100. Horizon fixed at N=5 (from the edge-decay result).

Compares, per threshold:
  UNFILTERED : take every imbalance signal (baseline, = backtest_stage1)
  FILTERED   : take the signal only when its direction matches the trend sign

Same discipline as everything else: leave-one-day-out, non-overlapping trades, gross
mid-to-mid ticks (no costs here -- costs are the same ~1.8-tick wall regardless, and the
question here is whether the trend filter improves the GROSS edge first).
"""

import os
import numpy as np
import pandas as pd

BASE = r"C:\Users\evezh\Downloads\OrderFlowReactor"
DATES = ["20260427", "20260505", "20260513", "20260521"]
N_L1_BINS = 20
TAUS = [0.00, 0.04, 0.08]           # a few thresholds; keep output compact
HORIZON = 5                          # from edge-decay: edge peaks ~5 states
TREND_KS = [20, 50, 100]             # look-back windows for the trend
TICK = 250_000_000
L1_EDGES = np.linspace(0.0, 1.0, N_L1_BINS + 1)


def load_day(date):
    df = pd.read_parquet(os.path.join(BASE, f"MNQ_book_states_{date}.parquet"),
                         columns=["seq", "mid", "imbalance"]).sort_values("seq").reset_index(drop=True)
    is_move = df["mid"] != df["mid"].shift(1)
    df["seg"] = is_move.cumsum()
    moves = df[is_move].reset_index(drop=True)
    moves["next_mid"] = moves["mid"].shift(-1)
    seg_next = moves["next_mid"].copy(); seg_next.index = range(1, len(moves) + 1)
    df["nm"] = df["seg"].map(seg_next)
    df["is_up"] = np.where(df["nm"].notna(), (df["nm"] > df["mid"]).astype(float), np.nan)
    df["l1_bin"] = np.clip(np.digitize(df["imbalance"], L1_EDGES) - 1, 0, N_L1_BINS - 1)
    return df


def fit_curve(train_days):
    t = pd.concat([d[["l1_bin", "is_up"]].dropna() for d in train_days], ignore_index=True)
    g = t.groupby("l1_bin")["is_up"].mean()
    rate = np.full(N_L1_BINS, t["is_up"].mean())
    for b, v in g.items():
        rate[b] = v
    return rate


def backtest_day(df, rate, tau, N, K, use_filter):
    p = rate[df["l1_bin"].to_numpy()]
    mid = df["mid"].to_numpy()
    n = len(df)
    signal = np.where(p > 0.5 + tau, 1, np.where(p < 0.5 - tau, -1, 0))

    # trend at t = sign(mid_t - mid_{t-K}); undefined (0) for the first K states
    trend = np.zeros(n, dtype=np.int8)
    if K < n:
        trend[K:] = np.sign(mid[K:] - mid[:-K]).astype(np.int8)

    trades = []
    i = 0
    while i < n - N:
        s = signal[i]
        if s == 0:
            i += 1
            continue
        if use_filter:
            # only take the signal if its direction agrees with the trend (skip if against or flat)
            if trend[i] == 0 or np.sign(s) != trend[i]:
                i += 1
                continue
        trades.append(s * (mid[i + N] - mid[i]) / TICK)
        i += N
    return np.array(trades)


def run(rate_by_fold, days, tau, N, K, use_filter):
    all_t, per_day = [], []
    for test_day in DATES:
        t = backtest_day(days[test_day], rate_by_fold[test_day], tau, N, K, use_filter)
        all_t.append(t)
        per_day.append(t.mean() if len(t) else np.nan)
    t = np.concatenate(all_t)
    return t, per_day


def main():
    days = {d: load_day(d) for d in DATES}
    rate_by_fold = {d: fit_curve([days[x] for x in DATES if x != d]) for d in DATES}

    print(f"Trend filter vs unfiltered imbalance signal (horizon N={HORIZON}, gross ticks, no costs)\n")

    for K in TREND_KS:
        print(f"===== trend look-back K = {K} states =====")
        print(f"{'thr':>5} {'':>3} {'trades':>9} {'gross tk/tr':>12} {'win%':>6} {'daily tk/tr (4 folds)':>26}")
        print("-" * 70)
        for tau in TAUS:
            # unfiltered baseline
            tu, pdu = run(rate_by_fold, days, tau, HORIZON, K, use_filter=False)
            # trend-filtered
            tf, pdf = run(rate_by_fold, days, tau, HORIZON, K, use_filter=True)

            for label, t, pd_ in [("base", tu, pdu), ("filt", tf, pdf)]:
                if len(t) == 0:
                    print(f"{0.5+tau:>5.2f} {label:>3} {'0':>9}")
                    continue
                win = 100 * np.mean(t > 0)
                pd_str = " ".join(f"{v:+.3f}" for v in pd_)
                print(f"{0.5+tau:>5.2f} {label:>3} {len(t):>9,} {t.mean():>+12.4f} {win:>5.1f}   {pd_str:>24}")
            print()

    print("Reading it:")
    print("  compare 'filt' vs 'base' at each threshold:")
    print("    filt gross tk/tr > base -> trend filter improves the gross edge (your intuition")
    print("       holds: it removes against-trend losers). Note how many trades it costs.")
    print("    filt ~ base (just fewer trades) -> trend is redundant with imbalance; filtering")
    print("       throws away trades without improving per-trade edge (like depth/volume).")
    print("    filt < base -> trend filter HURTS (imbalance already anticipates reversals the")
    print("       trend would wrongly veto).")
    print("  even if filt improves the gross edge, remember the ~1.8-tick spread wall still")
    print("  applies -- a better gross edge is a real finding but may still not clear costs.")


if __name__ == "__main__":
    main()
