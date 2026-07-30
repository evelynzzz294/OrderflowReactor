"""
Strict real-time entry filter -- all criteria decidable per-state as it arrives (NO ranking,
no lookahead, no top-N selection). Goal: tighten imbalance + volume + short-K trend until
only ~1000 trades/day survive, and see the edge on those strongest setups.

Per-state entry rule (all must hold):
  1. imbalance strong:   p_hat > 0.5 + tau   (long)  or  < 0.5 - tau  (short)
  2. trend confirms:     sign(signal) == sign(mid_t - mid_{t-K})     (K in {5, 10})
  3. volume thick:       total top-of-book size >= a fixed high percentile cutoff
                         (cutoff learned from TRAINING days -> usable in real time)

Every gate uses only information available AT time t, so this is a real-time-executable
rule, not a whole-day ranking.

Swept: tau (imbalance tightness), volume percentile (volume tightness), K (trend window).
Reports per config: trades/day, gross tk/tr, win%, and NET after a ~1.8-tick round-trip
spread so the cost wall stays visible.

Horizon N=5, leave-one-day-out, non-overlapping.
"""

import os
import numpy as np
import pandas as pd

BASE = r"C:\Users\evezh\Downloads\OrderFlowReactor"
DATES = ["20260427", "20260505", "20260513", "20260521"]
N_L1_BINS = 20
HORIZON = 5
TREND_KS = [5, 10]
TAUS = [0.10, 0.12, 0.14]                 # tight imbalance cutoffs
VOL_PCTS = [0.75, 0.90, 0.95]             # thick-book cutoffs (training percentile)
ROUND_TRIP_COST = 1.8                     # ticks, from Stage 2 realized spread
TICK = 250_000_000
L1_EDGES = np.linspace(0.0, 1.0, N_L1_BINS + 1)


def load_day(date):
    df = pd.read_parquet(os.path.join(BASE, f"MNQ_book_states_{date}.parquet"),
                         columns=["seq", "mid", "imbalance", "bid_size_1", "ask_size_1"])
    df = df.sort_values("seq").reset_index(drop=True)
    is_move = df["mid"] != df["mid"].shift(1)
    df["seg"] = is_move.cumsum()
    moves = df[is_move].reset_index(drop=True)
    moves["next_mid"] = moves["mid"].shift(-1)
    seg_next = moves["next_mid"].copy(); seg_next.index = range(1, len(moves) + 1)
    df["nm"] = df["seg"].map(seg_next)
    df["is_up"] = np.where(df["nm"].notna(), (df["nm"] > df["mid"]).astype(float), np.nan)
    df["l1_bin"] = np.clip(np.digitize(df["imbalance"], L1_EDGES) - 1, 0, N_L1_BINS - 1)
    df["total_vol"] = df["bid_size_1"] + df["ask_size_1"]
    return df


def fit_curve(train_days):
    t = pd.concat([d[["l1_bin", "is_up"]].dropna() for d in train_days], ignore_index=True)
    g = t.groupby("l1_bin")["is_up"].mean()
    rate = np.full(N_L1_BINS, t["is_up"].mean())
    for b, v in g.items():
        rate[b] = v
    return rate


def backtest_day(df, rate, tau, N, K, vol_cut):
    p = rate[df["l1_bin"].to_numpy()]
    mid = df["mid"].to_numpy()
    vol = df["total_vol"].to_numpy()
    n = len(df)
    signal = np.where(p > 0.5 + tau, 1, np.where(p < 0.5 - tau, -1, 0))
    trend = np.zeros(n, dtype=np.int8)
    if K < n:
        trend[K:] = np.sign(mid[K:] - mid[:-K]).astype(np.int8)

    trades = []
    i = 0
    while i < n - N:
        s = signal[i]
        if s == 0:
            i += 1; continue
        if trend[i] == 0 or np.sign(s) != trend[i]:      # trend confirmation
            i += 1; continue
        if vol[i] < vol_cut:                              # thick book
            i += 1; continue
        trades.append(s * (mid[i + N] - mid[i]) / TICK)
        i += N
    return np.array(trades)


def main():
    days = {d: load_day(d) for d in DATES}
    rate_by_fold = {d: fit_curve([days[x] for x in DATES if x != d]) for d in DATES}

    print(f"Real-time strict filter (horizon N={HORIZON}, per-state gates, no ranking)")
    print(f"net = gross - {ROUND_TRIP_COST} tick round-trip cost\n")

    for K in TREND_KS:
        print(f"===== trend K = {K} =====")
        print(f"{'tau':>5} {'volpct':>7} {'trades/day':>11} {'gross tk/tr':>12} {'win%':>6} {'net tk/tr':>10}")
        print("-" * 60)
        for tau in TAUS:
            for vp in VOL_PCTS:
                # volume cutoff from TRAINING folds (real-time usable, no test-day leak)
                vol_cut_by_fold = {d: pd.concat([days[x]["total_vol"] for x in DATES if x != d]).quantile(vp)
                                   for d in DATES}
                all_t = []
                for d in DATES:
                    all_t.append(backtest_day(days[d], rate_by_fold[d], tau, HORIZON, K, vol_cut_by_fold[d]))
                t = np.concatenate(all_t)
                per_day = len(t) / len(DATES)
                if len(t) == 0:
                    print(f"{0.5+tau:>5.2f} {vp:>7.2f} {0:>11}")
                    continue
                gross = t.mean()
                win = 100 * np.mean(t > 0)
                net = gross - ROUND_TRIP_COST
                print(f"{0.5+tau:>5.2f} {vp:>7.2f} {per_day:>11,.0f} {gross:>+12.4f} {win:>5.1f} {net:>+10.4f}")
            print()

    print("Reading it:")
    print("  trades/day: find the config near ~1000. Tighter tau / higher volpct -> fewer trades.")
    print("  gross tk/tr: the strongest setups carry the most edge -- expect it to climb as you")
    print("     tighten, toward a ceiling (strong-signal edge was ~0.24 in the strength table).")
    print("  net tk/tr: gross minus the ~1.8-tick round-trip spread. This is the honest bottom")
    print("     line -- even the strictest, highest-conviction setups are compared to real cost.")
    print("  win%: how often the strongest setups are directionally right.")
    print("  NOTE: with very few trades/day the per-config edge gets noisier -- 1000/day x 4 days")
    print("     = 4000 trades total, still a usable sample, but treat single-config extremes with care.")


if __name__ == "__main__":
    main()
