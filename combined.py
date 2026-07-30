"""
Strict combined entry filter: imbalance threshold + trend confirmation + volume.

Goal: trade quality over quantity. Stack filters and see (a) how high the gross edge goes,
(b) the marginal contribution of EACH filter, (c) how fast trade count collapses, and
(d) whether volume -- null on its own earlier -- adds anything IN COMBINATION with trend.

Filters, applied on top of a fixed imbalance threshold (tau):
  TREND : take the signal only if its direction agrees with sign(mid_t - mid_{t-K}).
  VOL   : take the signal only if total top-of-book volume is "thick" (>= its median),
          the usual high-conviction intuition. (Tested here IN COMBINATION, since the
          earlier null was volume alone.)

Four configurations compared at each threshold:
  base            : imbalance only
  +trend          : imbalance + trend confirmation
  +vol            : imbalance + volume (thick book)
  +trend+vol      : all three (strictest)

Trend look-back K is swept so you can see where confirmation helps most.
Horizon N=5. Leave-one-day-out, non-overlapping, gross ticks (no costs).

HONEST FRAME: threshold and trend are known to help; volume was null alone. Even stacked,
the edge ceiling is well under the ~1.8-tick round-trip spread, and trade counts drop fast
-- watch that the sample stays large enough to trust.
"""

import os
import numpy as np
import pandas as pd

BASE = r"C:\Users\evezh\Downloads\OrderFlowReactor"
DATES = ["20260427", "20260505", "20260513", "20260521"]
N_L1_BINS = 20
TAUS = [0.04, 0.08, 0.10]
HORIZON = 5
TREND_KS = [10, 20, 30, 50]
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


def backtest_day(df, rate, tau, N, K, vol_med, use_trend, use_vol):
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
        if use_trend and (trend[i] == 0 or np.sign(s) != trend[i]):
            i += 1; continue
        if use_vol and vol[i] < vol_med:      # require thick book
            i += 1; continue
        trades.append(s * (mid[i + N] - mid[i]) / TICK)
        i += N
    return np.array(trades)


def run(rate_by_fold, days, vol_med_by_fold, tau, N, K, use_trend, use_vol):
    all_t, per_day = [], []
    for d in DATES:
        t = backtest_day(days[d], rate_by_fold[d], tau, N, K, vol_med_by_fold[d], use_trend, use_vol)
        all_t.append(t); per_day.append(t.mean() if len(t) else np.nan)
    t = np.concatenate(all_t)
    return t, per_day


def main():
    days = {d: load_day(d) for d in DATES}
    rate_by_fold = {d: fit_curve([days[x] for x in DATES if x != d]) for d in DATES}
    # volume median from TRAINING folds (avoid using test day's own median)
    vol_med_by_fold = {d: pd.concat([days[x]["total_vol"] for x in DATES if x != d]).median()
                       for d in DATES}

    print(f"Strict combined filter (horizon N={HORIZON}, gross ticks, no costs)")
    print("configs: base | +trend | +vol(thick) | +trend+vol\n")

    for K in TREND_KS:
        print(f"===== trend K = {K} =====")
        print(f"{'thr':>5} {'config':>12} {'trades':>10} {'gross tk/tr':>12} {'win%':>6}")
        print("-" * 52)
        for tau in TAUS:
            configs = [
                ("base",        False, False),
                ("+trend",      True,  False),
                ("+vol",        False, True),
                ("+trend+vol",  True,  True),
            ]
            for name, ut, uv in configs:
                t, pd_ = run(rate_by_fold, days, vol_med_by_fold, tau, HORIZON, K, ut, uv)
                if len(t) == 0:
                    print(f"{0.5+tau:>5.2f} {name:>12} {'0':>10}")
                    continue
                win = 100 * np.mean(t > 0)
                print(f"{0.5+tau:>5.2f} {name:>12} {len(t):>10,} {t.mean():>+12.4f} {win:>5.1f}")
            print()

    print("Reading it:")
    print("  Compare within each (K, threshold) block:")
    print("    +trend vs base   -> trend's marginal value (known to help, biggest at small K)")
    print("    +vol vs base     -> volume's marginal value ALONE (earlier: null)")
    print("    +trend+vol vs +trend -> does volume add ANYTHING on top of trend? (the real")
    print("       test of whether volume earns its place in the combination)")
    print("  Watch the trade count: stacking filters shrinks it fast. Below ~100k the edge")
    print("     estimate gets noisy -- a big number on few trades is not trustworthy.")
    print("  And keep the wall in view: even the best stacked edge is compared against a")
    print("     ~1.8-tick round-trip cost. A higher gross edge is a real finding; clearing")
    print("     1.8 ticks by filtering is not on the table.")


if __name__ == "__main__":
    main()
