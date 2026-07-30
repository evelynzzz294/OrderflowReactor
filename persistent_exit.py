"""
Signal-persistence exit -- the last untested mechanism.

Instead of a fixed N-state hold, STAY in the position as long as the entry criteria still
hold, and exit the moment the signal breaks. This tests whether a subset of trades where
imbalance + trend PERSIST can ride a bigger cumulative move that clears the ~1.8-tick spread
-- something the fixed-horizon backtest and the (plateauing) edge-decay curve can't directly show.

Entry (same strict real-time gate, no ranking, no lookahead):
  imbalance strong (p_hat beyond 0.5 +/- tau)  AND  trend confirms (sign(mid_t - mid_{t-K})).
  (Volume ignored -- shown to not help / mildly hurt.)

Exit (hold while it holds):
  exit at the first later state where EITHER
    - imbalance weakens back through 0.5 +/- tau (signal gone), OR
    - trend flips against the position,
  or at a hard max-hold cap (safety).
Exit P&L = mid_exit - mid_entry (gross). Then net = gross - 1.8-tick round-trip spread.

If the persistence subset produces large winners, mean hold will be long and gross edge will
rise well above the ~0.4 fixed-horizon ceiling. If moves don't persist (decay-curve prior),
hold stays short and net stays ~ -1.4. K=10, leave-one-day-out, non-overlapping.
"""

import os
import numpy as np
import pandas as pd

BASE = r"C:\Users\evezh\Downloads\OrderFlowReactor"
DATES = ["20260427", "20260505", "20260513", "20260521"]
N_L1_BINS = 20
K = 10
TAUS = [0.14, 0.18, 0.22]     # offsets from 0.5 -> thresholds 0.64, 0.68, 0.72
MAX_HOLD = 200
ROUND_TRIP_COST = 1.8
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


def backtest_day(df, rate, tau, K):
    p = rate[df["l1_bin"].to_numpy()]
    mid = df["mid"].to_numpy()
    n = len(df)
    long_ok = p > 0.5 + tau
    short_ok = p < 0.5 - tau
    trend = np.zeros(n, dtype=np.int8)
    if K < n:
        trend[K:] = np.sign(mid[K:] - mid[:-K]).astype(np.int8)

    trades, holds = [], []
    i = 0
    while i < n - 1:
        # entry: strong imbalance AND trend confirms
        if long_ok[i] and trend[i] == 1:
            side = 1
        elif short_ok[i] and trend[i] == -1:
            side = -1
        else:
            i += 1
            continue
        # hold while entry criteria still hold; exit when signal weakens or trend flips
        entry = i
        j = i + 1
        limit = min(i + MAX_HOLD, n - 1)
        while j < limit:
            imb_ok = long_ok[j] if side == 1 else short_ok[j]
            trend_ok = (trend[j] == side)
            if not (imb_ok and trend_ok):
                break
            j += 1
        trades.append(side * (mid[j] - mid[entry]) / TICK)
        holds.append(j - entry)
        i = j            # non-overlapping: next search starts at exit
    return np.array(trades), np.array(holds)


def main():
    days = {d: load_day(d) for d in DATES}
    rate_by_fold = {d: fit_curve([days[x] for x in DATES if x != d]) for d in DATES}

    print(f"Signal-persistence exit (hold while imbalance+trend hold, K={K}, max hold {MAX_HOLD})")
    print(f"net = gross - {ROUND_TRIP_COST} tick round-trip\n")
    print(f"{'tau':>5} {'trades/day':>11} {'gross tk/tr':>12} {'win%':>6} {'mean hold':>10} {'max hold':>9} {'net tk/tr':>10}")
    print("-" * 68)

    for tau in TAUS:
        all_t, all_h = [], []
        for d in DATES:
            t, h = backtest_day(days[d], rate_by_fold[d], tau, K)
            all_t.append(t); all_h.append(h)
        t = np.concatenate(all_t); h = np.concatenate(all_h)
        if len(t) == 0:
            print(f"{0.5+tau:>5.2f} {'0':>11}")
            continue
        per_day = len(t) / len(DATES)
        print(f"{0.5+tau:>5.2f} {per_day:>11,.0f} {t.mean():>+12.4f} {100*np.mean(t>0):>5.1f} "
              f"{h.mean():>10.1f} {int(h.max()):>9} {t.mean()-ROUND_TRIP_COST:>+10.4f}")

    print("\nReading it:")
    print("  mean hold: if the signal persists, this is LONG (positions ride the move) and gross")
    print("     edge rises above the ~0.4 fixed-horizon ceiling. If moves don't persist, hold")
    print("     stays short (a few states) and gross stays ~0.2-0.4.")
    print("  gross tk/tr: does letting winners run beat the fixed-horizon ceiling? (decay-curve")
    print("     prior says no -- cumulative edge plateaued.)")
    print("  net tk/tr: the bottom line vs the 1.8-tick spread. If still ~-1.4, the persistence")
    print("     mechanism does not rescue it either, and every taker-side lever is exhausted.")


if __name__ == "__main__":
    main()
