"""
Stage 1 backtest -- does the L1 imbalance signal have GROSS directional value over a
fixed holding horizon? (Mid-to-mid P&L, NO costs. NOT tradable edge yet -- costs come in
Stage 2. This measures gross directional value only.)

Horizons 5 and 10 states: the edge-decay analysis identified 5-10 as the candidate range
where most of the cumulative edge is realized. (This does NOT establish 5 or 10 as optimal
-- risk-adjusted comparison is what the P&L distribution below starts to show.)

Rules (locked):
  - Signal: empirical P(up | L1 imbalance), fit on 3 training days, applied to held-out day.
  - Enter long if p_hat > 0.5 + tau, short if p_hat < 0.5 - tau, else flat.
  - tau in {0.00,0.02,0.04,0.06,0.08,0.10}  (long cutoff 0.50..0.60).
  - ONE position at a time: enter at t, hold to t+N, ignore signals between, then look at t+N.
  - Leave-one-day-out; reported pooled AND per held-out day.

Primary metric: average GROSS ticks per trade. Stage 2 compares this against realized
bid/ask crossing costs + fees to compute the break-even transaction cost. Whether the
strategy survives costs is a Stage-2 question -- NOT concluded here.
"""

import os
import numpy as np
import pandas as pd

BASE = r"C:\Users\evezh\Downloads\OrderFlowReactor"
DATES = ["20260427", "20260505", "20260513", "20260521"]
N_L1_BINS = 20
TAUS = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]
HORIZONS = [5, 10]      # decay analysis identifies 5-10 states as the candidate range
TICK = 250_000_000
L1_EDGES = np.linspace(0.0, 1.0, N_L1_BINS + 1)


def load_day(date):
    path = os.path.join(BASE, f"MNQ_book_states_{date}.parquet")
    df = pd.read_parquet(path, columns=["seq", "ts_event", "mid", "imbalance"])
    df = df.sort_values("seq").reset_index(drop=True)

    # robust timestamp -> int64 ns
    if pd.api.types.is_datetime64_any_dtype(df["ts_event"]):
        df["ts_ns"] = df["ts_event"].astype("int64")
    else:
        df["ts_ns"] = df["ts_event"].astype(np.int64)

    is_move = df["mid"] != df["mid"].shift(1)
    df["seg"] = is_move.cumsum()
    moves = df[is_move].reset_index(drop=True)
    moves["next_mid"] = moves["mid"].shift(-1)
    seg_next = moves["next_mid"].copy(); seg_next.index = range(1, len(moves) + 1)
    df["nm"] = df["seg"].map(seg_next)
    df["is_up"] = np.where(df["nm"].notna(), (df["nm"] > df["mid"]).astype(float), np.nan)
    df["l1_bin"] = np.clip(np.digitize(df["imbalance"], L1_EDGES) - 1, 0, N_L1_BINS - 1)

    # defensive checks
    assert df["seq"].is_monotonic_increasing, f"{date}: seq not monotonic"
    assert df["mid"].notna().all(), f"{date}: NaN mid"
    assert df["imbalance"].between(0, 1).all(), f"{date}: imbalance out of [0,1]"
    assert df["ts_ns"].is_monotonic_increasing, f"{date}: ts not monotonic"

    # confirm TICK units: smallest nonzero mid change should equal one tick
    moves = df["mid"].diff().abs()
    smallest = moves[moves > 0].min()
    print(f"{date}: smallest nonzero mid change = {smallest:.0f} raw units "
          f"(TICK={TICK}; ratio={smallest/TICK:.3f} -- want ~1.0 or a small integer)")
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
    ts = df["ts_ns"].to_numpy()
    n = len(df)
    signal = np.where(p > 0.5 + tau, 1, np.where(p < 0.5 - tau, -1, 0))

    trades, hold_s, sides = [], [], []
    i = 0
    while i < n - N:
        s = signal[i]
        if s == 0:
            i += 1
            continue
        trades.append(s * (mid[i + N] - mid[i]) / TICK)
        hold_s.append((ts[i + N] - ts[i]) / 1e9)
        sides.append(s)
        i += N
    return np.array(trades), np.array(hold_s), np.array(sides)


def main():
    days = {d: load_day(d) for d in DATES}
    print("Stage 1: GROSS directional value (mid-to-mid, NO costs). NOT tradable edge yet.\n")

    for N in HORIZONS:
        print(f"=== Horizon N = {N} states ===")
        hdr = (f"{'long>':>6} {'trades':>8} {'long%':>6} {'win%':>6} {'flat%':>6} {'loss%':>6} "
               f"{'gross tk/tr':>12} {'tot tk':>9} {'t-stat':>7} {'hold(s)':>8} {'daily tk/tr':>28}")
        print(hdr); print("-" * len(hdr))
        for tau in TAUS:
            per_day_ticks, per_day_meanticks, all_t, all_h, all_s = [], [], [], [], []
            for test_day in DATES:
                rate = fit_curve([days[d] for d in DATES if d != test_day])
                t, h, s = backtest_day(days[test_day], rate, tau, N)
                all_t.append(t); all_h.append(h); all_s.append(s)
                per_day_ticks.append(t.sum())
                per_day_meanticks.append(t.mean() if len(t) else np.nan)
            t = np.concatenate(all_t); h = np.concatenate(all_h); s = np.concatenate(all_s)
            if len(t) == 0:
                print(f"{0.5+tau:>6.2f} {'0':>8}")
                continue

            win = np.mean(t > 0) * 100; flat = np.mean(t == 0) * 100; loss = np.mean(t < 0) * 100
            longpct = np.mean(s > 0) * 100
            tstat = (t.mean() / t.std(ddof=1) * np.sqrt(len(t))) if len(t) > 1 and t.std(ddof=1) > 0 else np.nan

            pd_str = " ".join(f"{v:+.3f}" for v in per_day_meanticks)
            print(f"{0.5+tau:>6.2f} {len(t):>8,} {longpct:>5.1f} {win:>5.1f} {flat:>5.1f} {loss:>5.1f} "
                  f"{t.mean():>+12.4f} {t.sum():>+9.1f} {tstat:>+7.2f} {h.mean():>8.2f}   {pd_str:>26}")
        print()

    print("Reading it:")
    print("  gross tk/tr = avg gross ticks per trade BEFORE costs -- THE key number.")
    print("     Stage 2 will compare this against realized bid/ask crossing costs and fees.")
    print("  daily tk/tr = per-held-out-day avg ticks/trade (4 numbers). Positive on ALL four")
    print("     = consistent gross directional value; sign-flips = fragile.")
    print("  win/flat/loss: many 'flat' at fixed horizon means unchanged mid, not wrong direction.")
    print("  t-stat = significance of per-trade P&L (NOT a Sharpe).")
    print("  (A 4-day annualized Sharpe was omitted -- too few days to be meaningful.)")


if __name__ == "__main__":
    main()
