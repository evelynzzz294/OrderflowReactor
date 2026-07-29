"""
Edge decay of the L1 imbalance signal: how long does its directional predictive value last?

For each horizon N and each state t, compute the SIGNED forward return:
    sign(p_hat(t) - 0.5) * (mid_{t+N} - mid_t)   in ticks
averaged over all states. This is "if I take the signal's directional bet and hold N
states, what's the average gross ticks?" -- measured over ALL states (no threshold, no
position management), so it's a clean signal-property curve, not a trading result.

Leave-one-day-out: the p_hat curve is fit on 3 days, decay measured on the held-out day.
Overlapping windows are fine here -- we measure a statistical property (avg forward
return), not tradable non-overlapping P&L.

Reading the curve vs N:
  rises then flattens -> signal predicts a move that persists, then no new info
  peaks early then decays -> predictive value is short-lived (next-move only)
  goes negative at large N -> mean reversion (signal's move reverses later)
This tells you the holding horizon to use in the actual backtest, instead of guessing.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = r"C:\Users\evezh\Downloads\OrderFlowReactor"
DATES = ["20260427", "20260505", "20260513", "20260521"]
N_L1_BINS = 20
HORIZONS = [1, 2, 5, 10, 20, 50, 100, 200]
TICK = 250_000_000
L1_EDGES = np.linspace(0.0, 1.0, N_L1_BINS + 1)
PLOT = os.path.join(BASE, "edge_decay.png")


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


def decay_for_day(df, rate):
    """Decay measures at each horizon, over all states AND per SIGNAL-STRENGTH group.
       sign-weighted:     sign(p-0.5) * fwd_return   (directional edge)
       strength-weighted: (2p-1)      * fwd_return   (do stronger-predicted states
                                                       contribute more directional move?)
       Strength groups by |p-0.5|: weak <0.05, medium 0.05-0.15, strong >=0.15.
    """
    p = rate[df["l1_bin"].to_numpy()]
    direction = np.sign(p - 0.5)
    strength_w = 2 * p - 1
    absd = np.abs(p - 0.5)
    grp = np.select([absd < 0.05, absd < 0.15], ["weak", "medium"], default="strong")
    mid = df["mid"].to_numpy()

    sign_out, strw_out, nwin, grp_out = {}, {}, {}, {}
    for N in HORIZONS:
        fwd = np.full(len(mid), np.nan)
        fwd[:-N] = (mid[N:] - mid[:-N]) / TICK
        valid = ~np.isnan(fwd)
        nwin[N] = int(valid.sum())
        sign_out[N] = np.nanmean(direction * fwd)
        strw_out[N] = np.nanmean(strength_w * fwd)
        sv = direction * fwd
        gr = {}
        for g in ["weak", "medium", "strong"]:
            m = valid & (grp == g)
            gr[g] = np.nanmean(sv[m]) if m.any() else np.nan
        grp_out[N] = gr
    return sign_out, strw_out, nwin, grp_out


def main():
    days = {d: load_day(d) for d in DATES}

    sign_day, strw_day, nwin_day, grp_day = {}, {}, {}, {}
    for test_day in DATES:
        rate = fit_curve([days[d] for d in DATES if d != test_day])
        s, sw, nw, gr = decay_for_day(days[test_day], rate)
        sign_day[test_day] = s; strw_day[test_day] = sw
        nwin_day[test_day] = nw; grp_day[test_day] = gr

    sign_pool = {N: np.mean([sign_day[d][N] for d in DATES]) for N in HORIZONS}   # equal-day mean
    strw_pool = {N: np.mean([strw_day[d][N] for d in DATES]) for N in HORIZONS}

    def print_table(name, per_day, pooled):
        print(f"\n{name} -- equal-day mean, by horizon and held-out day:")
        print(f"{'N':>5} " + " ".join(f"{d[-4:]:>9}" for d in DATES) + f"{'eq-day':>10}")
        print("-" * (6 + 10 * len(DATES) + 10))
        for N in HORIZONS:
            row = " ".join(f"{per_day[d][N]:>+9.4f}" for d in DATES)
            print(f"{N:>5} {row} {pooled[N]:>+10.4f}")

    print_table("SIGN-weighted (directional edge, ticks)", sign_day, sign_pool)
    print_table("STRENGTH-weighted ((2p-1)*return; do stronger-predicted states move more?)",
                strw_day, strw_pool)

    # overlap context -- NOT effective sample size, just window counts
    print("\nOverlap context (means/shape are UNBIASED; overlap only limits certainty):")
    print(f"{'N':>5} {'valid rows (pooled)':>21} {'~non-overlap windows (n/N)':>28}")
    for N in HORIZONS:
        total = sum(nwin_day[d][N] for d in DATES)
        print(f"{N:>5} {total:>21,} {total // N:>28,}")
    print("  overlapping windows are highly correlated -> do NOT read row-level t-stats or")
    print("  tiny CIs off these. The 4 per-day columns are the honest robustness check.")

    # signal-strength decay: DO STRONGER SIGNALS PERSIST LONGER?
    print("\nSIGN-weighted decay BY SIGNAL STRENGTH |p-0.5| (equal-day mean):")
    print(f"{'N':>5} {'weak':>10} {'medium':>10} {'strong':>10}")
    print("-" * 40)
    for N in HORIZONS:
        vals = [np.nanmean([grp_day[d][N][g] for d in DATES]) for g in ["weak", "medium", "strong"]]
        print(f"{N:>5} " + " ".join(f"{v:>+10.4f}" for v in vals))
    print("  -> if 'strong' stays elevated at large N while 'weak' fades, strong signals")
    print("     persist longer -> favors a high threshold + longer hold. This is the clean")
    print("     test (better than comparing the two aggregate curves).")

    print("\nReading it:")
    print("  SIGN-weighted: pure directional edge; peak N = most gross directional edge.")
    print("  STRENGTH-weighted: do stronger-predicted states contribute more move? (NOT a")
    print("     calibration test; compressed scale -- compare shape.)")
    print("  DISTINGUISH three large-N behaviors:")
    print("     rises then PLATEAUS      = move persists, no new info after (e.g. +.10 -> +.19 -> +.18)")
    print("     rises then FALLS to zero = initial move gets undone (decay / mean reversion)")
    print("     goes NEGATIVE            = reversal overshoots; signal direction now anti-correlated")
    print("  signs consistent across the 4 day-columns = robust.")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for d in DATES:
        ax[0].plot(HORIZONS, [sign_day[d][N] for N in HORIZONS], marker="o", ms=3, alpha=0.5, label=d[-4:])
        ax[1].plot(HORIZONS, [strw_day[d][N] for N in HORIZONS], marker="o", ms=3, alpha=0.5, label=d[-4:])
    ax[0].plot(HORIZONS, [sign_pool[N] for N in HORIZONS], marker="o", lw=2.5, color="black", label="eq-day")
    ax[1].plot(HORIZONS, [strw_pool[N] for N in HORIZONS], marker="o", lw=2.5, color="black", label="eq-day")
    for a, title in zip(ax, ["Sign-weighted (directional edge)", "Strength-weighted ((2p-1)*return)"]):
        a.axhline(0, color="gray", lw=1, ls="--")
        a.set_xlabel("holding horizon N (states)"); a.set_ylabel("avg return (ticks)")
        a.set_title(title); a.set_xscale("log"); a.legend(fontsize=8); a.grid(alpha=0.2)
    plt.tight_layout(); plt.savefig(PLOT, dpi=130)
    print(f"\nsaved -> {PLOT}")


if __name__ == "__main__":
    main()
