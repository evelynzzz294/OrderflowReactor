"""
Robustness check for the depth finding: run the levels-2-10 information test on all
four sessions and compare the mean spread per day.

The single-day result was: within fixed L1-imbalance bins, states with HIGHER deep
(levels 2-10) bid imbalance go up LESS often -- a clean, contrarian ~ -0.04 spread.
This checks whether that sign and size hold across days.

Regenerates any missing state files first (keeps them this time -- batch.py deleted them).
"""

import os, glob
import pandas as pd
import numpy as np
import databento as db
import pyarrow as pa, pyarrow.parquet as pq
from book import Book

BASE = r"C:\Users\evezh\Downloads\OrderFlowReactor"
RAW_DIR = os.path.join(BASE, "MNQ_c_0")
DATES = ["20260427", "20260505", "20260513", "20260521"]
DEPTH = 10
N_BINS = 20
FLUSH_EVERY = 1_000_000
DEPTH_LEVELS = range(2, 11)

CORE = ["ts_event", "seq", "best_bid", "best_ask", "bid_size_1", "ask_size_1", "mid", "imbalance"]
DCOLS = ([f"bid_px_{i}" for i in range(1, DEPTH+1)] + [f"bid_sz_{i}" for i in range(1, DEPTH+1)]
         + [f"ask_px_{i}" for i in range(1, DEPTH+1)] + [f"ask_sz_{i}" for i in range(1, DEPTH+1)])
COLS = CORE + DCOLS
SCHEMA = pa.schema([(c, pa.float64() if c == "imbalance" else pa.int64()) for c in COLS])


def padded(levels, n):
    pr = [p for p, s in levels[:n]] + [0]*(n - min(len(levels), n))
    sz = [s for p, s in levels[:n]] + [0]*(n - min(len(levels), n))
    return pr, sz


def build_states(raw_path, states_path):
    store = db.DBNStore.from_file(raw_path)
    book = Book(strict=False)
    writer = pq.ParquetWriter(states_path, SCHEMA)
    buf, seq, prev = [], 0, None

    def flush():
        nonlocal buf
        if buf:
            writer.write_table(pa.Table.from_pandas(pd.DataFrame(buf, columns=COLS),
                                                    schema=SCHEMA, preserve_index=False))
            buf = []

    for msg in store:
        a, side = msg.action, msg.side
        if a == "A":   book.add(msg.order_id, side, msg.price, msg.size)
        elif a == "C": book.cancel(msg.order_id, msg.size)
        elif a == "M": book.modify(msg.order_id, side, msg.price, msg.size)
        elif a == "R": book.clear()
        if msg.flags & db.RecordFlags.F_LAST:
            bb, ba = book.best_bid(), book.best_ask()
            if bb and ba:
                cur = (bb[0], bb[1], ba[0], ba[1])
                if cur != prev:
                    tb, ta = book.top_n_bids(DEPTH), book.top_n_asks(DEPTH)
                    bpx, bsz = padded(tb, DEPTH); apx, asz = padded(ta, DEPTH)
                    buf.append([msg.ts_event, seq, bb[0], ba[0], bb[1], ba[1],
                                (bb[0]+ba[0])//2, bb[1]/(bb[1]+ba[1])] + bpx+bsz+apx+asz)
                    seq += 1
                    if len(buf) >= FLUSH_EVERY: flush()
                prev = cur
    flush(); writer.close()


def depth_spread(states_path):
    """Return per-bin spreads and per-bin L1 gaps for one day."""
    cols = ["seq", "mid", "imbalance"] + \
           [f"bid_sz_{i}" for i in DEPTH_LEVELS] + [f"ask_sz_{i}" for i in DEPTH_LEVELS]
    df = pd.read_parquet(states_path, columns=cols).sort_values("seq").reset_index(drop=True)

    # first-passage label
    is_move = df["mid"] != df["mid"].shift(1)
    moves = df[is_move].reset_index(drop=True)
    moves["next_mid"] = moves["mid"].shift(-1)
    df["seg"] = is_move.cumsum()
    seg_next = moves["next_mid"].copy(); seg_next.index = range(1, len(moves)+1)
    df["next_mid"] = df["seg"].map(seg_next)
    df = df[df["next_mid"].notna()].copy()
    df["is_up"] = (df["next_mid"] > df["mid"]).astype(int)

    bid_deep = df[[f"bid_sz_{i}" for i in DEPTH_LEVELS]].sum(axis=1)
    ask_deep = df[[f"ask_sz_{i}" for i in DEPTH_LEVELS]].sum(axis=1)
    tot = bid_deep + ask_deep
    df["deep_imb"] = np.where(tot > 0, bid_deep / tot, 0.5)

    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    df["bin"] = pd.cut(df["imbalance"], bins=edges, include_lowest=True)

    spreads, l1gaps = [], []
    for _, g in df.groupby("bin", observed=True):
        if len(g) < 5000:
            continue
        med = g["deep_imb"].median()
        hi = g[g["deep_imb"] > med]; lo = g[g["deep_imb"] <= med]
        if len(hi) < 1000 or len(lo) < 1000:
            continue
        spreads.append(hi["is_up"].mean() - lo["is_up"].mean())
        l1gaps.append(abs(hi["imbalance"].mean() - lo["imbalance"].mean()))
    return np.array(spreads), np.array(l1gaps)


def main():
    results = {}
    for d in DATES:
        states_path = os.path.join(BASE, f"MNQ_book_states_{d}.parquet")
        if not os.path.exists(states_path):
            raw = os.path.join(RAW_DIR, f"MNQ_c_0_mbo_{d}.dbn.zst")
            if not os.path.exists(raw):
                print(f"{d}: raw file missing, skipping"); continue
            print(f"{d}: regenerating state file...")
            build_states(raw, states_path)
        spreads, l1gaps = depth_spread(states_path)
        results[d] = (spreads.mean(), spreads.min(), spreads.max(),
                      bool((spreads < 0).all()), l1gaps.max())
        print(f"{d}: done")

    print(f"\n{'date':>10} {'mean spread':>12} {'min':>8} {'max':>8} "
          f"{'all neg?':>9} {'max L1 gap':>11}")
    print("-" * 62)
    for d, (mean, lo, hi, allneg, l1) in results.items():
        print(f"{d:>10} {mean:>+12.4f} {lo:>+8.4f} {hi:>+8.4f} {str(allneg):>9} {l1:>11.4f}")

    if results:
        means = np.array([v[0] for v in results.values()])
        print("-" * 62)
        print(f"across-day mean of per-day mean spreads: {means.mean():+.4f}")
        print(f"per-day means all same sign? {bool((means < 0).all() or (means > 0).all())}")
        print("\nIf all four days show a consistent negative spread with tiny L1 gaps,")
        print("the contrarian depth signal is stable -- not a one-session artifact.")


if __name__ == "__main__":
    main()
