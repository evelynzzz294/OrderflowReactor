import os
import pandas as pd
import numpy as np
from book import Book
import databento as db
import pyarrow as pa, pyarrow.parquet as pq

BASE = r"C:\Users\evezh\Downloads\OrderFlowReactor"
RAW_DIR = os.path.join(BASE, "MNQ_c_0")
DEPTH = 10
FLUSH_EVERY = 1_000_000

CORE = ["ts_event", "seq", "best_bid", "best_ask", "bid_size_1", "ask_size_1", "mid", "imbalance"]
DCOLS = ([f"bid_px_{i}" for i in range(1, DEPTH+1)] + [f"bid_sz_{i}" for i in range(1, DEPTH+1)]
         + [f"ask_px_{i}" for i in range(1, DEPTH+1)] + [f"ask_sz_{i}" for i in range(1, DEPTH+1)])
COLS = CORE + DCOLS
SCHEMA = pa.schema([(c, pa.float64() if c == "imbalance" else pa.int64()) for c in COLS])


def padded(levels, n):
    pr = [p for p, s in levels[:n]] + [0]*(n - min(len(levels), n))
    sz = [s for p, s in levels[:n]] + [0]*(n - min(len(levels), n))
    return pr, sz


def record(raw_path, states_path):
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
    return seq


def label(states_path, labeled_path):
    df = pd.read_parquet(states_path, columns=["ts_event", "seq", "mid", "imbalance"])
    df = df.sort_values("seq").reset_index(drop=True)
    is_move = df["mid"] != df["mid"].shift(1)
    moves = df[is_move].reset_index(drop=True)
    moves["next_mid"] = moves["mid"].shift(-1)
    df["seg"] = is_move.cumsum()
    seg_next = moves["next_mid"].copy(); seg_next.index = range(1, len(moves)+1)
    df["next_mid"] = df["seg"].map(seg_next)
    lab = df[df["next_mid"].notna()].copy()
    lab["label"] = np.where(lab["next_mid"] > lab["mid"], 1, -1)
    lab.drop(columns=["seg"]).to_parquet(labeled_path, index=False)
    return len(lab), (lab["label"] == 1).mean()


def run_days(dates):
    for d in dates:
        raw = os.path.join(RAW_DIR, f"MNQ_c_0_mbo_{d}.dbn.zst")
        if not os.path.exists(raw):
            print(f"{d}: MISSING {raw}"); continue
        states = os.path.join(BASE, f"states_{d}.parquet")
        labeled = os.path.join(BASE, f"MNQ_labeled_{d}.parquet")
        n = record(raw, states)
        m, up = label(states, labeled)
        os.remove(states)   # free disk; keep only labeled
        print(f"{d}:  {n:,} states  ->  {m:,} labeled,  up={up:.3f}")


if __name__ == "__main__":
    SAMPLE = ["20260427", "20260505", "20260513", "20260521"]   # 4 spread days first
    run_days(SAMPLE)