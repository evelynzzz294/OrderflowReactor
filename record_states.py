import databento as db
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from book import Book

DATA_PATH = r"C:\Users\evezh\Downloads\OrderFlowReactor\MNQ_c_0\MNQ_c_0_mbo_20260521.dbn.zst"
OUT_PATH  = r"C:\Users\evezh\Downloads\OrderFlowReactor\MNQ_book_states_20260521.parquet"
MAX_RECORDS = None   # test slice first; set to None for the full day
DEPTH = 10
FLUSH_EVERY = 1_000_000

CORE = ["ts_event", "seq", "best_bid", "best_ask", "bid_size_1", "ask_size_1", "mid", "imbalance"]
DEPTH_COLS = (
    [f"bid_px_{i}" for i in range(1, DEPTH + 1)] + [f"bid_sz_{i}" for i in range(1, DEPTH + 1)]
    + [f"ask_px_{i}" for i in range(1, DEPTH + 1)] + [f"ask_sz_{i}" for i in range(1, DEPTH + 1)]
)
COLS = CORE + DEPTH_COLS
SCHEMA = pa.schema([(c, pa.float64() if c == "imbalance" else pa.int64()) for c in COLS])


def padded(levels, n):
    prices = [p for p, s in levels[:n]] + [0] * (n - min(len(levels), n))
    sizes  = [s for p, s in levels[:n]] + [0] * (n - min(len(levels), n))
    return prices, sizes


def record_states(path, out_path, max_records=None, depth=DEPTH):
    store = db.DBNStore.from_file(path)
    book = Book(strict=False)
    writer = pq.ParquetWriter(out_path, SCHEMA)

    buffer = []
    seq = 0
    n_records = 0
    prev_tob = None

    def flush():
        nonlocal buffer
        if buffer:
            df = pd.DataFrame(buffer, columns=COLS)
            writer.write_table(pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False))
            buffer = []

    for msg in store:
        a, side = msg.action, msg.side
        if a == "A":
            book.add(msg.order_id, side, msg.price, msg.size)
        elif a == "C":
            book.cancel(msg.order_id, msg.size)
        elif a == "M":
            book.modify(msg.order_id, side, msg.price, msg.size)
        elif a == "R":
            book.clear()

        n_records += 1
        if n_records % 1_000_000 == 0:
            print(f"processed {n_records:,}, saved {seq:,}")

        if msg.flags & db.RecordFlags.F_LAST:
            bb, ba = book.best_bid(), book.best_ask()
            if bb is not None and ba is not None:
                bb_px, bb_sz = bb
                ba_px, ba_sz = ba
                cur_tob = (bb_px, bb_sz, ba_px, ba_sz)

                if cur_tob != prev_tob:                 
                    top_bids = book.top_n_bids(depth)
                    top_asks = book.top_n_asks(depth)
                    bid_px, bid_sz = padded(top_bids, depth)
                    ask_px, ask_sz = padded(top_asks, depth)
                    mid = (bb_px + ba_px) // 2
                    imb = bb_sz / (bb_sz + ba_sz)
                    buffer.append(
                        [msg.ts_event, seq, bb_px, ba_px, bb_sz, ba_sz, mid, imb]
                        + bid_px + bid_sz + ask_px + ask_sz
                    )
                    seq += 1
                    if len(buffer) >= FLUSH_EVERY:
                        flush()

                prev_tob = cur_tob

        if max_records is not None and n_records >= max_records:
            break

    flush()
    writer.close()

    print(f"\nrecords processed:      {n_records:,}")
    print(f"rows saved (TOB change): {seq:,}")
    print(f"compaction ratio:       {n_records / max(seq, 1):.1f}x")
    print(f"saved -> {out_path}")

if __name__ == "__main__":
    record_states(DATA_PATH, OUT_PATH, MAX_RECORDS)