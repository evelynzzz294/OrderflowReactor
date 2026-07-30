import databento as db
from collections import defaultdict
from book import Book

DATA_PATH = r"C:\Users\evezh\Downloads\OrderFlowReactor\MNQ_c_0\MNQ_c_0_mbo_20260521.dbn.zst"
MAX_RECORDS = None     # slice for the first run; set to None for the full day
VALIDATE_EVERY = 50_000    # integrity check cadence
TICK = 250_000_000         # 0.25 index points in raw 1e-9 units


def run_diagnostic(path, max_records=None, validate_every=50_000):
    store = db.DBNStore.from_file(path)
    book = Book(strict=False)              # let it complete; counters surface anomalies

    action_counts = defaultdict(int)
    n_records = 0
    n_events = 0                           # completed events (F_LAST seen)
    samples = []

    for msg in store:
        a, side = msg.action, msg.side
        action_counts[a] += 1

        if a == "A":
            book.add(msg.order_id, side, msg.price, msg.size)
        elif a == "C":
            book.cancel(msg.order_id, msg.size)
        elif a == "M":
            book.modify(msg.order_id, side, msg.price, msg.size)
        elif a == "R":
            book.clear()
        # T, F, N: don't touch the book

        if msg.flags & db.RecordFlags.F_LAST:
            n_events += 1
            if len(samples) < 10 and n_events % 2000 == 0:
                samples.append((book.best_bid(), book.best_ask()))

        n_records += 1
        if n_records % validate_every == 0:
            try:
                book.validate()
            except AssertionError:
                print(f"VALIDATE FAILED at record {n_records:,}")
                raise

        if max_records is not None and n_records >= max_records:
            break

    book.validate()  # final check

    print(f"\nrecords processed:        {n_records:,}")
    print(f"completed events (F_LAST): {n_events:,}")
    print("\naction counts:")
    for a in sorted(action_counts):
        print(f"  {a}: {action_counts[a]:,}")
    print("\nanomaly counters:", dict(book.stats))
    print("\nsample BBOs (raw int price x size):")
    for bb, ba in samples:
        bp = bb[0] / 1e9 if bb else None
        ap = ba[0] / 1e9 if ba else None
        print(f"  bid {bb}  (~{bp})   ask {ba}  (~{ap})")

    bb, ba = book.best_bid(), book.best_ask()
    if bb and ba:
        print(f"\nfinal spread: {(ba[0] - bb[0]) / TICK:.1f} ticks")
    return book


if __name__ == "__main__":
    run_diagnostic(DATA_PATH, MAX_RECORDS, VALIDATE_EVERY)