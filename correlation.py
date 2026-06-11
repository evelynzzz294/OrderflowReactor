import pandas as pd
import numpy as np
STATES_PATH = r"C:\Users\evezh\Downloads\OrderFlowReactor\MNQ_book_states_20260521.parquet"


def queue_correlation(states_path):
    df = pd.read_parquet(states_path, columns=["seq", "bid_size_1", "ask_size_1"])
    df = df.sort_values("seq").reset_index(drop=True)

    # increments: change in best-bid and best-ask size from one state to the next
    d_qb = df["bid_size_1"].diff()
    d_qa = df["ask_size_1"].diff()

    # keep only rows where SOMETHING changed (drop the first NaN and no-op rows)
    mask = d_qb.notna() & d_qa.notna() & ((d_qb != 0) | (d_qa != 0))
    d_qb, d_qa = d_qb[mask], d_qa[mask]

    rho = np.corrcoef(d_qb, d_qa)[0, 1]

    print(f"usable increments:        {len(d_qb):,}")
    print(f"corr(d_qb, d_qa):         {rho:+.4f}")
    print(f"mean d_qb, d_qa:          {d_qb.mean():+.3f}, {d_qa.mean():+.3f}")
    return rho

if __name__ == "__main__":
    queue_correlation(STATES_PATH)