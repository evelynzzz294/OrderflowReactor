import pandas as pd
import numpy as np

STATES_PATH = r"C:\Users\evezh\Downloads\OrderFlowReactor\MNQ_book_states_20260521.parquet"
OUT_PATH    = r"C:\Users\evezh\Downloads\OrderFlowReactor\MNQ_labeled_20260521.parquet"


def label_states(states_path, out_path):
    df = pd.read_parquet(states_path, columns=["ts_event", "seq", "mid", "imbalance"])
    df = df.sort_values("seq").reset_index(drop=True)

    # 1. Collapse to PRICE MOVES only: keep one row each time mid actually changes.
    #    This gives the ordered sequence of distinct mid prices and WHERE each began.
    is_move = df["mid"] != df["mid"].shift(1)   
    moves = df[is_move].reset_index(drop=True)  

    # 2. For each move, the "next distinct mid" is simply the next move's mid.
    moves["next_mid"] = moves["mid"].shift(-1)

    # 3. Map that answer back onto EVERY original row.
    #    Every row belongs to the move-segment it sits in; forward-fill carries
    #    each segment's next_mid down to all rows within it.
    df["seg"] = is_move.cumsum()                 
    seg_next = moves["next_mid"].copy()
    seg_next.index = range(1, len(moves) + 1)    
    df["next_mid"] = df["seg"].map(seg_next)

    # 4. Drop rows whose segment has no future move (the end-of-day tail).
    labeled = df[df["next_mid"].notna()].copy()

    # 5. Label.
    labeled["label"] = np.where(labeled["next_mid"] > labeled["mid"], 1, -1)
    labeled = labeled.drop(columns=["seg"])

    labeled.to_parquet(out_path, index=False)

    print(f"total state rows:     {len(df):,}")
    print(f"labeled rows:         {len(labeled):,}")
    print(f"dropped (no future):  {len(df) - len(labeled):,}")
    print(f"up fraction:          {(labeled['label'] == 1).mean():.3f}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    label_states(STATES_PATH, OUT_PATH)