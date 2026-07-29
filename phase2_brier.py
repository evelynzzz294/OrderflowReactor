"""
Phase 2 -- does depth improve OUT-OF-SAMPLE probability forecasts beyond L1 imbalance?

Design (leave-one-day-out, 4 folds):
  Model A: P(up) from 20 L1-imbalance bins (training up-rate per bin).
  Model B: P(up) from 20 L1 bins x depth-imbalance bins, with count-weighted
           shrinkage of each joint bucket toward its L1-only rate.
  Fit on 3 days, score the held-out day. Rotate. Primary metric: Brier score.

Compares empirical L1 vs empirical L1+depth. A positive result means depth improves
held-out forecasts BEYOND an empirical L1 model -- it does NOT by itself beat the
arctan PDE curve (that is a later overlay plot). Honest claim:
  "Adding depth improves out-of-sample probability forecasts beyond L1 imbalance alone,
   suggesting deeper liquidity captures information missing from the L1 framework."

Estimand: per RECORDED STATE (row-weighted). Long price segments carry more weight.

Implementation: fully vectorized prediction (20 x nd lookup arrays, no row-wise apply)
with slim training frames. Depth bin EDGES are learned from TRAINING ONLY and applied
unchanged to the held-out day (no test-day leakage). Not "reduced to counts" -- each
fold still concatenates the 3 training days to compute quantiles/groupbys, so expect
minutes, not seconds.
"""

import os
import pandas as pd
import numpy as np

BASE = r"C:\Users\evezh\Downloads\OrderFlowReactor"
DATES = ["20260427", "20260505", "20260513", "20260521"]
N_L1_BINS = 20
N_DEPTH_BINS = 5
SHRINK_LAMBDA = 500
MIN_JOINT = 50
DEPTH_LEVELS = range(2, 11)
OUT_PRED = os.path.join(BASE, "phase2_predictions.parquet")
L1_EDGES = np.linspace(0.0, 1.0, N_L1_BINS + 1)


def load_day(date):
    path = os.path.join(BASE, f"MNQ_book_states_{date}.parquet")
    cols = ["seq", "mid", "imbalance"] + \
           [f"bid_sz_{i}" for i in DEPTH_LEVELS] + [f"ask_sz_{i}" for i in DEPTH_LEVELS]
    df = pd.read_parquet(path, columns=cols).sort_values("seq").reset_index(drop=True)

    is_move = df["mid"] != df["mid"].shift(1)
    df["seg_id"] = is_move.cumsum()
    moves = df[is_move].reset_index(drop=True)
    moves["next_mid"] = moves["mid"].shift(-1)
    seg_next = moves["next_mid"].copy(); seg_next.index = range(1, len(moves) + 1)
    df["next_mid"] = df["seg_id"].map(seg_next)
    df = df[df["next_mid"].notna()].copy()
    df["is_up"] = (df["next_mid"] > df["mid"]).astype(np.int8)

    bid_deep = df[[f"bid_sz_{i}" for i in DEPTH_LEVELS]].sum(axis=1)
    ask_deep = df[[f"ask_sz_{i}" for i in DEPTH_LEVELS]].sum(axis=1)
    tot = bid_deep + ask_deep
    df["deep_imb"] = np.where(tot > 0, bid_deep / tot, 0.5)

    df["l1_bin"] = np.clip(np.digitize(df["imbalance"], L1_EDGES) - 1, 0, N_L1_BINS - 1).astype(np.int16)
    df["date"] = date
    return df[["date", "seq", "seg_id", "imbalance", "deep_imb", "is_up", "l1_bin"]]


def fit_from_days(train_days):
    train = pd.concat(train_days, ignore_index=True)

    # depth bin EDGES from TRAINING ONLY, then applied unchanged to test (no leakage)
    _, depth_edges = pd.qcut(train["deep_imb"], q=N_DEPTH_BINS, retbins=True, duplicates="drop")
    depth_edges = np.unique(depth_edges)
    depth_edges[0], depth_edges[-1] = -np.inf, np.inf
    nd = len(depth_edges) - 1

    train["d_bin"] = np.clip(np.digitize(train["deep_imb"], depth_edges) - 1, 0, nd - 1).astype(np.int16)

    global_rate = float(train["is_up"].mean())

    l1_grp = train.groupby("l1_bin")["is_up"].agg(["sum", "count"])
    l1_rate = np.full(N_L1_BINS, global_rate)
    for lb, r in l1_grp.iterrows():
        l1_rate[lb] = r["sum"] / r["count"]

    joint = train.groupby(["l1_bin", "d_bin"])["is_up"].agg(["sum", "count"])
    joint_pred = np.full((N_L1_BINS, nd), np.nan)
    joint_dense = np.zeros((N_L1_BINS, nd), dtype=bool)
    for (lb, dbn), r in joint.iterrows():
        base = l1_rate[lb]; n = r["count"]
        if n < MIN_JOINT:
            joint_pred[lb, dbn] = base
        else:
            joint_pred[lb, dbn] = (r["sum"] + SHRINK_LAMBDA * base) / (n + SHRINK_LAMBDA)
            joint_dense[lb, dbn] = True
    return l1_rate, global_rate, depth_edges, nd, joint_pred, joint_dense


def predict(test, l1_rate, depth_edges, nd, joint_pred, joint_dense):
    test = test.copy()
    test["d_bin"] = np.clip(np.digitize(test["deep_imb"], depth_edges) - 1, 0, nd - 1)
    lb = test["l1_bin"].to_numpy()
    db = test["d_bin"].to_numpy()
    pred_L1 = l1_rate[lb]
    dense = joint_dense[lb, db]
    pred_B = np.where(dense, joint_pred[lb, db], pred_L1)
    test["pred_L1"] = pred_L1
    test["pred_L1_depth"] = pred_B
    test["used_joint"] = dense
    return test


def brier(p, y): return float(np.mean((p - y) ** 2))
def logloss(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    days = {d: load_day(d) for d in DATES}

    print(f"{'test day':>10} {'tr rows':>10} {'tr segs':>9} {'te rows':>10} {'te segs':>9} "
          f"{'BrierA':>8} {'BrierB':>8} {'delta':>10} {'rel%':>6} {'joint%':>7}")
    print("-" * 104)

    deltas, all_preds = [], []
    for test_day in DATES:
        train_days = [days[d] for d in DATES if d != test_day]
        l1_rate, global_rate, depth_edges, nd, jp, jd = fit_from_days(train_days)
        pred = predict(days[test_day], l1_rate, depth_edges, nd, jp, jd)

        y = pred["is_up"].to_numpy()
        bA = brier(pred["pred_L1"].to_numpy(), y)
        bB = brier(pred["pred_L1_depth"].to_numpy(), y)
        delta = bA - bB
        deltas.append(delta)

        n_tr = sum(len(d) for d in train_days)
        n_tr_seg = sum(d["seg_id"].nunique() for d in train_days)   # seg_id restarts per day
        n_te_seg = pred["seg_id"].nunique()

        print(f"{test_day:>10} {n_tr:>10,} {n_tr_seg:>9,} {len(pred):>10,} {n_te_seg:>9,} "
              f"{bA:>8.5f} {bB:>8.5f} {delta:>+10.6f} {100*delta/bA:>+5.2f} "
              f"{100*pred['used_joint'].mean():>6.1f}")
        all_preds.append(pred[["date", "seq", "imbalance", "deep_imb", "is_up",
                               "pred_L1", "pred_L1_depth", "used_joint"]])

    deltas = np.array(deltas)
    print("-" * 104)
    print(f"mean delta across folds: {deltas.mean():+.6f}")
    print(f"positive delta on all four folds? {bool((deltas > 0).all())}")

    print("\nlog-loss check (secondary):")
    for p in all_preds:
        y = p["is_up"].to_numpy()
        llA = logloss(p["pred_L1"].to_numpy(), y)
        llB = logloss(p["pred_L1_depth"].to_numpy(), y)
        print(f"  {p['date'].iloc[0]}: L1={llA:.5f}  L1+depth={llB:.5f}  delta={llA-llB:+.6f}")

    pd.concat(all_preds, ignore_index=True).to_parquet(OUT_PRED, index=False, compression="zstd")
    print(f"\nsaved held-out predictions -> {OUT_PRED}")
    print("\nReading it:")
    print("  positive delta on all four folds -> this depth-aware estimator consistently")
    print("     improves held-out forecasts relative to the L1 bucket estimator")
    print("     (conditional on this binning and shrinkage design).")
    print("  mixed / near-zero -> this estimator didn't improve prediction here")
    print("     (not proof depth is useless; a smoother model might still find it).")
    print("  low joint% -> most test states fell back to L1, so B~A by construction.")
    print("  a suspiciously large delta would suggest a leak -- expect a SMALL effect.")


if __name__ == "__main__":
    main()
