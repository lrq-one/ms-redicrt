import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_split_dp", required=True)
    ap.add_argument("--qc_csv", required=True)
    ap.add_argument("--out_split_dp", required=True)
    ap.add_argument("--target_keep_rate", type=float, default=0.8130)
    args = ap.parse_args()

    base_split_dp = Path(args.base_split_dp)
    out_split_dp = Path(args.out_split_dp)
    out_split_dp.mkdir(parents=True, exist_ok=True)

    train_ids = pd.read_csv(base_split_dp / "train_ids.csv")
    val_ids = pd.read_csv(base_split_dp / "val_ids.csv")
    test_ids = pd.read_csv(base_split_dp / "test_ids.csv")

    qc = pd.read_csv(args.qc_csv)

    # 保证只处理原 random train 里的 spec
    df = train_ids.merge(qc, on=["spec_id", "mol_id", "group_id"], how="left")

    missing_qc = df["support_PWR_abs006"].isna().sum()
    if missing_qc > 0:
        print("[WARN] missing QC rows:", missing_qc)

    # 缺 QC 的默认当作坏样本
    for c in [
        "n_peaks",
        "max_intensity_frac",
        "entropy_norm",
        "precursor_survival_yield",
        "support_PR_abs006",
        "support_PWR_abs006",
        "true_oos_intensity",
    ]:
        if c not in df.columns:
            raise RuntimeError(f"missing QC column: {c}")

    df["n_peaks"] = df["n_peaks"].fillna(0)
    df["max_intensity_frac"] = df["max_intensity_frac"].fillna(1.0)
    df["entropy_norm"] = df["entropy_norm"].fillna(0.0)
    df["precursor_survival_yield"] = df["precursor_survival_yield"].fillna(1.0)
    df["support_PWR_abs006"] = df["support_PWR_abs006"].fillna(0.0)
    df["support_PR_abs006"] = df["support_PR_abs006"].fillna(0.0)

    # hard bad: 非常明显的问题谱图
    df["bad_too_few_peaks"] = df["n_peaks"] < 3
    df["bad_low_support"] = df["support_PWR_abs006"] < 0.35
    df["bad_precursor_dominated"] = df["precursor_survival_yield"] > 0.95
    df["bad_single_peak"] = df["max_intensity_frac"] > 0.98
    df["bad_low_entropy"] = df["entropy_norm"] < 0.03

    hard_bad_cols = [
        "bad_too_few_peaks",
        "bad_low_support",
        "bad_precursor_dominated",
        "bad_single_peak",
        "bad_low_entropy",
    ]
    df["hard_bad"] = df[hard_bad_cols].any(axis=1)

    # soft bad score：越大越差
    support_bad = (1.0 - df["support_PWR_abs006"]).clip(0.0, 1.0)
    precursor_bad = df["precursor_survival_yield"].clip(0.0, 1.0)
    single_peak_bad = df["max_intensity_frac"].clip(0.0, 1.0)
    entropy_bad = (1.0 - df["entropy_norm"]).clip(0.0, 1.0)
    peak_count_bad = (1.0 / np.sqrt(df["n_peaks"].clip(lower=1))).clip(0.0, 1.0)

    df["qcv1_bad_score"] = (
        0.45 * support_bad
        + 0.20 * precursor_bad
        + 0.15 * single_peak_bad
        + 0.10 * entropy_bad
        + 0.10 * peak_count_bad
    )

    n_total = len(df)
    target_keep = int(round(n_total * args.target_keep_rate))

    # 先去掉 hard bad，再按 bad_score 保留最好的 target_keep 个
    candidates = df[~df["hard_bad"]].copy()

    if len(candidates) >= target_keep:
        keep_df = candidates.sort_values("qcv1_bad_score", ascending=True).head(target_keep).copy()
    else:
        print("[WARN] candidates fewer than target_keep; keeping all non-hard-bad candidates")
        keep_df = candidates.copy()

    keep_ids = set(keep_df["spec_id"].tolist())
    new_train = train_ids[train_ids["spec_id"].isin(keep_ids)].copy()
    drop_df = df[~df["spec_id"].isin(keep_ids)].copy()

    new_train.to_csv(out_split_dp / "train_ids.csv", index=False)
    val_ids.to_csv(out_split_dp / "val_ids.csv", index=False)
    test_ids.to_csv(out_split_dp / "test_ids.csv", index=False)

    keep_df.to_csv(out_split_dp / "qcv1_train_keep_details.csv", index=False)
    drop_df.to_csv(out_split_dp / "qcv1_train_drop_details.csv", index=False)

    print("base_split_dp:", base_split_dp)
    print("out_split_dp:", out_split_dp)
    print("old train rows:", len(train_ids), "mols:", train_ids["mol_id"].nunique())
    print("new train rows:", len(new_train), "mols:", new_train["mol_id"].nunique())
    print("drop rows:", len(drop_df))
    print("keep rate:", len(new_train) / max(1, len(train_ids)))
    print("val unchanged rows:", len(val_ids), "mols:", val_ids["mol_id"].nunique())
    print("test unchanged rows:", len(test_ids), "mols:", test_ids["mol_id"].nunique())

    print("\nHard bad reason counts:")
    for c in hard_bad_cols:
        print(c, int(df[c].sum()))

    print("\nKept QC summary:")
    print(keep_df[[
        "n_peaks",
        "max_intensity_frac",
        "entropy_norm",
        "precursor_survival_yield",
        "support_PWR_abs006",
        "qcv1_bad_score",
    ]].describe())

    print("\nDropped QC summary:")
    print(drop_df[[
        "n_peaks",
        "max_intensity_frac",
        "entropy_norm",
        "precursor_survival_yield",
        "support_PWR_abs006",
        "qcv1_bad_score",
    ]].describe())


if __name__ == "__main__":
    main()
