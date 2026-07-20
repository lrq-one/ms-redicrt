import pandas as pd
from pathlib import Path

bad_mols = [20273, 24874, 26644]

old_proc = Path("data/proc/nist20_qtof_cid_safe19707")
old_split = Path("data/split/nist20_qtof_cid_safe19707_qcv1_trainonly")

out_proc = Path("data/proc/nist20_qtof_cid_safe19659")
out_split = Path("data/split/nist20_qtof_cid_safe19659_qcv1_trainonly")
dag_dir = Path("data/frag/nist20_qtof_cid_safe19707_d3_mhp_qtof_cid_base/dags")

out_proc.mkdir(parents=True, exist_ok=True)
out_split.mkdir(parents=True, exist_ok=True)

spec = pd.read_pickle(old_proc / "spec_df.pkl")
mol = pd.read_pickle(old_proc / "mol_df.pkl")
ann = pd.read_pickle(old_proc / "ann_df.pkl")

train = pd.read_csv(old_split / "train_ids.csv")
val = pd.read_csv(old_split / "val_ids.csv")
test = pd.read_csv(old_split / "test_ids.csv")

spec2 = spec[~spec["mol_id"].isin(bad_mols)].copy()
mol2 = mol[~mol["mol_id"].isin(bad_mols)].copy()
train2 = train[~train["mol_id"].isin(bad_mols)].copy()
val2 = val.copy()
test2 = test.copy()

spec2.to_pickle(out_proc / "spec_df.pkl")
mol2.to_pickle(out_proc / "mol_df.pkl")
ann.to_pickle(out_proc / "ann_df.pkl")

train2.to_csv(out_split / "train_ids.csv", index=False)
val2.to_csv(out_split / "val_ids.csv", index=False)
test2.to_csv(out_split / "test_ids.csv", index=False)

used_mols = set(spec2["mol_id"].astype(int))
missing = []
for mid in sorted(used_mols):
    p = dag_dir / f"{mid}.pickle.bz2"
    if not p.exists():
        missing.append(mid)

print("===== PRUNED SAFE FULL SUMMARY =====")
print("removed bad mols:", bad_mols)
print("spec:", spec2.shape)
print("mol:", mol2.shape)
print("train:", train2.shape)
print("val:", val2.shape)
print("test:", test2.shape)
print("total split spectra:", len(train2) + len(val2) + len(test2))
print("used molecules:", len(used_mols))
print("missing dag count:", len(missing))
print("missing dag mol_ids:", missing[:50])

print("\npaths:")
print(f"spec_fp: {out_proc}/spec_df.pkl")
print(f"mol_fp: {out_proc}/mol_df.pkl")
print(f"ann_fp: {out_proc}/ann_df.pkl")
print(f"split_dp: {out_split}")
print(f"base_frag_dp: {dag_dir}")
