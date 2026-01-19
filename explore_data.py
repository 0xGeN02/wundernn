"""Script to explore the competition datasets."""

import pandas as pd

print("Loading datasets...")
train = pd.read_parquet("competition_package/datasets/train.parquet")
valid = pd.read_parquet("competition_package/datasets/valid.parquet")

print("=== Dataset Shapes ===")
print(f"Train: {train.shape}")
print(f"Valid: {valid.shape}")

print("\n=== Columns ===")
print(train.columns.tolist())

print("\n=== Sequence info ===")
print(f"Train sequences: {train.seq_ix.nunique()}")
print(f"Valid sequences: {valid.seq_ix.nunique()}")
print(f"Steps per sequence: {train.step_in_seq.max() + 1}")

print("\n=== Target stats ===")
print(train[["t0", "t1"]].describe())

print("\n=== Feature correlations with targets ===")
features = [
    c
    for c in train.columns
    if c not in ["seq_ix", "step_in_seq", "need_prediction", "t0", "t1"]
]
for target in ["t0", "t1"]:
    print(f"\nTop correlations with {target}:")
    corrs = train[features].corrwith(train[target]).abs().sort_values(ascending=False)
    print(corrs.head(10))
