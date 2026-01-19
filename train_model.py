"""
Training script for the competition.

This script trains a LightGBM model with rolling window features
for fast CPU inference.
"""

import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

# Paths
TRAIN_PATH = "competition_package/datasets/train.parquet"
VALID_PATH = "competition_package/datasets/valid.parquet"
OUTPUT_DIR = "competition_package/my_solution"

# Feature columns
PRICE_FEATURES = [f"p{i}" for i in range(12)]
VOLUME_FEATURES = [f"v{i}" for i in range(12)]
TRADE_PRICE_FEATURES = [f"dp{i}" for i in range(4)]
TRADE_VOLUME_FEATURES = [f"dv{i}" for i in range(4)]
ALL_FEATURES = (
    PRICE_FEATURES + VOLUME_FEATURES + TRADE_PRICE_FEATURES + TRADE_VOLUME_FEATURES
)

# Window sizes for rolling features
WINDOWS = [5, 10, 20, 50]


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rolling window features for each sequence.

    This function creates:
    - Rolling mean, std for each window size
    - Lagged features
    - Difference features

    Optimized for low memory usage.
    """
    import gc

    print("Creating rolling features...")

    # Sort by sequence and step
    df = df.sort_values(["seq_ix", "step_in_seq"]).reset_index(drop=True)

    # Convert to float32 to save memory
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)

    # Group by sequence
    grouped = df.groupby("seq_ix")

    # Key features to compute rolling stats for (reduced set for speed)
    key_features = ["p0", "p6", "v0", "dp0"]  # Reducido aún más

    for feat in tqdm(key_features, desc="Rolling features"):
        for w in WINDOWS:
            # Rolling mean
            df[f"{feat}_mean_{w}"] = (
                grouped[feat]
                .transform(lambda x: x.rolling(window=w, min_periods=1).mean())
                .astype(np.float32)
            )

            # Rolling std
            df[f"{feat}_std_{w}"] = (
                grouped[feat]
                .transform(lambda x: x.rolling(window=w, min_periods=1).std())
                .fillna(0)
                .astype(np.float32)
            )
        gc.collect()

    # Add lag features only for key features (not all)
    print("Creating lag features...")
    for feat in tqdm(key_features, desc="Lag features"):
        # Lag 1
        lag1 = grouped[feat].shift(1).fillna(0).astype(np.float32)
        df[f"{feat}_lag1"] = lag1
        # Difference
        df[f"{feat}_diff1"] = (df[feat] - lag1).astype(np.float32)
        del lag1
        gc.collect()

    # Combine spread features (bid-ask)
    df["spread_p0_p6"] = (df["p0"] - df["p6"]).astype(np.float32)
    df["spread_v0_v6"] = (df["v0"] - df["v6"]).astype(np.float32)

    # Volume imbalance
    bid_vol = df[["v0", "v1", "v2", "v3", "v4", "v5"]].sum(axis=1)
    ask_vol = df[["v6", "v7", "v8", "v9", "v10", "v11"]].sum(axis=1)
    df["vol_imbalance"] = ((bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)).astype(
        np.float32
    )
    del bid_vol, ask_vol
    gc.collect()

    # Trade imbalance
    df["trade_intensity"] = (
        df[["dv0", "dv1", "dv2", "dv3"]].sum(axis=1).astype(np.float32)
    )

    # Step in sequence (normalized)
    df["step_norm"] = (df["step_in_seq"] / 999.0).astype(np.float32)

    gc.collect()
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get all feature column names."""
    exclude = ["seq_ix", "step_in_seq", "need_prediction", "t0", "t1"]
    return [c for c in df.columns if c not in exclude]


def train_lgb_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    target_name: str,
) -> lgb.Booster:
    """Train a LightGBM model for one target."""

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 100,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
        "seed": 42,
        "n_jobs": -1,
    }

    print(f"\nTraining LightGBM for {target_name}...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50),
        ],
    )

    return model


def evaluate_model(
    model_t0: lgb.Booster,
    model_t1: lgb.Booster,
    X: np.ndarray,
    y_t0: np.ndarray,
    y_t1: np.ndarray,
    feature_names: list,
) -> dict:
    """Evaluate models using the competition metric."""
    from competition_package.utils import weighted_pearson_correlation

    pred_t0 = model_t0.predict(X)
    pred_t1 = model_t1.predict(X)

    score_t0 = weighted_pearson_correlation(y_t0, pred_t0)
    score_t1 = weighted_pearson_correlation(y_t1, pred_t1)
    mean_score = (score_t0 + score_t1) / 2

    print("\nValidation Scores:")
    print(f"  t0: {score_t0:.6f}")
    print(f"  t1: {score_t1:.6f}")
    print(f"  Mean: {mean_score:.6f}")

    return {"t0": score_t0, "t1": score_t1, "mean": mean_score}


def main():
    import gc

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print("Loading training data...")
    train_df = pd.read_parquet(TRAIN_PATH)
    print(f"Train shape: {train_df.shape}")

    print("Loading validation data...")
    valid_df = pd.read_parquet(VALID_PATH)
    print(f"Valid shape: {valid_df.shape}")

    # Create rolling features
    print("\n--- Processing TRAIN data ---")
    train_df = create_rolling_features(train_df)

    # Get feature columns
    feature_cols = get_feature_columns(train_df)
    print(f"\nTotal features: {len(feature_cols)}")

    # Filter to only rows where prediction is needed (steps 99-999)
    print("Filtering scored samples...")
    train_scored = train_df[train_df["need_prediction"] == 1].copy()
    print(f"Scored train samples: {len(train_scored)}")

    # Prepare train data and FREE original train_df
    X_train = train_scored[feature_cols].values.astype(np.float32)
    y_train_t0 = train_scored["t0"].values.astype(np.float32)
    y_train_t1 = train_scored["t1"].values.astype(np.float32)
    del train_df, train_scored
    gc.collect()
    print("Train data prepared, memory freed.")

    # Now process validation
    print("\n--- Processing VALID data ---")
    valid_df = create_rolling_features(valid_df)

    valid_scored = valid_df[valid_df["need_prediction"] == 1].copy()
    print(f"Scored valid samples: {len(valid_scored)}")

    X_valid = valid_scored[feature_cols].values.astype(np.float32)
    y_valid_t0 = valid_scored["t0"].values.astype(np.float32)
    y_valid_t1 = valid_scored["t1"].values.astype(np.float32)
    del valid_df, valid_scored
    gc.collect()
    print("Valid data prepared, memory freed.")

    # Train models
    model_t0 = train_lgb_model(X_train, y_train_t0, X_valid, y_valid_t0, "t0")
    model_t1 = train_lgb_model(X_train, y_train_t1, X_valid, y_valid_t1, "t1")

    # Evaluate
    scores = evaluate_model(
        model_t0, model_t1, X_valid, y_valid_t0, y_valid_t1, feature_cols
    )

    # Save models and config
    print("\nSaving models...")
    model_t0.save_model(f"{OUTPUT_DIR}/model_t0.txt")
    model_t1.save_model(f"{OUTPUT_DIR}/model_t1.txt")

    # Save feature configuration
    config = {
        "feature_columns": feature_cols,
        "windows": WINDOWS,
        "all_raw_features": ALL_FEATURES,
        "key_features": ["p0", "p6", "v0", "dp0"],
    }
    with open(f"{OUTPUT_DIR}/config.pkl", "wb") as f:
        pickle.dump(config, f)

    print(f"\nModels saved to {OUTPUT_DIR}/")
    print("Done!")

    return scores


if __name__ == "__main__":
    main()
