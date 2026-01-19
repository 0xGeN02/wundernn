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

# Window sizes for rolling features (reduced for memory)
WINDOWS = [5, 10, 50]

# Lag periods (reduced for memory)
LAGS = [1, 3, 10]


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rolling window features for each sequence.

    Memory-optimized version that assigns features directly to DataFrame
    and only processes rows where prediction is needed.
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

    # Key features to compute rolling stats for
    key_features = ["p0", "p6", "v0", "v6", "dp0", "dv0"]

    # Process rolling features - assign directly to avoid memory buildup
    for feat in tqdm(key_features, desc="Rolling features"):
        for w in WINDOWS:
            df[f"{feat}_mean_{w}"] = (
                grouped[feat]
                .transform(lambda x: x.rolling(window=w, min_periods=1).mean())
                .astype(np.float32)
            )
            df[f"{feat}_std_{w}"] = (
                grouped[feat]
                .transform(lambda x: x.rolling(window=w, min_periods=1).std())
                .fillna(0)
                .astype(np.float32)
            )

        # Add min/max only for window 50
        df[f"{feat}_max_50"] = (
            grouped[feat]
            .transform(lambda x: x.rolling(window=50, min_periods=1).max())
            .astype(np.float32)
        )
        df[f"{feat}_min_50"] = (
            grouped[feat]
            .transform(lambda x: x.rolling(window=50, min_periods=1).min())
            .astype(np.float32)
        )

        gc.collect()

    # Add lag features
    print("Creating lag and momentum features...")
    for feat in tqdm(key_features, desc="Lag features"):
        for lag in LAGS:
            # Lag
            lag_feat = grouped[feat].shift(lag).fillna(0).astype(np.float32)
            df[f"{feat}_lag{lag}"] = lag_feat

            # Difference from lag
            df[f"{feat}_diff{lag}"] = (df[feat] - lag_feat).astype(np.float32)

            del lag_feat
        gc.collect()

    print("Creating spread features...")
    # Collect simple features in a dict to avoid fragmentation
    simple_features = {}

    # Spread features (bid-ask)
    simple_features["spread_p0_p6"] = (df["p0"] - df["p6"]).astype(np.float32)
    simple_features["spread_v0_v6"] = (df["v0"] - df["v6"]).astype(np.float32)

    print("Creating mid price features...")
    # Mid price and movement
    mid_price = ((df["p0"] + df["p6"]) / 2).astype(np.float32)
    simple_features["mid_price"] = mid_price
    simple_features["mid_price_lag1"] = (
        mid_price.groupby(df["seq_ix"]).shift(1).fillna(0).astype(np.float32)
    )
    simple_features["mid_price_change"] = (
        mid_price - simple_features["mid_price_lag1"]
    ).astype(np.float32)
    del mid_price

    print("Creating volume imbalance features...")
    # Volume imbalance
    bid_vol = df[["v0", "v1", "v2", "v3", "v4", "v5"]].sum(axis=1).astype(np.float32)
    ask_vol = df[["v6", "v7", "v8", "v9", "v10", "v11"]].sum(axis=1).astype(np.float32)
    simple_features["vol_imbalance"] = (
        (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)
    ).astype(np.float32)
    simple_features["total_volume"] = (bid_vol + ask_vol).astype(np.float32)
    del bid_vol, ask_vol

    print("Creating trade features...")
    # Trade features
    simple_features["trade_intensity"] = (
        df[["dv0", "dv1", "dv2", "dv3"]].sum(axis=1).astype(np.float32)
    )
    simple_features["trade_price_mean"] = (
        df[["dp0", "dp1", "dp2", "dp3"]].mean(axis=1).astype(np.float32)
    )

    print("Creating step normalization features...")
    # Step in sequence (normalized and polynomial)
    step_norm = (df["step_in_seq"] / 999.0).astype(np.float32)
    simple_features["step_norm"] = step_norm
    simple_features["step_norm_sq"] = (step_norm**2).astype(np.float32)
    del step_norm

    # Add all simple features at once to avoid fragmentation
    print("Adding simple features to dataframe...")
    for col_name, col_data in simple_features.items():
        df[col_name] = col_data
    del simple_features

    gc.collect()
    print(f"Total columns after feature engineering: {len(df.columns)}")
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

    # Different params for t0 vs t1
    if target_name == "t1":
        # t1 is harder - use deeper trees, more regularization
        params = {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "num_leaves": 127,  # More capacity
            "learning_rate": 0.03,  # Lower LR
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "min_child_samples": 50,  # Less regularization on samples
            "lambda_l1": 0.5,  # More L1
            "lambda_l2": 0.5,  # More L2
            "min_gain_to_split": 0.01,
            "verbose": -1,
            "seed": 42,
            "n_jobs": -1,
        }
    else:  # t0
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
        num_boost_round=1000,  # More rounds
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=75),  # More patience
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
    reference_price: np.ndarray = None,
    t1_is_relative: bool = False,
) -> dict:
    """Evaluate models using the competition metric."""
    from competition_package.utils import weighted_pearson_correlation

    pred_t0 = model_t0.predict(X)
    pred_t1_relative = model_t1.predict(X)

    # Convert t1 from relative to absolute
    if t1_is_relative and reference_price is not None:
        pred_t1 = reference_price * (1 + pred_t1_relative)
    else:
        pred_t1 = pred_t1_relative

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

    # CRITICAL: Filter early to reduce memory - keep only what we need for rolling features
    # We need some context before step 99 for rolling windows, so keep step >= 0
    print("Filtering to essential rows for training...")
    # Keep all rows but mark what we'll use for training
    train_filtered = train_df.copy()
    del train_df
    gc.collect()

    print("Loading validation data...")
    valid_df = pd.read_parquet(VALID_PATH)
    print(f"Valid shape: {valid_df.shape}")

    # Create rolling features on FULL sequences (needed for rolling windows)
    print("\n--- Processing TRAIN data ---")
    train_filtered = create_rolling_features(train_filtered)

    # NOW filter to only scored samples
    print("Filtering to scored samples only...")
    train_scored = train_filtered[train_filtered["need_prediction"] == 1].copy()
    print(f"Scored train samples: {len(train_scored)}")

    # Get feature columns
    feature_cols = get_feature_columns(train_scored)
    print(f"\nTotal features: {len(feature_cols)}")

    # Prepare train data and FREE original train_filtered
    X_train = train_scored[feature_cols].values.astype(np.float32)
    y_train_t0 = train_scored["t0"].values.astype(np.float32)

    # For t1: use relative change approach
    # Reference: mid price (average of best bid and ask)
    reference_price_train = train_scored["mid_price"].values.astype(np.float32)
    y_train_t1_absolute = train_scored["t1"].values.astype(np.float32)
    y_train_t1_relative = (y_train_t1_absolute - reference_price_train) / (
        np.abs(reference_price_train) + 1e-8
    )

    print(
        f"t1 relative change stats: mean={y_train_t1_relative.mean():.6f}, std={y_train_t1_relative.std():.6f}"
    )

    del train_filtered, train_scored
    gc.collect()
    print("Train data prepared, memory freed.")

    # Now process validation
    print("\n--- Processing VALID data ---")
    valid_df = create_rolling_features(valid_df)

    valid_scored = valid_df[valid_df["need_prediction"] == 1].copy()
    print(f"Scored valid samples: {len(valid_scored)}")

    X_valid = valid_scored[feature_cols].values.astype(np.float32)
    y_valid_t0 = valid_scored["t0"].values.astype(np.float32)

    # For t1: use relative change
    reference_price_valid = valid_scored["mid_price"].values.astype(np.float32)
    y_valid_t1_absolute = valid_scored["t1"].values.astype(np.float32)
    y_valid_t1_relative = (y_valid_t1_absolute - reference_price_valid) / (
        np.abs(reference_price_valid) + 1e-8
    )
    del valid_df, valid_scored
    gc.collect()
    print("Valid data prepared, memory freed.")

    # Train models
    model_t0 = train_lgb_model(X_train, y_train_t0, X_valid, y_valid_t0, "t0")
    model_t1 = train_lgb_model(
        X_train, y_train_t1_relative, X_valid, y_valid_t1_relative, "t1"
    )

    # Evaluate (convert t1 predictions back to absolute values)
    scores = evaluate_model(
        model_t0,
        model_t1,
        X_valid,
        y_valid_t0,
        y_valid_t1_absolute,
        feature_cols,
        reference_price=reference_price_valid,
        t1_is_relative=True,
    )

    # Save models and config
    print("\nSaving models...")
    model_t0.save_model(f"{OUTPUT_DIR}/model_t0.txt")
    model_t1.save_model(f"{OUTPUT_DIR}/model_t1.txt")

    # Save feature configuration
    config = {
        "feature_columns": feature_cols,
        "windows": WINDOWS,
        "lags": LAGS,
        "all_raw_features": ALL_FEATURES,
        "key_features": ["p0", "p6", "v0", "v6", "dp0", "dv0"],
        "t1_is_relative": True,  # t1 model predicts relative changes
    }
    with open(f"{OUTPUT_DIR}/config.pkl", "wb") as f:
        pickle.dump(config, f)

    print(f"\nModels saved to {OUTPUT_DIR}/")
    print("Done!")

    return scores


if __name__ == "__main__":
    main()
