"""
Competition solution using LightGBM with rolling window features.

This solution:
1. Maintains a history buffer for each sequence
2. Computes rolling statistics on-the-fly
3. Uses LightGBM for fast CPU inference
"""

import os
import pickle

import lightgbm as lgb
import numpy as np

# Adjust path to import utils from parent directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
import sys

sys.path.append(f"{CURRENT_DIR}/..")

from utils import DataPoint, ScorerStepByStep


class FeatureBuilder:
    """Builds features from sequence history."""

    def __init__(self, config: dict):
        self.windows = config["windows"]  # [5, 10, 20, 50]
        self.key_features = config["key_features"]
        self.all_raw_features = config["all_raw_features"]
        self.feature_columns = config["feature_columns"]

        # Create feature index mapping
        self.raw_feat_indices = {f: i for i, f in enumerate(self.all_raw_features)}

    def build_features(self, history: list, step_in_seq: int) -> np.ndarray:
        """Build feature vector from sequence history."""

        # Convert history to numpy array
        history_arr = np.array(history, dtype=np.float32)
        n_steps = len(history)
        current = history_arr[-1]

        features = []

        # 1. Add all raw features (32)
        features.extend(current.tolist())

        # 2. Rolling features for key features
        for feat_name in self.key_features:
            feat_idx = self.raw_feat_indices[feat_name]
            feat_history = history_arr[:, feat_idx]

            for w in self.windows:
                # Rolling mean
                if n_steps >= w:
                    roll_mean = np.mean(feat_history[-w:])
                    roll_std = np.std(feat_history[-w:])
                else:
                    roll_mean = np.mean(feat_history)
                    roll_std = np.std(feat_history) if n_steps > 1 else 0.0

                features.append(roll_mean)
                features.append(roll_std)

        # 3. Lag and diff features for all features
        for feat_name in self.all_raw_features:
            feat_idx = self.raw_feat_indices[feat_name]
            current_val = current[feat_idx]

            # Lag 1
            if n_steps > 1:
                lag1 = history_arr[-2, feat_idx]
            else:
                lag1 = 0.0

            features.append(lag1)
            features.append(current_val - lag1)  # diff

        # 4. Spread features
        p0_idx = self.raw_feat_indices["p0"]
        p6_idx = self.raw_feat_indices["p6"]
        v0_idx = self.raw_feat_indices["v0"]
        v6_idx = self.raw_feat_indices["v6"]

        features.append(current[p0_idx] - current[p6_idx])  # spread_p0_p6
        features.append(current[v0_idx] - current[v6_idx])  # spread_v0_v6

        # 5. Volume imbalance
        bid_vol = sum(current[self.raw_feat_indices[f"v{i}"]] for i in range(6))
        ask_vol = sum(current[self.raw_feat_indices[f"v{i}"]] for i in range(6, 12))
        vol_imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)
        features.append(vol_imbalance)

        # 6. Trade intensity
        trade_vol = sum(current[self.raw_feat_indices[f"dv{i}"]] for i in range(4))
        features.append(trade_vol)

        # 7. Normalized step
        features.append(step_in_seq / 999.0)

        return np.array(features, dtype=np.float32)


class PredictionModel:
    """
    LightGBM-based solution with rolling features.

    Uses two LightGBM models (one for each target) with features
    computed from a rolling window of sequence history.
    """

    def __init__(self, model_path=""):
        self.current_seq_ix = None
        self.sequence_history = []

        # Load configuration and models
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Load config
        config_path = os.path.join(base_dir, "config.pkl")
        with open(config_path, "rb") as f:
            self.config = pickle.load(f)

        # Initialize feature builder
        self.feature_builder = FeatureBuilder(self.config)

        # Load LightGBM models
        self.model_t0 = lgb.Booster(model_file=os.path.join(base_dir, "model_t0.txt"))
        self.model_t1 = lgb.Booster(model_file=os.path.join(base_dir, "model_t1.txt"))

        print(f"Loaded LightGBM models from {base_dir}")

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        """Make prediction for the given data point."""

        # Reset state on new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.sequence_history = []

        # Update history with current state
        self.sequence_history.append(data_point.state.copy())

        # If prediction not needed, return None
        if not data_point.need_prediction:
            return None

        # Build features from history
        features = self.feature_builder.build_features(
            self.sequence_history, data_point.step_in_seq
        )

        # Reshape for prediction (1, n_features)
        X = features.reshape(1, -1)

        # Predict
        pred_t0 = self.model_t0.predict(X)[0]
        pred_t1 = self.model_t1.predict(X)[0]

        return np.array([pred_t0, pred_t1], dtype=np.float32)


if __name__ == "__main__":
    # Local testing
    test_file = f"{CURRENT_DIR}/../datasets/valid.parquet"

    if os.path.exists(test_file):
        model = PredictionModel()
        scorer = ScorerStepByStep(test_file)

        print("Testing LightGBM Solution...")
        results = scorer.score(model)

        print("\nResults:")
        print(f"Mean Weighted Pearson correlation: {results['weighted_pearson']:.6f}")
        for target in scorer.targets:
            print(f"  {target}: {results[target]:.6f}")
    else:
        print("Valid parquet not found for testing.")
