"""
Training Script
================
Phase 2: Train the PTE-TFE network and initialize ESGD from collected data.

Usage:
    python train.py --data data_store/collected_*.json
    python train.py --data data_store/  (loads all JSON files in directory)
"""

import argparse
import glob
import json
import logging
import os
import numpy as np
import config as cfg
from model.trainer import Trainer
from model.esgd import ESGDModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-12s] %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")


def load_collected_data(path: str) -> list:
    """Load collected samples from JSON file(s)."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.json")))
    else:
        files = sorted(glob.glob(path))

    if not files:
        raise FileNotFoundError(f"No data files found at {path}")

    all_samples = []
    for f in files:
        log.info(f"Loading {f}")
        with open(f) as fh:
            data = json.load(fh)
            all_samples.extend(data["samples"])

    log.info(f"Loaded {len(all_samples)} total samples from {len(files)} file(s)")
    return all_samples


def build_training_pairs(samples: list) -> tuple:
    """
    Build (features, labels) pairs for training.
    
    Labels are forward returns computed from consecutive close prices.
    Returns features_list, labels_list.
    """
    # Group by symbol and sort by timestamp
    by_symbol = {}
    for s in samples:
        sym = s["symbol"]
        if sym not in by_symbol:
            by_symbol[sym] = []
        by_symbol[sym].append(s)

    for sym in by_symbol:
        by_symbol[sym].sort(key=lambda x: x["ts"])

    features_list = []
    labels_list = []

    for sym, sym_samples in by_symbol.items():
        log.info(f"  {sym}: {len(sym_samples)} samples")

        for i in range(len(sym_samples) - cfg.FORWARD_BARS):
            sample = sym_samples[i]
            future_sample = sym_samples[i + cfg.FORWARD_BARS]

            # Forward return label
            close_now = sample["close"]
            close_future = future_sample["close"]
            if close_now <= 0:
                continue
            label = (close_future - close_now) / close_now

            # Convert features from lists back to arrays
            feats = {}
            for key in ["A", "B", "C", "D"]:
                feats[key] = np.array(sample["features"][key], dtype=np.float32)

            features_list.append(feats)
            labels_list.append(label)

    log.info(f"Built {len(features_list)} training pairs")
    return features_list, labels_list


def main():
    parser = argparse.ArgumentParser(description="MSIF-OEM Training")
    parser.add_argument("--data", type=str, default=cfg.DATA_DIR, help="Data path or directory")
    parser.add_argument("--device", type=str, default=None, help="PyTorch device (cuda/cpu)")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("  MSIF-OEM Training Pipeline")
    log.info("=" * 60)

    # ── Step 1: Load data ──
    samples = load_collected_data(args.data)

    # ── Step 2: Build training pairs ──
    features_list, labels_list = build_training_pairs(samples)

    if len(features_list) < 100:
        log.error(f"Only {len(features_list)} training pairs — need at least 100.")
        log.error("Collect more data with: python collect.py --hours 24")
        return

    # ── Step 3: Train PTE-TFE network ──
    log.info("\n" + "─" * 40)
    log.info("Training PTE-TFE Transformer network")
    log.info("─" * 40)

    trainer = Trainer(device=args.device)
    feat_a, feat_b, feat_c, feat_d, labels = trainer.build_dataset(
        features_list, labels_list
    )

    model = trainer.train(feat_a, feat_b, feat_c, feat_d, labels)

    # Save model
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    model_path = os.path.join(cfg.MODEL_DIR, "ptetfe.pt")
    trainer.save_model(model, model_path)

    # ── Step 4: Extract features for ESGD initialization ──
    log.info("\n" + "─" * 40)
    log.info("Extracting features for ESGD")
    log.info("─" * 40)

    extracted = trainer.extract_features(model, feat_a, feat_b, feat_c, feat_d)
    log.info(f"Extracted features: {extracted.shape}")

    # ── Step 5: Initialize ESGD ──
    log.info("\n" + "─" * 40)
    log.info("Initializing ESGD model")
    log.info("─" * 40)

    raw_labels = np.array(labels_list[:extracted.shape[0]])
    esgd = ESGDModel()
    esgd.initialize(extracted, raw_labels)

    # Save ESGD
    esgd_path = os.path.join(cfg.MODEL_DIR, "esgd.pkl")
    esgd.save(esgd_path)

    # ── Summary ──
    log.info("\n" + "=" * 60)
    log.info("  TRAINING COMPLETE")
    log.info(f"  PTE-TFE model: {model_path}")
    log.info(f"  ESGD model:    {esgd_path}")
    log.info(f"  Training samples: {len(features_list)}")
    log.info(f"  Feature dim: {extracted.shape[1]}")
    log.info(f"  Clusters: {cfg.N_CLUSTERS}")
    log.info("=" * 60)
    log.info("\nNext: python main.py")


if __name__ == "__main__":
    main()
