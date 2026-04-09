"""
PTE-TFE Trainer
================
Offline training for the Transformer feature extraction network.
Handles dataset construction, training with composite loss, early stopping,
and temporal K-fold cross-validation per the paper's Section 4.2.
"""

import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model.transformer import PTETFE, PTETFELoss
import config as cfg

log = logging.getLogger(__name__)


class FeatureDataset(Dataset):
    """Dataset of (4 feature tensors, label) samples."""

    def __init__(self, feat_a, feat_b, feat_c, feat_d, labels):
        self.a = torch.FloatTensor(feat_a)
        self.b = torch.FloatTensor(feat_b)
        self.c = torch.FloatTensor(feat_c)
        self.d = torch.FloatTensor(feat_d)
        self.y = torch.FloatTensor(labels)

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, idx):
        return self.a[idx], self.b[idx], self.c[idx], self.d[idx], self.y[idx]


class Trainer:
    """Trains the PTE-TFE network on collected feature data."""

    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        log.info(f"Trainer using device: {self.device}")

        self.model = None
        self.best_model_state = None

    def build_dataset(self, all_features: list, all_labels: list) -> tuple:
        """
        Build training arrays from collected data.
        
        Args:
            all_features: list of dicts {"A": [T, D_A], "B": ..., "C": ..., "D": ...}
            all_labels: list of floats (forward returns)
            
        Returns:
            (feat_a, feat_b, feat_c, feat_d, labels) as numpy arrays
        """
        n = len(all_features)
        assert n == len(all_labels), "Features and labels must match"

        feat_a = np.stack([f["A"] for f in all_features])  # [N, T, D_A]
        feat_b = np.stack([f["B"] for f in all_features])
        feat_c = np.stack([f["C"] for f in all_features])
        feat_d = np.stack([f["D"] for f in all_features])
        labels = np.array(all_labels)

        # Standard normalization per feature dimension
        for arr in [feat_a, feat_b, feat_c, feat_d]:
            for d in range(arr.shape[2]):
                col = arr[:, :, d]
                mu, sigma = col.mean(), col.std()
                if sigma > 1e-8:
                    arr[:, :, d] = (col - mu) / sigma

        # Normalize labels
        labels = (labels - labels.mean()) / (labels.std() + 1e-8)

        log.info(f"Dataset: {n} samples, A={feat_a.shape}, labels range="
                 f"[{labels.min():.3f}, {labels.max():.3f}]")
        return feat_a, feat_b, feat_c, feat_d, labels

    def train(self, feat_a, feat_b, feat_c, feat_d, labels,
              val_split: float = 0.2) -> PTETFE:
        """
        Train the PTE-TFE network.
        
        Uses temporal split (last val_split fraction as validation).
        
        Returns:
            Trained PTETFE model
        """
        n = feat_a.shape[0]
        split = int(n * (1 - val_split))

        train_ds = FeatureDataset(
            feat_a[:split], feat_b[:split], feat_c[:split], feat_d[:split],
            labels[:split]
        )
        val_ds = FeatureDataset(
            feat_a[split:], feat_b[split:], feat_c[split:], feat_d[split:],
            labels[split:]
        )

        train_dl = DataLoader(train_ds, batch_size=min(256, split), shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

        # Build model
        input_dims = {
            "A": feat_a.shape[2],
            "B": feat_b.shape[2],
            "C": feat_c.shape[2],
            "D": feat_d.shape[2],
        }
        self.model = PTETFE(input_dims).to(self.device)
        criterion = PTETFELoss().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.LEARNING_RATE)

        log.info(f"Training PTE-TFE: {sum(p.numel() for p in self.model.parameters())} params")
        log.info(f"  Train: {split} samples, Val: {n - split} samples")

        best_val_loss = float("inf")
        patience_counter = 0
        self.best_model_state = None

        for epoch in range(cfg.TRAIN_EPOCHS):
            # ── Train ──
            self.model.train()
            train_losses = []

            for a, b, c, d, y in train_dl:
                a, b, c, d, y = [t.to(self.device) for t in (a, b, c, d, y)]
                optimizer.zero_grad()
                features = self.model(a, b, c, d)
                loss, (pred_l, ortho_l) = criterion(features, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # ── Validate ──
            self.model.eval()
            with torch.no_grad():
                val_losses = []
                for a, b, c, d, y in val_dl:
                    a, b, c, d, y = [t.to(self.device) for t in (a, b, c, d, y)]
                    features = self.model(a, b, c, d)
                    loss, (pred_l, ortho_l) = criterion(features, y)
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            if epoch % 5 == 0 or epoch == cfg.TRAIN_EPOCHS - 1:
                log.info(f"  Epoch {epoch:3d}: train={train_loss:.4f}, val={val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= cfg.EARLY_STOP_PATIENCE:
                    log.info(f"  Early stopping at epoch {epoch}")
                    break

        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        self.model.eval()

        log.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
        return self.model

    def extract_features(self, model: PTETFE, feat_a, feat_b, feat_c, feat_d) -> np.ndarray:
        """
        Extract features from trained model for ESGD initialization.
        
        Returns:
            [N, D_o] numpy array of extracted features
        """
        model.eval()
        ds = FeatureDataset(feat_a, feat_b, feat_c, feat_d,
                            np.zeros(feat_a.shape[0]))
        dl = DataLoader(ds, batch_size=256, shuffle=False)

        all_features = []
        with torch.no_grad():
            for a, b, c, d, _ in dl:
                a, b, c, d = [t.to(self.device) for t in (a, b, c, d)]
                features = model(a, b, c, d)
                all_features.append(features.cpu().numpy())

        return np.concatenate(all_features, axis=0)

    def save_model(self, model: PTETFE, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(model.state_dict(), path)
        log.info(f"Model saved to {path}")

    def load_model(self, path: str, input_dims: dict) -> PTETFE:
        model = PTETFE(input_dims).to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        log.info(f"Model loaded from {path}")
        return model
