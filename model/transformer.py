"""
PTE-TFE Network
================
Parallel Transformer Encoder Temporal Feature Extraction.
4 parallel Transformer encoders (one per data category) → FC fusion → orthogonal features.

Architecture matches Zhao et al. (2026) Section 3.2:
  - Input: 4 time series [batch, T, D_in_k] for k in {A, B, C, D}
  - Each encoder: Linear embed → Positional encoding → Transformer encoder → last timestep
  - Fusion: Concat(4 latent vectors) → FC layers → D_o orthogonal features
  - Loss: -λ1 * Corr(mean_features, labels) + λ2 * ||F^T F||_F (orthogonality)
"""

import math
import torch
import torch.nn as nn
import numpy as np
import config as cfg


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Eq. 2 in paper)."""

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class BranchEncoder(nn.Module):
    """Single Transformer encoder branch for one data category."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.embed = nn.Linear(input_dim, cfg.EMBED_DIM)
        self.pos_enc = PositionalEncoding(cfg.EMBED_DIM, max_len=200)
        self.dropout = nn.Dropout(cfg.DROPOUT)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.EMBED_DIM,
            nhead=cfg.N_HEADS,
            dim_feedforward=cfg.FF_DIM,
            dropout=cfg.DROPOUT,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.N_ENCODER_LAYERS,
        )

    def forward(self, x):
        """
        Args:
            x: [batch, T, D_in]
        Returns:
            latent: [batch, D_emb] — last timestep hidden state
        """
        h = self.embed(x)           # [B, T, D_emb]
        h = self.pos_enc(h)         # + positional encoding
        h = self.dropout(h)
        h = self.transformer(h)     # [B, T, D_emb]
        return h[:, -1, :]          # last timestep


class PTETFE(nn.Module):
    """
    Parallel Transformer Encoder Temporal Feature Extraction network.
    
    4 branch encoders → concatenate → FC fusion → D_o orthogonal features
    """

    def __init__(self, input_dims: dict):
        """
        Args:
            input_dims: {"A": 5, "B": 10, "C": 24, "D": 14}
        """
        super().__init__()

        self.branch_a = BranchEncoder(input_dims["A"])
        self.branch_b = BranchEncoder(input_dims["B"])
        self.branch_c = BranchEncoder(input_dims["C"])
        self.branch_d = BranchEncoder(input_dims["D"])

        concat_dim = cfg.EMBED_DIM * 4  # 256

        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, concat_dim),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(concat_dim, cfg.OUTPUT_DIM),
        )

    def forward(self, xa, xb, xc, xd):
        """
        Args:
            xa: [B, T, D_A] — OHLCV
            xb: [B, T, D_B] — intra-bar PV
            xc: [B, T, D_C] — multi-bar PV
            xd: [B, T, D_D] — microstructure
            
        Returns:
            features: [B, D_o] — orthogonal fused features
        """
        la = self.branch_a(xa)
        lb = self.branch_b(xb)
        lc = self.branch_c(xc)
        ld = self.branch_d(xd)

        concat = torch.cat([la, lb, lc, ld], dim=-1)  # [B, 4*D_emb]
        features = self.fusion(concat)                  # [B, D_o]
        return features

    def get_latents(self, xa, xb, xc, xd):
        """Return individual branch latents for analysis."""
        with torch.no_grad():
            return {
                "A": self.branch_a(xa),
                "B": self.branch_b(xb),
                "C": self.branch_c(xc),
                "D": self.branch_d(xd),
            }


class PTETFELoss(nn.Module):
    """
    Composite loss (Eq. 6 in paper):
      L = -λ1 * Corr(mean(F), Y) + λ2 * ||F^T F||_F / (B * D_o^2)
    """

    def __init__(self, lambda_pred=cfg.LAMBDA_PRED, lambda_ortho=cfg.LAMBDA_ORTHO):
        super().__init__()
        self.lambda_pred = lambda_pred
        self.lambda_ortho = lambda_ortho

    def forward(self, features, labels):
        """
        Args:
            features: [B, D_o]
            labels: [B] — future returns
            
        Returns:
            loss, (pred_loss, ortho_loss) for logging
        """
        B, D = features.shape

        # Prediction loss: negative correlation between mean feature and labels
        feat_mean = features.mean(dim=1)  # [B]
        pred_loss = -self._pearson_corr(feat_mean, labels)

        # Orthogonality loss: ||F^T F||_F normalized
        cov = torch.mm(features.t(), features)  # [D, D]
        ortho_loss = torch.norm(cov, p="fro") / (B * D * D)

        loss = self.lambda_pred * pred_loss + self.lambda_ortho * ortho_loss
        return loss, (pred_loss.item(), ortho_loss.item())

    @staticmethod
    def _pearson_corr(x, y):
        """Pearson correlation between two 1D tensors."""
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        num = (x_centered * y_centered).sum()
        denom = torch.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum() + 1e-8)
        return num / denom
