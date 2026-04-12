"""
MSIF-OEM Configuration
======================
All architecture parameters in one place.
Matches the paper's Table 3 where applicable, adapted for crypto/Kraken.

API keys are loaded in this priority order:
  1. Environment variables (KRAKEN_API_KEY, KRAKEN_API_SECRET)
     → Set by GCP startup script from Secret Manager
     → Or export locally before running
  2. Hardcoded values below (for local development only)
"""

import os

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  KRAKEN API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Environment variables take priority (set by GCP Secret Manager)
# Fall back to hardcoded values for local development
API_KEY    = os.environ.get("KRAKEN_API_KEY", "")
API_SECRET = os.environ.get("KRAKEN_API_SECRET", "")

REST_BASE = "https://api.kraken.com"
WS_PUBLIC = "wss://ws.kraken.com/v2"

# Trading pairs (Kraken REST format → WS format)
PAIRS = {
    "SOLUSD":   "SOL/USD",
    "LINKUSD":  "LINK/USD",
    "AVAXUSD":  "AVAX/USD",
}
PAIR_LIST_REST = list(PAIRS.keys())
PAIR_LIST_WS = list(PAIRS.values())

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DATA INGESTION (Stage 1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OHLCV_INTERVAL = 60          # hourly bars
BOOK_DEPTH = 25              # order book levels
CANDLE_HISTORY_BARS = 720    # 720 hourly bars = 30 days

# Anti-spoof OBI slicing
OBI_NEAR_LEVELS = (0, 3)     # levels 1-3
OBI_MID_LEVELS = (3, 10)     # levels 4-10
OBI_DEEP_LEVELS = (10, 25)   # levels 11-25

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FEATURE ENGINEERING (Stage 2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOOKBACK_BARS = 12           # 12-hour rolling window (starts faster)
MULTIBAR_WINDOWS = [3, 6, 12, 24]  # 3h, 6h, 12h, 24h lookbacks

# Feature category dimensions (approximate, exact set in each module)
N_OHLCV_FEATURES = 5
N_INTRABAR_FEATURES = 10
N_MULTIBAR_FEATURES = 24
N_MICRO_FEATURES = 14

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PTE-TFE NETWORK (Stage 2 - Transformer)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EMBED_DIM = 64               # hidden dimension D_emb
N_HEADS = 8                  # multi-head attention heads
N_ENCODER_LAYERS = 2         # transformer encoder depth
FF_DIM = 256                 # feedforward hidden dim (4 * D_emb)
OUTPUT_DIM = 64              # fused orthogonal feature dimension D_o
DROPOUT = 0.1

# Training
LEARNING_RATE = 0.01
TRAIN_EPOCHS = 50
EARLY_STOP_PATIENCE = 5
LAMBDA_PRED = 0.99           # weight for prediction loss
LAMBDA_ORTHO = 0.01          # weight for orthogonality loss

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ESGD MODEL (Stage 3)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
N_CLUSTERS = 3               # K market regimes
CLUSTER_BATCH_SIZE = 1024
CLUSTER_MAX_ITER = 100
CLUSTER_LR = 0.01            # eta_c

SGD_LR = 0.01                # eta_s
SGD_REGULARIZATION = 0.001   # lambda_r (L2)
D_MIN = 1.0                  # distance threshold for outlier detection
C_MAX = 10                   # outlier count to trigger re-cluster

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PREDICTION & TRADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORWARD_BARS = 1             # predict next 1 hour
MIN_SIGNAL_THRESHOLD = 0.005 # minimum alpha to trade (0.5% predicted move)
MAX_POSITION_FRAC = 0.20     # max 20% of portfolio per asset
STOP_LOSS_PCT = 0.02         # 2% per-position stop
DAILY_DRAWDOWN_LIMIT = 0.05  # 5% daily max drawdown
ORDER_TYPE = "limit"         # limit orders to avoid taker fees
MIN_HOLD_MINUTES = 30        # hold at least 30 min (half an hour bar)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PATHS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA_DIR = "data_store"
MODEL_DIR = "models"
LOG_DIR = "logs"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MODES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PAPER_TRADING = True         # True = log signals only, no real orders
