"""
MSIF-OEM Configuration
======================
All architecture parameters in one place.
Matches the paper's Table 3 where applicable, adapted for crypto/Kraken.
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  KRAKEN API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
API_KEY = ""
API_SECRET = ""

REST_BASE = "https://api.kraken.com"
WS_PUBLIC = "wss://ws.kraken.com/v2"

# Trading pairs (Kraken REST format → WS format)
PAIRS = {
    "XXBTZUSD": "BTC/USD",
    "XETHZUSD": "ETH/USD",
    "SOLUSD":   "SOL/USD",
}
PAIR_LIST_REST = list(PAIRS.keys())
PAIR_LIST_WS = list(PAIRS.values())

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DATA INGESTION (Stage 1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OHLCV_INTERVAL = 1           # minutes
BOOK_DEPTH = 25              # order book levels
CANDLE_HISTORY_BARS = 720    # 12 hours of 1-min bars for initial load

# Anti-spoof OBI slicing
OBI_NEAR_LEVELS = (0, 3)     # levels 1-3
OBI_MID_LEVELS = (3, 10)     # levels 4-10
OBI_DEEP_LEVELS = (10, 25)   # levels 11-25

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FEATURE ENGINEERING (Stage 2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOOKBACK_BARS = 30           # rolling window for time series input
MULTIBAR_WINDOWS = [5, 10, 20, 60]  # lookback periods for multi-bar factors

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
FORWARD_BARS = 30            # prediction target: 30-min forward return
MIN_SIGNAL_THRESHOLD = 0.001 # minimum alpha to trade
MAX_POSITION_FRAC = 0.20     # max 20% of portfolio per asset
STOP_LOSS_PCT = 0.02         # 2% per-position stop
DAILY_DRAWDOWN_LIMIT = 0.05  # 5% daily max drawdown
ORDER_TYPE = "limit"         # limit orders to avoid taker fees

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
