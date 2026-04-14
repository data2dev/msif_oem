# MSIF-OEM Crypto Trading System

## What This Is

A complete algorithmic cryptocurrency trading system for the Kraken exchange,
operating at 1-minute bar frequency. It predicts short-term price movements
using a multi-source data fusion approach, then executes trades through a
reinforcement learning agent that optimizes for real-world execution quality
(accounting for slippage, partial fills, and fee structure).

The system is adapted from an academic paper on equity alpha factor synthesis,
with significant modifications for the unregulated crypto environment.

---

## Origin and Adaptation

**Source paper:** Zhao, Y. et al. (2026). "Transforming machine learning
strategies in quantitative stock investment: A multisource information fusion
and online ensemble modeling approach for superior alpha factors."
*Expert Systems With Applications*, 302, 130536.

The paper proposes the MSIF-OEM framework (Multisource Information Fusion and
Online Ensemble Modeling) for China's A-share stock market using daily-frequency
data across 5,000+ stocks. This codebase adapts that framework with these changes:

| Aspect | Original Paper | This Implementation |
|--------|---------------|---------------------|
| Market | China A-shares (regulated) | Crypto on Kraken (unregulated) |
| Frequency | Daily bars, 10-day labels | 1-hr bars, 30-minute labels |
| Universe | 5,000+ stocks | 3 pairs: SOL/USD, AVAX/USD, LINK/USD |
| Data sources | Wind terminal, DolphinDB | Kraken REST + WebSocket APIs |
| Anti-spoofing | Not needed (regulated) | VW-OBI, Multi-level OBI, OFI |
| Execution | Not addressed | SAC reinforcement learning agent |
| Unique signals | N/A | Funding rate (crypto-native) |

**Companion document:** An "OBI note" on Order Book Imbalance spoofing defenses.
It explains why raw OBI is dangerous in crypto (whales place fake orders to
mislead bots) and prescribes three defenses, all implemented in
`features/microstructure.py`.

---

## Architecture: Three Stages + RL Overlay

The system has three ML stages in sequence, plus an RL execution layer:

```
Stage 1: Data Ingestion (Kraken APIs)
  │
  │  4 data streams → anti-spoof filtering
  │
Stage 2: Feature Engineering + PTE-TFE Network
  │
  │  53 features across 4 categories → 4 parallel Transformer encoders
  │  → FC fusion → 64-dim orthogonal feature vector
  │
Stage 3: Online ESGD Alpha Synthesis
  │
  │  K-means regime clustering (K=3) → regime-specific SGD regressor
  │  → alpha signal (predicted 30-min forward return)
  │
RL Layer: SAC Execution Agent
  │
  │  Observes alpha + positions + spread + slippage + drawdown
  │  → outputs target portfolio weights → risk manager validates
  │  → executor places orders → outcome tracker corrects rewards
  │
  └→ Kraken
```

### Stage 1: Data Ingestion

Four data streams from Kraken, each capturing different market dynamics:

1. **OHLCV candles** (REST, 1-minute): Price bars — open, high, low, close, volume.
   Polled every 60 seconds. The backbone data source.

2. **Order book L2** (WebSocket, 25 levels): Real-time bid/ask depth.
   Provides the raw material for OBI and book pressure features.
   Maintained as local state via incremental WS updates.

3. **Trade stream** (WebSocket, tick-level): Every executed trade with price,
   volume, and aggressor side (buy vs sell). Used for TAQ-style features
   like trade imbalance and VPIN.

4. **Funding rate** (REST, periodic): Crypto-native signal from perpetual
   futures. Extreme funding = crowded positioning = mean reversion opportunity.
   Not in the original paper.

**Anti-spoofing gate** (from the OBI note): Before order book data enters the
feature pipeline, three defenses filter out manipulation:

- **Multi-level OBI**: Book sliced into 3 depth bands (levels 1-3, 4-10, 11-25).
  Computed separately so the model can detect divergence between near-spread
  (genuine) and deep-book (often spoofed) imbalance.

- **VW-OBI**: Volume-weighted OBI that discounts orders by inverse distance
  from mid-price. Deep phantom walls get near-zero weight automatically.

- **OFI (Order Flow Imbalance)**: Tracks additions vs cancellations of orders
  over time via WebSocket deltas. Measures actual aggression, not resting state.
  Strips away spoof orders entirely because spoofed orders are placed and
  cancelled — OFI captures that cancellation pattern.

### Stage 2: Feature Engineering + PTE-TFE Network

**Feature categories** (53 total features, grouped into 4 categories):

- **Category A — Raw OHLCV** (5 features): Open, high, low, close, volume.
  Scale-eliminated: prices divided by last bar's close, volume by last bar's
  volume. This normalizes across assets (BTC at $60K vs SOL at $150).

- **Category B — Intra-bar PV** (10 features): TWAP, VWAP ratio, bar range,
  bar return, upper/lower shadow, body ratio, volume intensity, buy ratio
  (from trade stream), trade intensity.

- **Category C — Multi-bar PV** (24 features): 6 base factors × 4 lookback
  windows [5, 10, 20, 60 bars]. Base factors: cumulative return, volatility,
  RSI, Amihud illiquidity, volume-return correlation, Bollinger %b.

- **Category D — Microstructure** (14 features): Multi-level OBI (3 slices),
  VW-OBI, OFI, spread, spread imbalance, depth ratio, trade imbalance,
  large-trade fraction, VPIN, cancel rate, OBI divergence (spoofing
  signature), weighted book pressure.

Each category is serialized as a **30-bar rolling window** (30 minutes of
history). The four resulting tensors have shapes:
- A: [30, 5], B: [30, 10], C: [30, 24], D: [30, 14]

**PTE-TFE Network** (Parallel Transformer Encoder Temporal Feature Extraction):

Four independent Transformer encoders process the four categories in parallel.
Each encoder: Linear embed (input_dim → 64) → sinusoidal positional encoding
→ 2-layer Transformer encoder (8 attention heads, FFN dim 256) → take last
timestep's hidden state as the latent vector [64-dim].

The four 64-dim latent vectors are concatenated [256-dim] and passed through
FC fusion layers (256 → 256 → ReLU → 64) to produce the final **64-dimensional
orthogonal fused feature vector**.

**Why parallel encoders?** The paper shows each encoder develops specialized
temporal attention patterns. OHLCV and multi-bar encoders peak at the last
timestep (macro trends). Microstructure and intra-bar encoders sometimes
attend to slightly earlier bars (micro-events peak before bar close).
A single encoder mixing all inputs loses this specialization.

**Composite loss function** (Eq. 6 in paper):
```
L = -λ1 * Pearson_Corr(mean(features), labels) + λ2 * ||F^T F||_F / (B * D²)
```
- First term: features should predict future returns (prediction alignment).
- Second term: feature dimensions should be uncorrelated (orthogonality).
  This is critical because the downstream ESGD uses linear regressors —
  orthogonal inputs make linear models maximally effective.

**Training**: Offline, weekly. ~485,760 parameters. Uses temporal train/val
split (no shuffling — respects time ordering). Early stopping with patience=5.

### Stage 3: Online ESGD Alpha Synthesis

The ESGD (Ensemble Stochastic Gradient Descent) model synthesizes the final
alpha signal from the 64-dim features. It has two components:

**Mini-batch K-means clustering (K=3)**: Clusters the daily-averaged feature
vectors into 3 market regimes. The paper found K=3 optimal via sensitivity
analysis. In crypto, these map roughly to: trending, range-bound, and
high-volatility/crisis. All assets on the same bar get the same regime label.

**SGD linear regressors (one per cluster)**: Each regressor is a simple linear
function: `alpha = θ_k^T * features`. Because inputs are orthogonal (enforced
by the PTE-TFE loss), linear models are sufficient and avoid overfitting on
streaming data. L2 regularization (λ=0.001) constrains weights.

**At prediction time**: Compute distance from current features to each cluster
centroid → select nearest cluster's regressor → output alpha signal. This is
hard-routing (only one specialist speaks per bar).

**Online update algorithm** (runs every minute bar):
1. New features arrive → compute distance to all centroids.
2. If all distances > d_min (=1.0): increment outlier counter.
3. Update nearest centroid with SGD-like step.
4. Update nearest regressor via `partial_fit()` with new (features, label) pair.
5. When outlier_count ≥ c_max (=10): trigger full re-clustering.
6. Re-clustering uses Calinski-Harabasz index with rollback protection:
   if new CH < old CH, revert to previous clustering.
7. After re-clustering, rebuild all regressors from archived data.

**Why this design?** The expensive part (Transformer training) happens weekly.
The cheap part (SGD gradient step) happens every minute. This separation is
what makes minute-level trading computationally feasible.

### RL Layer: SAC Execution Agent

**Why RL instead of fixed rules?** The original system used Kelly-fraction
position sizing with fixed stop-losses. RL replaces this because execution
is a sequential decision problem: current position affects future options,
fees are path-dependent, and slippage varies with market conditions. A fixed
rule can't jointly optimize these tradeoffs.

**Algorithm: Soft Actor-Critic (SAC)**
- Continuous action space: one weight per asset in [-1, 1]
  (-1 = max short, 0 = flat, +1 = max long)
- Automatic entropy temperature tuning (balances exploration/exploitation)
- Twin Q-networks to reduce overestimation
- Soft target updates (Polyak averaging, τ=0.005)

**State vector (34 dimensions for 3 assets)**:

Per asset (8 × 3 = 24 dims):
- Alpha signal from ESGD
- Alpha confidence (inverse cluster distance)
- Current position as fraction of equity
- Unrealized P&L
- Time held (normalized)
- Current bid-ask spread in bps
- Rolling mean slippage from recent fills
- Recent fill rate (fraction of limit orders that executed)

Global (10 dims):
- Equity / starting equity (drawdown gauge)
- Today's realized P&L / starting equity
- Drawdown budget remaining (0 = halted, 1 = fresh)
- Regime cluster one-hot (3 dims for K=3)
- Rolling volatility (normalized)
- Time-of-day sin/cos encoding (2 dims)

**Action → Position conversion**:
Agent outputs [-1, 1] per asset → scaled by MAX_POSITION_FRAC (20%) →
validated by risk manager (hard caps) → delta from current position
→ orders placed via executor.

Dead zone: |action| < 0.05 treated as flat. Prevents microfluctuations
from generating constant tiny trades.

**Reward function (shaped, not raw P&L)**:
```
r = ΔPortfolio_value
    - 2.0 × |transaction_costs|     # fee penalty
    - 3.0 × |slippage_cost|         # slippage penalty (heaviest)
    - 5.0 × max(0, drawdown_excess) # drawdown penalty
    - 0.0001 × holding_cost         # anti-stagnation
    + 0.0002 × flat_bonus           # reward for not trading on weak signal
```
Normalized by equity for scale-invariance. The heavy slippage penalty teaches
the agent to avoid conditions where fills are poor.

**Outcome correction loop (the key differentiator)**:

This is why the RL agent learns about real-world execution quality:

1. Agent acts → OutcomeTracker records EXPECTED fills (intended price, volume).
2. RewardCalculator computes EXPECTED reward → stored in replay buffer.
3. Later, when Kraken confirms fills, tracker computes ACTUAL reward:
   - Adjusts for real slippage (actual fill price vs intended)
   - Adjusts for partial fills (got 80% of intended volume)
   - Adjusts for taker-vs-maker fee difference
   - Penalizes missed fills (order didn't execute at all)
4. Replay buffer entry is retroactively corrected with actual reward.
5. Slippage and fill rate stats flow back into StateBuilder, so the
   agent's NEXT observation reflects actual execution quality.
6. When SAC samples that transition for training, it learns from
   the real outcome — not the idealized one.

This creates a tight feedback loop: bad execution conditions → agent
observes high slippage in state → learns to avoid those conditions →
execution quality improves.

**Warmup period**: First 500 steps use random actions (fine in paper
trading mode). After warmup, SAC trains on every step with batch_size=256
sampled from the replay buffer (capacity 100,000 transitions).

**Safety architecture**: The risk manager is an uncrossable guardrail:
- Max 20% of equity per asset (configurable)
- Max 80% total portfolio exposure
- 2% hard stop-loss per position
- 5% daily drawdown circuit breaker (kills all trading)
- `--no-rl` flag falls back to rule-based execution

---

## File Structure and Responsibilities

```
msif_oem/
├── config.py                 # ALL tunable parameters in one place
├── requirements.txt          # pip dependencies
├── backfill.py               # Pulls historical data from Kraken REST (run once)
├── collect.py                # Real-time data collection (optional, full fidelity)
├── train.py                  # Trains PTE-TFE + initializes ESGD (run once/weekly)
├── main.py                   # ENTRY POINT — auto-bootstraps if models missing
│
├── data/                     # Data ingestion layer
│   ├── kraken_rest.py        #   REST client: OHLCV, ticker, book, trades, auth
│   ├── kraken_ws.py          #   WebSocket client: book + trade streaming, auto-reconnect
│   └── store.py              #   Central DataStore: candle history, local book state,
│                             #   OFI tracking (additions/cancellations), bar completion
│
├── features/                 # Feature engineering (Stage 2 input)
│   ├── pipeline.py           #   Orchestrates all 4 categories, produces tensors
│   ├── ohlcv.py              #   Category A: 5 features, scale-eliminated
│   ├── intrabar.py           #   Category B: 10 features, within-bar patterns
│   ├── multibar.py           #   Category C: 24 features, rolling technical factors
│   └── microstructure.py     #   Category D: 14 features, anti-spoof OBI/OFI/book
│
├── model/                    # ML models (Stages 2-3)
│   ├── transformer.py        #   PTE-TFE network: 4 parallel Transformer encoders,
│   │                         #   FC fusion, composite loss (prediction + orthogonality)
│   ├── esgd.py               #   ESGD: K-means clustering + SGD ensemble + online
│   │                         #   update with CH-index rollback protection
│   └── trainer.py            #   Training loop: dataset building, temporal split,
│                             #   early stopping, feature extraction for ESGD init
│
├── rl/                       # Reinforcement learning execution layer
│   ├── state.py              #   StateBuilder: constructs 34-dim observation from
│   │                         #   alpha + positions + spread + slippage + drawdown
│   ├── networks.py           #   SAC neural nets: Gaussian Actor + Twin Q-Network
│   ├── agent.py              #   SACAgent: action selection, training loop, warmup,
│   │                         #   automatic entropy tuning, save/load
│   ├── replay_buffer.py      #   Experience replay with retroactive reward correction
│   ├── reward.py             #   Shaped reward: P&L - fees - slippage - drawdown,
│   │                         #   expected vs actual reward computation
│   └── outcome_tracker.py    #   Monitors expected vs real fill outcomes, feeds
│                             #   corrections back into replay buffer + state builder
│
└── trading/                  # Execution layer
    ├── signal.py             #   Converts ESGD alpha → directional trading signal
    ├── risk.py               #   Risk manager: position caps, stop-loss, drawdown
    │                         #   circuit breaker, RL action validation
    └── executor.py           #   Order execution: paper mode (log only) + live mode
                              #   (Kraken API), limit orders to minimize fees
```

---

## Data Flow (One Minute Tick)

```
1. REST poll: get latest completed 1-min candle for each pair
2. WebSocket: book updates + trade ticks accumulate in DataStore
3. DataStore.complete_bar(): snapshot book state, aggregate trades, reset accumulators
4. FeaturePipeline.compute_batch():
   - Category A: ohlcv.compute(candles[-30:]) → [30, 5]
   - Category B: intrabar.compute(bars[-30:]) → [30, 10]
   - Category C: multibar.compute(full_candles)[-30:] → [30, 24]
   - Category D: microstructure.compute(bars[-30:]) → [30, 14]
5. PTE-TFE forward pass: 4 tensors → 4 encoders → concat → FC → [n_assets, 64]
6. ESGD.predict_with_confidence(): features → nearest cluster → SGD regressor → alpha
7. StateBuilder.build(): alpha + positions + spread + slippage + drawdown → [34]
8. SACAgent.select_action(state) → action [-1, 1] per asset
9. RiskManager.validate_rl_targets(): clamp sizes, cap exposure
10. Executor: compute position delta → place limit orders
11. OutcomeTracker.register_action(): record expected fills
12. [Later] OutcomeTracker.resolve_fills(): actual fill data arrives →
    correct replay buffer reward → update slippage/fill stats
13. SACAgent.step(): store transition, sample batch, train SAC
14. [After 30 min delay] ESGD.online_update(): realized return label arrives →
    partial_fit SGD regressor + update cluster centroid
```

---

## ML Methods Stack

| Layer | Method | Role | Params | Update |
|-------|--------|------|--------|--------|
| Feature extraction | 4× Transformer encoder | Temporal pattern extraction per data category | ~485K | Weekly retrain |
| Feature fusion | FC layers + orthogonality loss | Cross-source fusion, decorrelation | Included above | Weekly retrain |
| Regime detection | Mini-batch K-means (K=3) | Market regime clustering | 3 centroids | Online (triggered) |
| Alpha prediction | 3× SGD linear regressor | Return prediction per regime | 64 weights each | Every minute bar |
| Execution | SAC (Actor-Critic) | Portfolio weight optimization | ~228K | Every minute bar |
| Outcome learning | Replay buffer correction | Real slippage/fill feedback | N/A | On fill confirmation |

---

## Configuration Reference (config.py)

**Kraken API**: API_KEY, API_SECRET, REST_BASE, WS_PUBLIC
**Pairs**: XXBTZUSD/BTC/USD, XETHZUSD/ETH/USD, SOLUSD/SOL/USD

**Data**: OHLCV_INTERVAL=1min, BOOK_DEPTH=25, CANDLE_HISTORY=720 bars
**Anti-spoof**: OBI_NEAR=(0,3), OBI_MID=(3,10), OBI_DEEP=(10,25)

**Features**: LOOKBACK_BARS=30, MULTIBAR_WINDOWS=[5,10,20,60]
**PTE-TFE**: EMBED_DIM=64, N_HEADS=8, N_ENCODER_LAYERS=2, FF_DIM=256,
  OUTPUT_DIM=64, DROPOUT=0.1, LR=0.01, EPOCHS=50, EARLY_STOP=5,
  LAMBDA_PRED=0.99, LAMBDA_ORTHO=0.01

**ESGD**: N_CLUSTERS=3, SGD_LR=0.01, SGD_REG=0.001, D_MIN=1.0, C_MAX=10
**Trading**: FORWARD_BARS=30, MIN_SIGNAL=0.001, MAX_POSITION_FRAC=0.20,
  STOP_LOSS=2%, DAILY_DRAWDOWN=5%, ORDER_TYPE=limit
**Mode**: PAPER_TRADING=True (default)

---

## How to Run

```bash
pip install -r requirements.txt
# Edit config.py: set API_KEY and API_SECRET
python main.py              # auto-backfills + trains on first run (~15-20 min)
                            # paper trading by default
python main.py --live       # real orders on Kraken
python main.py --no-rl      # rule-based execution (no RL agent)
python main.py --retrain    # force fresh backfill + retrain
python main.py --backfill-days 14  # more history for first training
```

On first run, main.py checks for models/ptetfe.pt and models/esgd.pkl.
If missing, it automatically runs backfill.py (fetches historical candles
and trades from Kraken REST, ~10-15 min) then train.py (trains Transformer
and initializes ESGD, ~2-5 min). Subsequent runs start trading immediately.

---

## Capital Deployment

The system has no fixed dollar amount. It reads your Kraken account balance
each tick and sizes everything as percentages:
- MAX_POSITION_FRAC = 0.20 → max 20% of equity per asset
- Risk manager caps total exposure at 80% of equity
- With 3 assets: theoretical max deployment = 60% of equity

To limit capital at risk: fund Kraken with only your trading budget,
or lower MAX_POSITION_FRAC in config.py.

Worst-case daily loss: 5% of equity (circuit breaker).

---

## Trade Frequency

The system evaluates every 60 seconds but does NOT trade every minute:
- RL agent's dead zone (|action| < 0.05 = flat) prevents micro-trades
- Fee penalty in reward (2× actual fee) discourages unnecessary trades
- Slippage penalty (3× actual) teaches avoidance of bad conditions
- 30-min prediction horizon means no edge in rapid flipping

Expected: 5-20 trades per day per asset after RL warmup.
During warmup (first ~8 hours): random exploration, more frequent trades.

---

## Key Design Decisions and Rationale

**Why 30-minute forward return target?** Balances signal decay (microstructure
features lose predictive power within hours) vs fee drag (Kraken taker fee
~0.26% eats the edge on very short horizons). 30 minutes is the sweet spot
where the signal-to-cost ratio is highest.

**Why K=3 clusters?** Paper's sensitivity analysis (Fig. 9) shows K=3 maximizes
excess returns. Maps to observable crypto regimes: trending, ranging, volatile.
K=2 loses the volatile regime distinction. K>3 overfits to noise.

**Why linear SGD regressors on top of Transformers?** The Transformer does the
hard nonlinear work (feature extraction). The linear model on orthogonal features
is maximally efficient: fast to update online (<10ms), resistant to overfitting
on streaming data, and convergence is guaranteed under Robbins-Monro conditions.

**Why SAC for the RL agent (not PPO, DQN, etc.)?** Continuous action space
(portfolio weights) rules out DQN. SAC's entropy regularization prevents the
policy from collapsing to a deterministic "always long BTC" strategy, which is
critical in a non-stationary market. The automatic temperature tuning means
one less hyperparameter to manage.

**Why outcome correction instead of just training on actual rewards?** Because
the reward for a transition is computed at action time (we don't know the fill
yet). By the time the fill arrives, we've already moved on. Retroactive
correction is the only way to make the replay buffer reflect reality without
blocking the main loop waiting for Kraken confirmations.

**Why approximate order book in backfill?** Kraken has no historical L2 book
API. The backfill synthesizes plausible book levels from trade data. This
means Category D features are degraded during initial training. Once main.py
runs live with WebSocket book data, the ESGD online update self-corrects with
full-fidelity micro features. The backfill is a bootstrap, not the final model.

---

## What Would Need Changing For...

**Different exchange**: Replace data/kraken_rest.py and data/kraken_ws.py.
The rest of the pipeline is exchange-agnostic. Config.py pair format changes.

**Different assets**: Edit PAIRS dict in config.py. Add/remove pairs.
The RL state dimension auto-adjusts (n_assets × 8 + global dims).

**Different timeframe**: Change OHLCV_INTERVAL and FORWARD_BARS in config.py.
Feature lookback windows in MULTIBAR_WINDOWS may need adjustment.

**More features**: Add a new .py in features/, register it in pipeline.py,
update the corresponding N_*_FEATURES in config.py.

**Different RL algorithm**: Replace rl/agent.py and rl/networks.py. The
state builder, reward calculator, and outcome tracker are algorithm-agnostic.
