# Architecture Overview

## Current State (Phase 0 Audit)

This document describes the current architecture of the ML-Stock_Trading repository as of the Phase 0 audit. It will be updated as the refactoring progresses.

---

## System Diagram (Text)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ML-Stock_Trading                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐    │
│  │   Data Layer     │     │   Model Layer    │     │  Evaluation      │    │
│  │                  │     │                  │     │                  │    │
│  │  yfinance API ───┼────▶│  StockPredictor  │────▶│  backtest.py     │    │
│  │                  │     │  (beta2.py)      │     │                  │    │
│  │  data/*.csv      │     │                  │     │  backtest_       │    │
│  │  (index data)    │     │  Models:         │     │  summary.py      │    │
│  │                  │     │  - LSTM          │     │                  │    │
│  └──────────────────┘     │  - GRU           │     └──────────────────┘    │
│                           │  - BiLSTM        │                              │
│                           │  - CNN-LSTM      │     ┌──────────────────┐    │
│                           │  - Hybrid        │     │  Simulation      │    │
│                           │                  │     │                  │    │
│                           └────────┬─────────┘     │  stock_direction │    │
│                                    │               │  _trader.py      │    │
│                                    ▼               │  (backtest only) │    │
│                           ┌──────────────────┐     │                  │    │
│                           │  Prediction      │     └──────────────────┘    │
│                           │                  │                              │
│                           │  single_stock_   │                              │
│                           │  predict.py      │                              │
│                           │                  │                              │
│                           └──────────────────┘                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Descriptions

### 1. Data Layer (`data/`)

**Files:**
- `datadownload.py` - Script to download index data from yfinance
- `*.csv` - Pre-downloaded index data (NASDAQ, DJI, NDX, RUT, RUI, SPY)

**Data Flow:**
- Individual stock data is fetched on-demand via `yfinance` in `StockPredictor.fetch_data()`
- S&P 500 data is fetched for relative performance comparison
- Technical indicators are calculated inline via `add_technical_indicators()`

**Current Limitations:**
- No survivorship bias handling
- No corporate actions adjustment
- Data fetched live each run (no caching strategy)
- No data validation or integrity checks

---

### 2. Model Layer (`models/`)

**Files:**
- `beta2.py` - Main `StockPredictor` class with multiple model architectures
- `beta1.0.py` - Legacy simpler LSTM-only predictor

**StockPredictor Class (beta2.py):**
```
Responsibilities:
├── Data Fetching (fetch_data)
│   └── Downloads OHLCV from yfinance
│   └── Fetches S&P 500 for comparison
│
├── Feature Engineering (add_technical_indicators)
│   └── Returns, MA5, MA20, RSI, MACD
│   └── Bollinger Bands, Stochastic, ATR, OBV
│   └── Relative performance vs S&P500
│
├── Data Preparation (prepare_data)
│   └── MinMaxScaler for prices
│   └── StandardScaler for features
│   └── Sliding window sequence creation
│
├── Model Building (build_model)
│   └── LSTM/GRU/BiLSTM/CNN-LSTM/Hybrid architectures
│   └── Huber loss for robustness
│
├── Training (train_model)
│   └── EarlyStopping, ReduceLROnPlateau
│   └── ModelCheckpoint, LearningRateScheduler
│   └── Sample weighting (favors recent data)
│
└── Prediction (make_predictions, predict_next_day, predict_future)
    └── Single-step and multi-step forecasting
    └── Trend-based fallback for long horizons
```

**Model Architectures:**
| Type | Description |
|------|-------------|
| `lstm` | 2-layer LSTM (128 units each) + Dense |
| `gru` | Uses GRU instead of LSTM |
| `bilstm` | Bidirectional LSTM |
| `cnn_lstm` | Conv1D + LSTM hybrid |
| `hybrid` | Dual-input (price + features) with separate LSTM paths |

---

### 3. Backtesting Layer (`models/backtest.py`)

**Responsibilities:**
- Random date range generation for backtests
- Train/test split on date ranges
- Model training and evaluation
- Metrics: MSE, RMSE, MAE, Directional Accuracy
- Visualization (regression plots, time series)

**Current Limitations:**
- No transaction costs
- No slippage modeling
- No walk-forward validation
- Random train/test periods (not systematic)
- No Sharpe, Sortino, CAGR, Max Drawdown metrics

---

### 4. Summary/Reporting (`models/backtest_summary.py`)

- Aggregates multiple backtest results
- Generates comparison plots by model type and stock
- Creates text summary reports
- Identifies best performing configurations

---

### 5. Trading Simulation (`tests/stock_direction_trader.py`)

**StockDirectionTrader Class:**
```
Purpose: Binary classification (up/down) with simulated trading

Features:
├── Direction prediction (LSTM classifier)
├── Signal generation (BUY/SELL/HOLD)
├── Trade execution (simulation only - NO LIVE TRADING)
├── Portfolio tracking
├── Performance metrics (accuracy, precision, recall, F1)
└── Transaction cost modeling (proportional fee)
```

**SAFETY: This is BACKTEST-ONLY. No broker API connections exist.**

---

## Current Issues Identified

### Critical (Data Leakage Risks)

1. **Scaler Fit-Transform on Full Data**
   - `prepare_data()` fits scalers on training data, but same scaler is used for test data
   - Risk: Information from test period may leak via scaler parameters if not carefully handled
   - Location: `beta2.py:349-350`

2. **Train/Val Split is Random**
   - Uses `validation_split` in Keras (random, not chronological)
   - Risk: Future data leaks into training
   - Location: `beta2.py:500`

3. **Technical Indicators May Use Future Data**
   - Forward-filling NaN values after indicator calculation
   - Rolling windows need sufficient burn-in period
   - Location: `beta2.py:315-317`

4. **S&P 500 Alignment**
   - S&P data aligned by date intersection
   - Potential for look-ahead if dates misaligned
   - Location: `beta2.py:167-194`

### Moderate (Unrealistic Assumptions)

1. **No Transaction Costs in Main Backtest**
   - `backtest.py` has no cost model
   - `stock_direction_trader.py` has optional cost but defaults to 0

2. **No Slippage Model**
   - Assumes trades execute at exact close prices

3. **No Bid-Ask Spread**
   - Assumes zero spread

4. **Missing Risk Metrics**
   - No Sharpe ratio, Sortino, CAGR, Calmar
   - Only RMSE, MAE, directional accuracy

### Low (Maintainability)

1. **No Reproducibility Seeds** (except in stock_direction_trader.py)
2. **No Configuration Files** - Parameters hardcoded
3. **Monolithic StockPredictor** - Does data, features, model, training, prediction
4. **No Package Structure** - Flat file layout

---

## Planned Architecture (Post-Refactor)

```
ml_stock_trading/
├── src/
│   ├── data/
│   │   ├── fetcher.py          # Data fetching with caching
│   │   ├── validator.py        # Data integrity checks
│   │   └── features.py         # Technical indicator calculation
│   │
│   ├── models/
│   │   ├── base.py             # Base model interface
│   │   ├── lstm.py             # LSTM variants
│   │   └── math_first/         # Non-ML baseline strategies
│   │       ├── momentum.py
│   │       └── mean_reversion.py
│   │
│   ├── portfolio/
│   │   ├── construction.py     # Position sizing, risk parity
│   │   └── risk.py             # Exposure limits, drawdown controls
│   │
│   ├── backtest/
│   │   ├── engine.py           # Walk-forward harness
│   │   ├── costs.py            # Transaction cost models
│   │   └── metrics.py          # Sharpe, Sortino, CAGR, etc.
│   │
│   └── cli.py                  # Unified CLI entry point
│
├── configs/                    # YAML configuration files
├── tests/                      # Unit and integration tests
├── docs/                       # Documentation
│   ├── architecture.md
│   ├── backtest_protocol.md
│   └── risk_policy.md
│
└── claude-progress.md          # Development log
```

---

## Entry Points

### Current
| Command | Description |
|---------|-------------|
| `python single_stock_predict.py AAPL` | Predict single stock |
| `python models/beta2.py NVDA hybrid` | Train and predict |
| `python models/backtest.py --stock AAPL --model lstm` | Run backtest |
| `./run_backtests.sh` | Run comprehensive backtests |
| `python tests/stock_direction_trader.py AAPL` | Trading simulation |

### Planned
| Command | Description |
|---------|-------------|
| `python -m ml_stock_trading backtest --config config.yaml` | Unified backtest |
| `python -m ml_stock_trading train --strategy momentum` | Train strategy |
| `python -m ml_stock_trading report --run-id abc123` | Generate report |

---

## Dependencies

**Core:**
- numpy, pandas, matplotlib
- tensorflow/keras
- scikit-learn
- yfinance

**Optional:**
- ta (technical analysis library, partially used)
- joblib (model persistence)

---

*Last Updated: Phase 0 Audit*
