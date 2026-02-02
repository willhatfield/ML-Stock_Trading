# Backtest Protocol

**Version:** 1.0
**Status:** DRAFT - Pending Review
**Purpose:** Single source of truth for backtesting evaluation

---

## Overview

This document defines the rules, assumptions, and metrics for evaluating any trading strategy in this repository. **No strategy results are valid unless they follow this protocol.**

---

## 1. Universe Definition

### 1.1 Stock Selection
- **Primary Universe:** Large-cap US equities with:
  - Minimum market cap: $10B (at time of inclusion)
  - Minimum average daily volume: $10M (20-day average)
  - Listed on NYSE or NASDAQ
  - Not an ADR, SPAC, or penny stock

### 1.2 Survivorship Bias Handling
- **Assumption (Current):** We use point-in-time constituent lists where available
- **Default:** If survivorship-free data unavailable, explicitly state "SURVIVORSHIP BIAS NOT ADDRESSED"
- **Mitigation:** Use broad index ETFs (SPY, QQQ) as proxies where individual stock data may have survivorship issues

### 1.3 Data Sources
| Source | Data Type | Frequency | Adjustment |
|--------|-----------|-----------|------------|
| yfinance | OHLCV | Daily | Split/dividend adjusted |
| Local CSV | Index data | Daily | Pre-adjusted |

---

## 2. Data Frequency and Alignment

### 2.1 Time Frequency
- **Primary:** Daily close prices (end-of-day)
- **Timestamps:** All times in UTC or explicit timezone
- **Trading Hours:** Only use data from market hours (9:30 AM - 4:00 PM ET)

### 2.2 Data Alignment Rules
1. All features must be aligned to the **same timestamp**
2. For multi-asset strategies, use **inner join** on dates (only days where all assets have data)
3. Missing data handling:
   - **Never forward-fill prices** (creates look-ahead)
   - Drop rows with missing critical data
   - Document any imputation clearly

### 2.3 Corporate Actions
- Use adjusted prices (already split/dividend adjusted)
- **Assumption:** yfinance provides correctly adjusted data
- **Risk:** Adjustment errors in historical data - document any anomalies

---

## 3. Train/Validation/Test Scheme

### 3.1 Walk-Forward Protocol (Required)

```
Time →
├─────────────────────────────────────────────────────────────────────────────┤
│  Train Period 1  │  Val 1  │  Test 1  │                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│         │  Train Period 2  │  Val 2  │  Test 2  │                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                   │  Train Period 3  │  Val 3  │  Test 3  │                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                             │  Train Period 4  │  Val 4  │  Test 4  │       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Split Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `train_years` | 2 | Years of training data |
| `val_months` | 3 | Validation period (parameter tuning) |
| `test_months` | 3 | Out-of-sample test period |
| `step_months` | 3 | How much to slide forward each fold |
| `embargo_days` | 5 | Gap between train/val and test to prevent leakage |

### 3.3 Rules
1. **NEVER** use random shuffling of time series data
2. **ALWAYS** ensure test data is strictly after training data
3. **NEVER** use any information from validation/test periods in training
4. **ALWAYS** refit scalers on training data only at each fold
5. **ENFORCE** embargo period between train and test to account for autocorrelation

### 3.4 Minimum Data Requirements
- Training: Minimum 252 trading days (1 year)
- Validation: Minimum 63 trading days (1 quarter)
- Test: Minimum 63 trading days (1 quarter)
- Total walk-forward folds: Minimum 4

---

## 4. Leakage Rules

### 4.1 Information Cutoff
At time `t`, only the following information is available:
- Prices and volumes for all times ≤ `t`
- Technical indicators computed using only data ≤ `t`
- Fundamental data with release dates ≤ `t`

### 4.2 Prohibited Operations
| Operation | Why It's Wrong | Correct Approach |
|-----------|----------------|------------------|
| `fit_transform(all_data)` | Leaks test distribution | `fit(train)` then `transform(test)` |
| `fillna(method='bfill')` | Backward fill uses future | `fillna(method='ffill')` or drop |
| `rolling().mean()` without shift | Includes current day | `rolling().mean().shift(1)` |
| `pct_change()` as feature and target | Target leaks via correlation | Use lagged returns only |
| Random validation split | Future samples in validation | Chronological split |

### 4.3 Scaler Protocol
```python
# CORRECT
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)  # NO fit
X_test_scaled = scaler.transform(X_test)  # NO fit

# WRONG
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)  # Leaks test stats
X_train_scaled = X_all_scaled[:train_idx]
```

### 4.4 Feature Engineering Protocol
All technical indicators must:
1. Use `.shift(1)` to ensure only past data is used
2. Have a burn-in period equal to the longest lookback window
3. Not use future-dated corporate actions or news

---

## 5. Transaction Cost and Slippage Model

### 5.1 Cost Components

| Component | Default Model | Realistic Range |
|-----------|---------------|-----------------|
| Commission | $0 (zero-commission era) | $0-0.01/share |
| Spread | 0.1% of trade value | 0.01%-0.5% |
| Slippage | 0.05% of trade value | 0%-0.2% |
| **Total One-Way** | **0.15%** | 0.01%-0.7% |
| **Total Round-Trip** | **0.30%** | 0.02%-1.4% |

### 5.2 Slippage Model (Default)

```python
def apply_slippage(price: float, is_buy: bool, volume: float, avg_daily_volume: float) -> float:
    """
    Simple volume-impact slippage model.

    Args:
        price: Intended execution price
        is_buy: True if buying, False if selling
        volume: Number of shares to trade
        avg_daily_volume: 20-day average daily volume

    Returns:
        Adjusted execution price
    """
    # Market impact: increases with trade size relative to ADV
    participation_rate = volume / avg_daily_volume
    market_impact = 0.1 * np.sqrt(participation_rate)  # Square-root impact

    # Add fixed spread component
    spread_cost = 0.001  # 10 bps half-spread

    total_impact = spread_cost + market_impact

    if is_buy:
        return price * (1 + total_impact)
    else:
        return price * (1 - total_impact)
```

### 5.3 Cost Sensitivity Analysis
All reported results must include sensitivity analysis at:
- Low cost: 0.05% round-trip
- Base cost: 0.30% round-trip (default)
- High cost: 1.00% round-trip

---

## 6. Performance Metrics

### 6.1 Required Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **CAGR** | `(final/initial)^(252/days) - 1` | Annualized return |
| **Volatility** | `std(returns) * sqrt(252)` | Annualized volatility |
| **Sharpe Ratio** | `(CAGR - rf) / Volatility` | Risk-adjusted return (rf=0.04) |
| **Sortino Ratio** | `(CAGR - rf) / Downside_Vol` | Downside risk-adjusted |
| **Max Drawdown** | `max(peak - trough) / peak` | Worst peak-to-trough loss |
| **Calmar Ratio** | `CAGR / Max_Drawdown` | Return per unit drawdown |
| **Win Rate** | `winning_trades / total_trades` | Proportion of winning trades |
| **Profit Factor** | `gross_profit / gross_loss` | Ratio of wins to losses |
| **Turnover** | `sum(abs(delta_position)) / 2 / avg_capital` | Annual turnover |
| **Exposure** | `avg(abs(position))` | Average capital deployed |

### 6.2 Benchmark Comparison
All strategies must report metrics relative to:
- **Buy & Hold:** 100% long the underlying asset
- **SPY:** S&P 500 ETF as market proxy
- **Risk-Free:** Treasury rate (assume 4% annual)

### 6.3 Statistical Significance
- Report **95% confidence intervals** for Sharpe ratio
- Bootstrap or block bootstrap standard errors
- Minimum required: `Sharpe > 0.5` with `p < 0.05` to claim statistical significance

---

## 7. Robustness Checks

### 7.1 Required Robustness Tests

1. **Cost Sensitivity**
   - Results at 0.05%, 0.30%, 1.00% round-trip costs
   - Report breakeven cost level

2. **Subperiod Analysis**
   - Split test period in half, report metrics for each half
   - Flag if second half significantly worse (potential overfitting)

3. **Regime Analysis**
   - Bull markets (S&P 500 > 200-day MA)
   - Bear markets (S&P 500 < 200-day MA)
   - High volatility (VIX > 25)
   - Low volatility (VIX < 15)

4. **Parameter Sensitivity**
   - Vary key parameters ±20%
   - Report performance degradation
   - Flag if performance cliff exists

5. **Universe Robustness**
   - Test on different stock subsets
   - Test on different time periods not used for development

### 7.2 Overfitting Red Flags
- [ ] Training Sharpe >> Test Sharpe (>50% degradation)
- [ ] Many parameters relative to data points
- [ ] Complex entry/exit rules with specific thresholds
- [ ] Performance concentrated in specific time periods
- [ ] Large difference between first and second half of test

---

## 8. Reporting Requirements

### 8.1 Required Report Contents

Every backtest report must include:

1. **Strategy Description**
   - Clear mathematical description
   - Parameters and how they were chosen
   - Data requirements

2. **Data Summary**
   - Date range
   - Number of assets
   - Missing data handling

3. **Walk-Forward Results**
   - Per-fold metrics
   - Aggregate metrics
   - Equity curve plot

4. **Benchmark Comparison**
   - vs. Buy & Hold
   - vs. SPY
   - Information ratio

5. **Cost Analysis**
   - Turnover
   - Cost sensitivity
   - Breakeven cost

6. **Robustness Tests**
   - Subperiod analysis
   - Parameter sensitivity
   - Regime analysis

7. **Statistical Tests**
   - Sharpe confidence interval
   - Significance test vs. benchmark

### 8.2 Report Format

```
backtest_results/
├── {strategy}_{timestamp}/
│   ├── config.yaml          # All parameters
│   ├── metrics.json          # Numerical results
│   ├── equity_curve.csv      # Daily portfolio values
│   ├── trades.csv            # All trades
│   ├── report.md             # Human-readable summary
│   └── plots/
│       ├── equity_curve.png
│       ├── drawdown.png
│       ├── monthly_returns.png
│       └── parameter_sensitivity.png
```

---

## 9. Code Verification Checklist

Before trusting any backtest results, verify:

- [ ] No `fit_transform` on combined train+test data
- [ ] Validation split is chronological, not random
- [ ] Embargo period exists between train and test
- [ ] Technical indicators use `.shift(1)` appropriately
- [ ] No `bfill` or backward-looking operations
- [ ] Scalers fitted only on training data per fold
- [ ] Transaction costs applied
- [ ] Results reproducible with fixed seeds
- [ ] Multiple walk-forward folds used
- [ ] Cost sensitivity analysis included

---

## Appendix A: Quick Reference

### Minimum Viable Backtest Command

```bash
python -m ml_stock_trading backtest \
    --strategy momentum \
    --start-date 2015-01-01 \
    --end-date 2024-01-01 \
    --walk-forward \
    --train-years 2 \
    --test-months 3 \
    --cost-bps 30 \
    --output results/
```

### Expected Output

```
Walk-Forward Backtest Results
=============================
Strategy: momentum_volatility_target
Period: 2017-01-01 to 2024-01-01 (28 folds)

Aggregate Metrics:
  CAGR:           8.2% ± 2.1%
  Sharpe:         0.72 ± 0.18
  Max Drawdown:   -15.3%
  Calmar:         0.54
  Turnover:       4.2x annual

Benchmark Comparison:
  vs Buy&Hold:    +1.2% CAGR, +0.15 Sharpe
  vs SPY:         -0.8% CAGR, +0.08 Sharpe

Cost Sensitivity:
  @ 5 bps:  Sharpe 0.85
  @ 30 bps: Sharpe 0.72
  @ 100 bps: Sharpe 0.41
```

---

*This protocol must be followed for all strategy evaluations. Any deviations must be explicitly documented and justified.*
