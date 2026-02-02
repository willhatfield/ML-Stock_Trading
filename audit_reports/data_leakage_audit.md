# Data Leakage Audit Report

**Repository:** ML-Stock_Trading
**Audit Date:** 2026-01-07
**Auditor:** DataLeakageAuditor Subagent

---

## Executive Summary

This audit identified **8 critical issues**, **5 moderate issues**, and **3 minor issues** related to data leakage and look-ahead bias in the ML-Stock_Trading codebase. The most severe problems are: (1) scalers are fit on the entire dataset including future data before train/test splits, (2) Keras `validation_split` shuffles data randomly causing temporal leakage, and (3) technical indicators use `fillna(method='ffill')` which can propagate future information when NaN handling occurs after index operations. **No backtests can be trusted until these critical issues are resolved.**

---

## Critical Issues (Must Fix)

### CRITICAL-1: Scaler Fitted on Full Dataset Before Split
**File:** `/Users/willhatfield/ML-Stock_Trading/models/beta2.py`
**Lines:** 349-350, 373-377

**Description:**
In `prepare_data()`, the `scaler_price` is fitted using `fit_transform()` on the entire `price_data` array before any train/test split occurs. This means the scaler learns the min/max values from future data that should be unseen during training.

```python
# Line 349-350
price_data_2d = price_data.reshape(-1, 1)
scaled_prices = self.scaler_price.fit_transform(price_data_2d).flatten()
```

Similarly, `scaler_features` is fitted on the entire feature dataset at lines 373-377:
```python
# Line 373-374
self.scaler_features.fit(feature_data)
scaled_features = self.scaler_features.transform(feature_data)
```

**Impact:** Severe look-ahead bias. The model has implicit knowledge of future price ranges.

**Recommended Fix:**
1. Split data chronologically BEFORE fitting scalers
2. Fit scalers only on training data
3. Use `transform()` (not `fit_transform()`) on validation/test data
4. Save scaler state with model for inference

---

### CRITICAL-2: Keras validation_split Causes Random Shuffling
**File:** `/Users/willhatfield/ML-Stock_Trading/models/beta2.py`
**Lines:** 548-556, 558-566

**Description:**
The `train_model()` function uses Keras's `validation_split=0.2` parameter, which by default takes the **last** 20% of the data as validation. However, this is problematic because:
1. Keras shuffles data by default unless `shuffle=False` is explicitly set
2. No `shuffle=False` parameter is passed to `model.fit()`

```python
# Lines 548-556
history = self.model.fit(
    [x_price, x_features], y,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=validation_split,
    callbacks=callbacks,
    sample_weight=sample_weights,
    verbose=1
)
```

**Impact:** If shuffle=True (default), future samples leak into training batches.

**Recommended Fix:**
Add `shuffle=False` to all `model.fit()` calls:
```python
history = self.model.fit(
    ...,
    validation_split=validation_split,
    shuffle=False,  # ADD THIS
    ...
)
```

---

### CRITICAL-3: fillna(method='ffill') After Rolling Windows
**File:** `/Users/willhatfield/ML-Stock_Trading/models/beta2.py`
**Lines:** 315-317

**Description:**
Technical indicators use forward-fill and back-fill for NaN values:

```python
# Line 317
df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
```

While `ffill` alone is fine, the combination with `bfill` creates look-ahead bias: `bfill` propagates future values backward in time. At the start of the dataset, the first N rows (where N is the rolling window size) will have their NaN values filled with data from the future.

**Impact:** Look-ahead bias in early samples, feature leakage.

**Recommended Fix:**
1. Remove `bfill()` entirely
2. Drop rows with NaN values at the beginning instead
3. Or fill with a neutral value (0 or column mean computed only on training data)

---

### CRITICAL-4: Target Variable Uses shift(-1) But No Alignment Protection
**File:** `/Users/willhatfield/ML-Stock_Trading/tests/stock_direction_trader.py`
**Line:** 125

**Description:**
The target variable is created correctly using `shift(-1)`:

```python
# Line 125
df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
```

However, `dropna()` at line 128 only removes NaN rows, but the LAST row's target is based on non-existent future data. If this row is included in training (before the final chronological split), it creates a data point with an invalid label.

**Impact:** Potential for including a sample with invalid/leaked target.

**Recommended Fix:**
After creating target, explicitly drop the last row:
```python
df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df = df.iloc[:-1]  # Remove last row with invalid target
df.dropna(inplace=True)
```

---

### CRITICAL-5: S&P 500 Alignment Uses Future Reference Price
**File:** `/Users/willhatfield/ML-Stock_Trading/models/beta2.py`
**Lines:** 174-184

**Description:**
The relative performance calculation uses the FIRST valid index to normalize, but the data may not be chronologically filtered:

```python
# Lines 175-184
first_valid_idx = indicators_df['SP500_Close'].first_valid_index()
if first_valid_idx:
    stock_start = indicators_df.loc[first_valid_idx, 'Close']
    sp500_start = indicators_df.loc[first_valid_idx, 'SP500_Close']

    if stock_start > 0 and sp500_start > 0:
        indicators_df['RelPerf'] = (indicators_df['Close'] / stock_start) / (
            indicators_df['SP500_Close'] / sp500_start)
```

If the S&P 500 data has a different date range or missing dates, the alignment could use prices from a different day. More critically, `RelPerf` is a cumulative metric that inherently contains information about the entire series.

**Impact:** Potential date misalignment and cumulative features contain global information.

**Recommended Fix:**
1. Verify strict date alignment between stock and S&P data
2. Calculate relative performance incrementally (daily relative returns) instead of cumulative
3. Or remove RelPerf feature entirely

---

### CRITICAL-6: No Walk-Forward Validation Implemented
**File:** `/Users/willhatfield/ML-Stock_Trading/models/beta2.py`, `/Users/willhatfield/ML-Stock_Trading/models/backtest.py`
**All files**

**Description:**
Neither file implements walk-forward (expanding window) validation. The model is trained once on all training data and tested once. This is problematic because:
1. A single train/test split may not represent realistic trading scenarios
2. The model doesn't adapt to regime changes
3. Performance metrics may be overly optimistic or pessimistic depending on the test period

**Impact:** Unrealistic performance estimates, no measure of model stability over time.

**Recommended Fix:**
Implement walk-forward validation:
```python
def walk_forward_validate(data, train_window, test_window, step_size):
    for start in range(0, len(data) - train_window - test_window, step_size):
        train = data[start:start + train_window]
        test = data[start + train_window:start + train_window + test_window]
        # Fit scaler on train only
        # Train model on train only
        # Evaluate on test
        # Collect metrics
```

---

### CRITICAL-7: Backtest Uses Overlapping Train/Test Data
**File:** `/Users/willhatfield/ML-Stock_Trading/models/backtest.py`
**Lines:** 224-229

**Description:**
The backtest concatenates training and test data for prediction inputs:

```python
# Lines 225-229
total_price_data = pd.concat((train_data['Close'], test_data['Close']), axis=0)
total_price_data = total_price_data[~total_price_data.index.duplicated(keep='first')]
price_inputs = total_price_data[len(total_price_data) - len(test_data) - predictor.prediction_days:].values
```

This approach is correct in principle (using historical data for prediction), but:
1. The scaler was fit on training data that overlaps with these price_inputs
2. No verification that duplicates aren't from the test period
3. The `keep='first'` could discard test data in favor of training data at overlapping dates

**Impact:** Subtle data leakage through duplicate handling.

**Recommended Fix:**
1. Ensure test data is strictly after training data with no overlap
2. Use `keep='last'` to prefer test data prices
3. Add explicit date range validation

---

### CRITICAL-8: Train/Test Split Not Strictly Chronological in StockDirectionTrader
**File:** `/Users/willhatfield/ML-Stock_Trading/tests/stock_direction_trader.py`
**Lines:** 629-631

**Description:**
The split is index-based but doesn't verify chronological ordering:

```python
# Lines 629-631
train_size = int(len(df) * train_ratio)
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]
```

If the DataFrame's index is not sorted by date (though unlikely with yfinance data), this could cause leakage.

**Impact:** Potential leakage if data is not chronologically sorted.

**Recommended Fix:**
Add explicit sort before split:
```python
df = df.sort_index()  # Ensure chronological order
train_size = int(len(df) * train_ratio)
```

---

## Moderate Issues (Should Fix)

### MODERATE-1: Rolling Windows Lack Explicit Burn-In Period
**File:** `/Users/willhatfield/ML-Stock_Trading/models/beta2.py`
**Lines:** 247-260, 274-288

**Description:**
Technical indicators like MA5, MA20, RSI (14-day), and Bollinger Bands create NaN values for the first N rows. While these are filled later (see CRITICAL-3), there's no explicit burn-in period that removes these unstable early values.

```python
# Lines 247-248
df['MA5'] = df['Close'].rolling(window=5).mean()
df['MA20'] = df['Close'].rolling(window=20).mean()
```

**Impact:** Early samples have low-quality/imputed features.

**Recommended Fix:**
After calculating all indicators, drop the first `max_window` rows:
```python
max_window = max(20, 14, 26)  # Largest window used
df = df.iloc[max_window:]
```

---

### MODERATE-2: OBV Uses Cumulative Sum (Global Information)
**File:** `/Users/willhatfield/ML-Stock_Trading/models/beta2.py`
**Lines:** 308-312

**Description:**
On-Balance Volume (OBV) uses cumulative sum:

```python
# Line 309
df['OBV'] = (df['Volume'] * (np.sign(df['Close'].diff()))).fillna(0).cumsum()
```

Cumulative features contain information about the entire history up to each point, which is technically valid. However, if the data is later shuffled or if samples are selected non-chronologically, OBV values from later periods could appear in earlier training batches.

**Impact:** Feature implicitly encodes temporal position; problematic if data is shuffled.

**Recommended Fix:**
Either:
1. Ensure data is never shuffled (already recommended)
2. Use OBV change rate instead of absolute OBV
3. Normalize OBV within sliding windows

---

### MODERATE-3: Feature Scaler in StockDirectionTrader Not Used
**File:** `/Users/willhatfield/ML-Stock_Trading/tests/stock_direction_trader.py`
**Lines:** 74, 140-166

**Description:**
A `StandardScaler` is initialized but never used in `extract_features()`:

```python
# Line 74
self.feature_scaler = StandardScaler() if StandardScaler else None

# Lines 140-166 - extract_features() never calls self.feature_scaler
```

**Impact:** Features are not normalized, which can hurt model performance. If scaling is added later improperly, it could introduce leakage.

**Recommended Fix:**
Either remove the unused scaler or implement proper scaling:
```python
def extract_features(self, df, fit_scaler=False):
    ...
    if fit_scaler:
        X_scaled = self.feature_scaler.fit_transform(X.reshape(-1, X.shape[-1]))
    else:
        X_scaled = self.feature_scaler.transform(X.reshape(-1, X.shape[-1]))
    return X_scaled.reshape(X.shape), y
```

---

### MODERATE-4: Prediction Window May Extend Into Test Period
**File:** `/Users/willhatfield/ML-Stock_Trading/models/beta2.py`
**Lines:** 1146-1147

**Description:**
In `main()`, data preparation uses all data together:

```python
# Line 1146
x_price_train, x_features_train, y_train = predictor.prepare_data(price_data, feature_data)
```

The `prepare_data()` function creates sequences using a sliding window. The last `prediction_days` of training data overlap with what should be test data.

**Impact:** Last training samples may peek into test period.

**Recommended Fix:**
Separate train and test data BEFORE calling `prepare_data()`.

---

### MODERATE-5: Sample Weights Bias Toward Recent Data
**File:** `/Users/willhatfield/ML-Stock_Trading/models/beta2.py`
**Lines:** 542-544

**Description:**
Sample weights are set to bias recent data:

```python
# Lines 543-544
if len(y) > 100:
    sample_weights = np.linspace(0.5, 1.0, len(y))
```

While this is a valid technique, if the validation split takes the last 20% of data, those samples will have higher weights during training AND be used for validation. This creates an indirect form of validation leakage.

**Impact:** Validation performance may be artificially inflated.

**Recommended Fix:**
Apply sample weights only to training samples, not validation samples. Use a custom training loop or separate train/val before applying weights.

---

## Minor Issues (Nice to Have)

### MINOR-1: Hardcoded Current Date
**File:** `/Users/willhatfield/ML-Stock_Trading/models/beta2.py`
**Lines:** 1120-1121

**Description:**
The current date is hardcoded:

```python
# Line 1120
current_date = datetime(2025, 3, 23).replace(tzinfo=None)
```

**Impact:** May cause confusion when running at different times; not technically leakage.

**Recommended Fix:**
Use `datetime.now()` or make it configurable.

---

### MINOR-2: Random Seed Not Set for Model Training
**File:** `/Users/willhatfield/ML-Stock_Trading/models/beta2.py`
**All files**

**Description:**
TensorFlow/Keras random seeds are not set, leading to non-reproducible results.

**Impact:** Cannot reproduce exact results for debugging.

**Recommended Fix:**
Add at the top of training:
```python
import tensorflow as tf
tf.random.set_seed(42)
np.random.seed(42)
```

Note: `stock_direction_trader.py` does set seeds at lines 23-25, which is good practice.

---

### MINOR-3: Debug Mode Adds Randomness to Predictions
**File:** `/Users/willhatfield/ML-Stock_Trading/tests/stock_direction_trader.py`
**Lines:** 312-323

**Description:**
The `predict_direction()` method previously had commented-out code that added random noise to predictions. The current version is clean, but any future modifications should avoid adding randomness:

```python
# Line 313
pred_prob = self.model.predict(features, verbose=0)
confidence = float(pred_prob[0][0])
signal = int(confidence > 0.5)
```

**Impact:** None currently, but worth monitoring.

---

## Summary Table

| ID | Severity | Issue | File | Lines |
|----|----------|-------|------|-------|
| CRITICAL-1 | Critical | Scaler fit on full data | beta2.py | 349-350, 373-377 |
| CRITICAL-2 | Critical | Keras validation_split shuffles | beta2.py | 548-566 |
| CRITICAL-3 | Critical | bfill causes look-ahead | beta2.py | 315-317 |
| CRITICAL-4 | Critical | Last row has invalid target | stock_direction_trader.py | 125-128 |
| CRITICAL-5 | Critical | S&P alignment issues | beta2.py | 174-184 |
| CRITICAL-6 | Critical | No walk-forward validation | All files | - |
| CRITICAL-7 | Critical | Backtest data overlap | backtest.py | 224-229 |
| CRITICAL-8 | Critical | Split not verified chronological | stock_direction_trader.py | 629-631 |
| MODERATE-1 | Moderate | No burn-in period | beta2.py | 247-260 |
| MODERATE-2 | Moderate | OBV cumulative feature | beta2.py | 308-312 |
| MODERATE-3 | Moderate | Unused scaler | stock_direction_trader.py | 74 |
| MODERATE-4 | Moderate | Prediction window overlap | beta2.py | 1146-1147 |
| MODERATE-5 | Moderate | Sample weights + validation | beta2.py | 542-544 |
| MINOR-1 | Minor | Hardcoded date | beta2.py | 1120-1121 |
| MINOR-2 | Minor | No random seed | beta2.py | - |
| MINOR-3 | Minor | Potential future randomness | stock_direction_trader.py | 312-323 |

---

## Recommended Action Plan

1. **Immediate (Before Any Further Backtests):**
   - Fix CRITICAL-1 through CRITICAL-3 (scaler fitting, shuffle=False, remove bfill)
   - These alone invalidate all current backtest results

2. **Short-term (Within 1 Week):**
   - Fix remaining critical issues (CRITICAL-4 through CRITICAL-8)
   - Implement proper chronological train/validation/test splits
   - Add explicit burn-in period removal

3. **Medium-term (Within 1 Month):**
   - Implement walk-forward validation
   - Address all moderate issues
   - Re-run all backtests with corrected pipeline

4. **Long-term (Ongoing):**
   - Add unit tests for data pipeline correctness
   - Add assertions to verify no future data in training
   - Document data pipeline clearly

---

## Conclusion

The current codebase has significant data leakage issues that make all backtesting results unreliable. The primary issues stem from fitting scalers on future data, potential shuffling during training, and improper NaN handling. The `stock_direction_trader.py` file shows better practices (e.g., `shuffle=False` in `train_model()`, random seeds set) and could serve as a reference for fixing the main `beta2.py` model.

**All performance metrics and backtests should be considered invalid until these issues are resolved.**
