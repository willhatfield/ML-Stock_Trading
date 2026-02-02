# Strategy Proposals: Mathematically Grounded Baselines

This document proposes mathematically grounded strategy families that serve as baselines before considering ML approaches. These strategies have known theoretical properties, are easy to implement, and provide benchmarks against which LSTM and other ML models must prove their worth.

---

## Table of Contents
1. [Momentum/Trend-Following Strategy Family](#1-momentumtrend-following-strategy-family)
2. [Mean-Reversion Strategy Family](#2-mean-reversion-strategy-family)
3. [Unified Interface Specification](#3-unified-interface-specification)
4. [Objective Functions](#4-objective-functions)
5. [Risk Controls](#5-risk-controls)

---

## 1. Momentum/Trend-Following Strategy Family

### Mathematical Foundation

**Time-Series Momentum (TSMOM)** is grounded in the empirical observation that assets exhibiting positive (negative) returns over the past 12 months tend to continue generating positive (negative) returns over the next month.

**Key Academic References:**
- Moskowitz, Ooi, and Pedersen (2012). "Time Series Momentum." *Journal of Financial Economics*, 104(2), 228-250.
- Jegadeesh and Titman (1993). "Returns to Buying Winners and Selling Losers." *Journal of Finance*, 48(1), 65-91.
- Asness, Moskowitz, and Pedersen (2013). "Value and Momentum Everywhere." *Journal of Finance*, 68(3), 929-985.

**Theoretical Basis:**
The persistence of trends can be attributed to:
1. **Behavioral biases**: Under-reaction to new information, anchoring, disposition effect
2. **Slow information diffusion**: Gradual price discovery
3. **Risk-based explanations**: Compensation for crash risk (momentum crashes)

### Specific Implementation: Volatility-Targeted Time-Series Momentum

#### Signal Generation

The core signal is based on the sign and magnitude of past returns, scaled by inverse volatility:

```
r_t = (P_t - P_{t-L}) / P_{t-L}                    # L-period return (typically L=252 for 12 months)

sigma_t = sqrt(sum_{i=1}^{N} (r_{t-i} - r_bar)^2 / (N-1))   # Realized volatility (N=20-60 days)

signal_t = sign(r_t) * min(1, |r_t| / sigma_t)     # Bounded normalized signal
```

**Lookback Periods:**
- Formation period: 252 days (12 months) - excludes last month to avoid reversal
- Skip period: 21 days (1 month) - accounts for short-term reversal
- Volatility window: 60 days (rolling)

#### Position Sizing Rules

**Volatility Targeting:**
```
target_vol = 0.15                                   # 15% annualized target volatility
current_vol = sigma_t * sqrt(252)                  # Annualized realized volatility

position_size = signal_t * (target_vol / current_vol)
position_size = clip(position_size, -1, 1)         # Bound to [-1, 1]
```

**Full Kelly and Fractional Kelly:**
```
# Estimated Sharpe from historical data
SR_est = mean(excess_returns) / std(excess_returns) * sqrt(252)

# Full Kelly fraction
f_kelly = SR_est / sigma_t^2

# Half-Kelly (more conservative)
f_half_kelly = 0.5 * f_kelly

position_size = min(position_size, f_half_kelly)
```

#### Entry/Exit Rules

```python
def generate_momentum_signal(prices: pd.Series,
                             formation_period: int = 252,
                             skip_period: int = 21,
                             vol_window: int = 60) -> float:
    """
    Generate momentum signal with skip period.

    Args:
        prices: Historical price series
        formation_period: Lookback period for return calculation
        skip_period: Days to skip (avoid short-term reversal)
        vol_window: Window for volatility estimation

    Returns:
        Signal between -1 and 1
    """
    if len(prices) < formation_period + skip_period:
        return 0.0

    # Calculate return excluding skip period
    p_now = prices.iloc[-skip_period-1]
    p_past = prices.iloc[-formation_period-skip_period]
    momentum_return = (p_now - p_past) / p_past

    # Calculate realized volatility
    returns = prices.pct_change().dropna()
    volatility = returns.iloc[-vol_window:].std()

    # Normalize signal
    if volatility > 0:
        raw_signal = momentum_return / (volatility * np.sqrt(formation_period))
        signal = np.sign(raw_signal) * min(1.0, abs(raw_signal))
    else:
        signal = 0.0

    return signal
```

**Entry Rules:**
- Enter long when `signal > entry_threshold` (default: 0.2)
- Enter short when `signal < -entry_threshold`
- Minimum holding period: 21 trading days

**Exit Rules:**
- Exit when signal crosses zero OR
- Stop-loss: 2x ATR from entry price OR
- Trailing stop: 3x ATR from peak since entry OR
- Maximum holding period: 63 days (3 months)

### Risk Controls

```python
# Exposure caps
MAX_GROSS_EXPOSURE = 1.0        # 100% of capital
MAX_NET_EXPOSURE = 0.5          # 50% long/short bias
MAX_SINGLE_POSITION = 0.10      # 10% per position

# Drawdown controls
MAX_DRAWDOWN_DAILY = 0.03       # 3% daily loss triggers reduction
MAX_DRAWDOWN_MONTHLY = 0.10     # 10% monthly drawdown triggers stop
RECOVERY_PERIOD = 5             # Days before re-entry after stop

# Position reduction on drawdown
def adjust_for_drawdown(position: float, current_drawdown: float) -> float:
    if current_drawdown > 0.05:  # 5% drawdown
        return position * 0.5    # Reduce to half
    if current_drawdown > 0.10:  # 10% drawdown
        return 0.0               # Exit completely
    return position
```

### Expected Characteristics

| Metric | Expected Value | Notes |
|--------|----------------|-------|
| Sharpe Ratio | 0.4 - 0.8 | Lower in recent years due to crowding |
| Annual Turnover | 200-400% | Monthly rebalancing |
| Win Rate | 40-45% | Relies on large gains to offset frequent small losses |
| Avg Holding Period | 1-3 months | |
| Max Drawdown | 20-40% | Can experience momentum crashes |
| Capacity | High | Works across asset classes |

### Parameters Requiring Calibration

| Parameter | Default | Range | Calibration Method |
|-----------|---------|-------|-------------------|
| `formation_period` | 252 | 60-504 | Walk-forward optimization |
| `skip_period` | 21 | 0-42 | Cross-validation |
| `vol_window` | 60 | 20-120 | Stability analysis |
| `target_volatility` | 0.15 | 0.05-0.25 | Risk preference |
| `entry_threshold` | 0.2 | 0.1-0.5 | Transaction cost analysis |
| `stop_loss_atr_mult` | 2.0 | 1.0-3.0 | Historical drawdown analysis |

---

## 2. Mean-Reversion Strategy Family

### Mathematical Foundation

**Mean Reversion** is based on the Ornstein-Uhlenbeck (OU) process, which models prices as reverting to a long-run mean:

```
dX_t = theta * (mu - X_t) * dt + sigma * dW_t
```

Where:
- `theta`: Speed of mean reversion (half-life = ln(2)/theta)
- `mu`: Long-run mean
- `sigma`: Volatility
- `W_t`: Wiener process (Brownian motion)

**Key Academic References:**
- Poterba and Summers (1988). "Mean Reversion in Stock Prices: Evidence and Implications." *Journal of Financial Economics*, 22(1), 27-59.
- Lo and MacKinlay (1988). "Stock Market Prices Do Not Follow Random Walks." *Review of Financial Studies*, 1(1), 41-66.
- Avellaneda and Lee (2010). "Statistical Arbitrage in the US Equities Market." *Quantitative Finance*, 10(7), 761-782.

**Theoretical Basis:**
1. **Market microstructure**: Bid-ask bounce, inventory effects
2. **Behavioral overreaction**: Price overshoots followed by correction
3. **Liquidity provision**: Compensation for providing liquidity

### Specific Implementation: Z-Score Mean Reversion with Bollinger Bands

#### Signal Generation

```
# Calculate z-score of price relative to moving average
MA_t = (1/N) * sum_{i=0}^{N-1} P_{t-i}           # N-period moving average (typically N=20)
sigma_t = sqrt((1/N) * sum_{i=0}^{N-1} (P_{t-i} - MA_t)^2)  # Rolling standard deviation

z_t = (P_t - MA_t) / sigma_t                      # Z-score

# Generate signal (contrarian)
signal_t = -z_t / z_threshold                     # z_threshold typically 2
signal_t = clip(signal_t, -1, 1)
```

**Bollinger Band Implementation:**
```
upper_band = MA_t + k * sigma_t                   # k = 2 for 95% confidence
lower_band = MA_t - k * sigma_t
%b = (P_t - lower_band) / (upper_band - lower_band)

# Mean reversion signal
signal = 1 - 2 * %b                               # +1 at lower band, -1 at upper band
```

#### Half-Life Estimation (OU Process)

To verify mean-reversion exists and estimate optimal holding period:

```python
def estimate_half_life(prices: pd.Series) -> float:
    """
    Estimate mean-reversion half-life using OLS regression.

    Model: delta_y = lambda * y_{t-1} + epsilon
    Half-life = -ln(2) / lambda
    """
    y = np.log(prices)
    y_lag = y.shift(1)
    delta_y = y - y_lag

    # Remove NaN
    y_lag = y_lag.dropna()
    delta_y = delta_y.dropna()

    # OLS regression
    X = sm.add_constant(y_lag)
    model = sm.OLS(delta_y, X).fit()
    lambda_coef = model.params[1]

    if lambda_coef >= 0:
        return np.inf  # No mean reversion

    half_life = -np.log(2) / lambda_coef
    return half_life
```

#### Position Sizing Rules

**Z-Score Based Sizing:**
```python
def calculate_position_size(z_score: float,
                           max_position: float = 1.0,
                           z_threshold: float = 2.0) -> float:
    """
    Scale position inversely with z-score (buy low, sell high).
    """
    if abs(z_score) < 0.5:  # Dead zone - no trade
        return 0.0

    # Linear scaling
    raw_size = -z_score / z_threshold

    # Apply bounds
    return np.clip(raw_size, -max_position, max_position)
```

**Volatility-Adjusted Sizing:**
```python
def vol_adjusted_size(z_score: float,
                      current_vol: float,
                      target_vol: float = 0.10) -> float:
    """
    Adjust position for current volatility regime.
    """
    base_size = calculate_position_size(z_score)
    vol_scalar = target_vol / (current_vol * np.sqrt(252))
    return base_size * min(vol_scalar, 2.0)  # Cap at 2x leverage
```

#### Entry/Exit Rules

```python
class MeanReversionStrategy:
    def __init__(self,
                 lookback: int = 20,
                 entry_z: float = 2.0,
                 exit_z: float = 0.5,
                 max_holding: int = 10):
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.max_holding = max_holding

    def generate_signal(self, prices: pd.Series) -> float:
        """Generate mean-reversion signal."""
        if len(prices) < self.lookback:
            return 0.0

        recent = prices.iloc[-self.lookback:]
        ma = recent.mean()
        std = recent.std()

        if std == 0:
            return 0.0

        z_score = (prices.iloc[-1] - ma) / std

        # Entry logic
        if z_score > self.entry_z:
            return -1.0  # Short (expect reversion down)
        elif z_score < -self.entry_z:
            return 1.0   # Long (expect reversion up)
        elif abs(z_score) < self.exit_z:
            return 0.0   # Exit zone
        else:
            return np.nan  # Hold current position
```

**Entry Rules:**
- Enter long when z-score < -2.0 (price 2 std below mean)
- Enter short when z-score > 2.0 (price 2 std above mean)
- Confirm with half-life < 20 days (mean reversion exists)

**Exit Rules:**
- Exit when z-score crosses 0 (price returns to mean) OR
- Exit when z-score crosses entry threshold in opposite direction (stop-loss) OR
- Time-based exit: position held > 2 * half_life days OR
- Maximum holding period: 10 trading days

### Risk Controls

```python
# Mean reversion specific risks
MAX_SINGLE_POSITION = 0.05      # 5% per position (tighter than momentum)
MAX_CORRELATED_EXPOSURE = 0.15  # 15% in correlated trades

# Stop-loss (critical for mean-reversion)
STOP_LOSS_Z = 3.5               # Exit if z-score reaches 3.5
STOP_LOSS_PCT = 0.05            # Or 5% loss from entry

# Time decay
def time_decay_factor(days_held: int, half_life: float) -> float:
    """Reduce position as time exceeds expected holding."""
    expected_hold = 2 * half_life
    if days_held > expected_hold:
        return max(0.2, 1 - (days_held - expected_hold) / expected_hold)
    return 1.0
```

### Expected Characteristics

| Metric | Expected Value | Notes |
|--------|----------------|-------|
| Sharpe Ratio | 0.5 - 1.0 | Higher than momentum in ranging markets |
| Annual Turnover | 1000-2000% | High frequency rebalancing |
| Win Rate | 55-65% | Many small wins |
| Avg Holding Period | 2-10 days | Short-term trades |
| Max Drawdown | 10-20% | Smaller but more frequent |
| Capacity | Low-Medium | Limited by liquidity and speed |

### Parameters Requiring Calibration

| Parameter | Default | Range | Calibration Method |
|-----------|---------|-------|-------------------|
| `lookback` | 20 | 5-60 | Half-life estimation |
| `entry_z` | 2.0 | 1.5-3.0 | Historical distribution analysis |
| `exit_z` | 0.5 | 0.0-1.0 | Transaction cost optimization |
| `max_holding` | 10 | 5-30 | 2x estimated half-life |
| `stop_loss_z` | 3.5 | 2.5-4.0 | Tail risk analysis |
| `bb_period` | 20 | 10-50 | Regime detection |
| `bb_std` | 2.0 | 1.5-2.5 | Entry frequency tuning |

---

## 3. Unified Interface Specification

All strategies (momentum, mean-reversion, and future LSTM models) must implement this interface:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np


class StrategyType(Enum):
    """Classification of strategy type for risk management."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    STATISTICAL = "statistical"
    ML_BASED = "ml_based"


@dataclass
class StrategyConfig:
    """Configuration parameters for strategy initialization."""
    name: str
    strategy_type: StrategyType
    parameters: Dict[str, Any]
    risk_limits: Dict[str, float]

    # Default risk limits
    max_position: float = 1.0
    max_drawdown: float = 0.20
    target_volatility: float = 0.15


@dataclass
class Signal:
    """Output of signal generation."""
    timestamp: pd.Timestamp
    symbol: str
    direction: float          # [-1, 1] where -1 is max short, +1 is max long
    confidence: float         # [0, 1] confidence in signal
    raw_score: float          # Unbounded raw signal before normalization
    metadata: Dict[str, Any]  # Strategy-specific information


@dataclass
class Position:
    """Current position information."""
    symbol: str
    size: float               # Signed position size
    entry_price: float
    entry_time: pd.Timestamp
    current_price: float
    unrealized_pnl: float
    holding_days: int


class Strategy(ABC):
    """
    Abstract base class for all trading strategies.

    All strategies must implement this interface to ensure:
    1. Consistent signal generation
    2. Comparable performance metrics
    3. Unified risk management
    4. Reproducible results
    """

    def __init__(self, config: StrategyConfig):
        """
        Initialize strategy with configuration.

        Args:
            config: Strategy configuration including parameters and risk limits
        """
        self.config = config
        self.name = config.name
        self.strategy_type = config.strategy_type
        self._is_fitted = False
        self._last_signal: Optional[Signal] = None

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'Strategy':
        """
        Fit strategy parameters to historical data.

        For rule-based strategies, this may just validate data.
        For ML strategies, this trains the model.

        Args:
            data: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
                  and DatetimeIndex

        Returns:
            self for method chaining

        Raises:
            ValueError: If data is insufficient or invalid
        """
        pass

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        """
        Generate trading signal from current market data.

        Args:
            data: DataFrame with OHLCV data. Most recent data at the end.
                  Must have at least `min_history` rows.

        Returns:
            Signal object with direction in [-1, 1]

        Raises:
            ValueError: If strategy not fitted or data insufficient
        """
        pass

    @abstractmethod
    def calculate_position_size(self,
                                signal: Signal,
                                portfolio_value: float,
                                current_positions: List[Position]) -> float:
        """
        Calculate position size given signal and portfolio state.

        Args:
            signal: Signal from generate_signal()
            portfolio_value: Current total portfolio value
            current_positions: List of current positions

        Returns:
            Target position size in dollars (signed)
        """
        pass

    @property
    @abstractmethod
    def min_history(self) -> int:
        """
        Minimum number of historical bars required for signal generation.

        Returns:
            Integer number of bars
        """
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """
        Get current strategy parameters.

        Returns:
            Dictionary of parameter names and values
        """
        pass

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data meets requirements.

        Args:
            data: Input DataFrame

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [c for c in required_columns if c not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if len(data) < self.min_history:
            raise ValueError(
                f"Insufficient data: {len(data)} rows, need {self.min_history}"
            )

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")

        if data.isnull().any().any():
            raise ValueError("Data contains NaN values")

        return True

    def get_metrics(self,
                   returns: pd.Series,
                   benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculate strategy performance metrics.

        Args:
            returns: Strategy returns series
            benchmark_returns: Optional benchmark returns for comparison

        Returns:
            Dictionary of metric names and values
        """
        metrics = {}

        # Basic metrics
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annual_return'] = (1 + metrics['total_return']) ** (252 / len(returns)) - 1
        metrics['volatility'] = returns.std() * np.sqrt(252)
        metrics['sharpe_ratio'] = metrics['annual_return'] / metrics['volatility'] if metrics['volatility'] > 0 else 0

        # Downside metrics
        downside_returns = returns[returns < 0]
        metrics['downside_vol'] = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        metrics['sortino_ratio'] = metrics['annual_return'] / metrics['downside_vol'] if metrics['downside_vol'] > 0 else 0

        # Drawdown metrics
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        metrics['max_drawdown'] = drawdowns.min()
        metrics['calmar_ratio'] = metrics['annual_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0

        # Win/loss metrics
        metrics['win_rate'] = (returns > 0).mean()
        metrics['avg_win'] = returns[returns > 0].mean() if (returns > 0).any() else 0
        metrics['avg_loss'] = returns[returns < 0].mean() if (returns < 0).any() else 0
        metrics['profit_factor'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else np.inf

        # Benchmark comparison
        if benchmark_returns is not None:
            excess_returns = returns - benchmark_returns
            tracking_error = excess_returns.std() * np.sqrt(252)
            metrics['information_ratio'] = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
            metrics['beta'] = returns.cov(benchmark_returns) / benchmark_returns.var() if benchmark_returns.var() > 0 else 0
            metrics['alpha'] = metrics['annual_return'] - metrics['beta'] * (benchmark_returns.mean() * 252)

        return metrics

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type={self.strategy_type.value})"


class MomentumStrategy(Strategy):
    """
    Time-series momentum strategy implementation.

    Example usage:
        config = StrategyConfig(
            name="TSMOM_12M",
            strategy_type=StrategyType.MOMENTUM,
            parameters={'formation_period': 252, 'skip_period': 21},
            risk_limits={'max_position': 1.0}
        )
        strategy = MomentumStrategy(config)
        strategy.fit(historical_data)
        signal = strategy.generate_signal(current_data)
    """

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.formation_period = config.parameters.get('formation_period', 252)
        self.skip_period = config.parameters.get('skip_period', 21)
        self.vol_window = config.parameters.get('vol_window', 60)
        self.target_vol = config.risk_limits.get('target_volatility', 0.15)

    @property
    def min_history(self) -> int:
        return self.formation_period + self.skip_period + self.vol_window

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            'formation_period': self.formation_period,
            'skip_period': self.skip_period,
            'vol_window': self.vol_window,
            'target_vol': self.target_vol
        }

    def fit(self, data: pd.DataFrame) -> 'MomentumStrategy':
        self.validate_data(data)
        self._is_fitted = True
        return self

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        if not self._is_fitted:
            raise ValueError("Strategy must be fitted before generating signals")

        self.validate_data(data)
        prices = data['Close']

        # Calculate momentum return (excluding skip period)
        p_now = prices.iloc[-self.skip_period - 1]
        p_past = prices.iloc[-self.formation_period - self.skip_period]
        momentum_return = (p_now - p_past) / p_past

        # Calculate realized volatility
        returns = prices.pct_change().dropna()
        volatility = returns.iloc[-self.vol_window:].std() * np.sqrt(252)

        # Normalize signal
        if volatility > 0:
            raw_score = momentum_return / volatility
            direction = np.sign(raw_score) * min(1.0, abs(raw_score))
        else:
            raw_score = 0.0
            direction = 0.0

        # Confidence based on signal strength
        confidence = min(1.0, abs(raw_score) / 2.0)

        signal = Signal(
            timestamp=data.index[-1],
            symbol=data.attrs.get('symbol', 'UNKNOWN'),
            direction=direction,
            confidence=confidence,
            raw_score=raw_score,
            metadata={
                'momentum_return': momentum_return,
                'volatility': volatility,
                'formation_period': self.formation_period
            }
        )

        self._last_signal = signal
        return signal

    def calculate_position_size(self,
                               signal: Signal,
                               portfolio_value: float,
                               current_positions: List[Position]) -> float:
        # Volatility targeting
        current_vol = signal.metadata.get('volatility', self.target_vol)
        vol_scalar = self.target_vol / current_vol if current_vol > 0 else 1.0
        vol_scalar = min(vol_scalar, 2.0)  # Cap at 2x

        # Base position
        base_size = signal.direction * portfolio_value * self.config.max_position

        # Apply volatility scaling
        target_size = base_size * vol_scalar

        # Apply confidence adjustment
        target_size *= signal.confidence

        return target_size


class MeanReversionStrategy(Strategy):
    """
    Z-score based mean reversion strategy implementation.
    """

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.lookback = config.parameters.get('lookback', 20)
        self.entry_z = config.parameters.get('entry_z', 2.0)
        self.exit_z = config.parameters.get('exit_z', 0.5)
        self.half_life = None  # Estimated during fit

    @property
    def min_history(self) -> int:
        return max(self.lookback * 3, 60)  # Need enough for half-life estimation

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            'lookback': self.lookback,
            'entry_z': self.entry_z,
            'exit_z': self.exit_z,
            'half_life': self.half_life
        }

    def fit(self, data: pd.DataFrame) -> 'MeanReversionStrategy':
        self.validate_data(data)

        # Estimate half-life
        prices = data['Close']
        self.half_life = self._estimate_half_life(prices)

        self._is_fitted = True
        return self

    def _estimate_half_life(self, prices: pd.Series) -> float:
        """Estimate mean-reversion half-life using OLS."""
        y = np.log(prices)
        y_lag = y.shift(1)
        delta_y = y - y_lag

        # Remove NaN
        valid_idx = ~(y_lag.isna() | delta_y.isna())
        y_lag_clean = y_lag[valid_idx].values.reshape(-1, 1)
        delta_y_clean = delta_y[valid_idx].values

        # OLS regression: delta_y = alpha + lambda * y_lag
        X = np.column_stack([np.ones(len(y_lag_clean)), y_lag_clean])
        coeffs = np.linalg.lstsq(X, delta_y_clean, rcond=None)[0]
        lambda_coef = coeffs[1]

        if lambda_coef >= 0:
            return np.inf  # No mean reversion

        return -np.log(2) / lambda_coef

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        if not self._is_fitted:
            raise ValueError("Strategy must be fitted before generating signals")

        self.validate_data(data)
        prices = data['Close']

        # Calculate z-score
        recent = prices.iloc[-self.lookback:]
        ma = recent.mean()
        std = recent.std()

        current_price = prices.iloc[-1]
        z_score = (current_price - ma) / std if std > 0 else 0

        # Generate contrarian signal
        if z_score > self.entry_z:
            direction = -1.0  # Short (expect reversion down)
        elif z_score < -self.entry_z:
            direction = 1.0   # Long (expect reversion up)
        elif abs(z_score) < self.exit_z:
            direction = 0.0   # Exit zone
        else:
            # Proportional signal in between
            direction = -z_score / self.entry_z
            direction = np.clip(direction, -1.0, 1.0)

        # Confidence based on distance from mean
        confidence = min(1.0, abs(z_score) / self.entry_z)

        signal = Signal(
            timestamp=data.index[-1],
            symbol=data.attrs.get('symbol', 'UNKNOWN'),
            direction=direction,
            confidence=confidence,
            raw_score=-z_score,  # Negative because contrarian
            metadata={
                'z_score': z_score,
                'moving_average': ma,
                'std_dev': std,
                'half_life': self.half_life,
                'lookback': self.lookback
            }
        )

        self._last_signal = signal
        return signal

    def calculate_position_size(self,
                               signal: Signal,
                               portfolio_value: float,
                               current_positions: List[Position]) -> float:
        # More conservative sizing for mean reversion
        z_score = abs(signal.metadata.get('z_score', 0))

        # Scale up position as z-score increases (stronger reversion expected)
        if z_score < 1.5:
            size_scalar = 0.25
        elif z_score < 2.0:
            size_scalar = 0.5
        elif z_score < 2.5:
            size_scalar = 0.75
        else:
            size_scalar = 1.0

        base_size = signal.direction * portfolio_value * self.config.max_position
        return base_size * size_scalar * signal.confidence
```

### LSTM Integration Example

```python
class LSTMStrategy(Strategy):
    """
    LSTM-based prediction strategy.
    Must beat baseline strategies to justify complexity.
    """

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.prediction_days = config.parameters.get('prediction_days', 60)
        self.model = None
        self.scaler = None

    @property
    def min_history(self) -> int:
        return self.prediction_days * 2

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            'prediction_days': self.prediction_days,
            'model_type': self.config.parameters.get('model_type', 'lstm')
        }

    def fit(self, data: pd.DataFrame) -> 'LSTMStrategy':
        # Delegate to existing StockPredictor
        from models.beta2 import StockPredictor

        self.validate_data(data)

        predictor = StockPredictor(
            prediction_days=self.prediction_days,
            model_type=self.config.parameters.get('model_type', 'lstm')
        )

        # Training logic...
        self._is_fitted = True
        return self

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        # Use model prediction to generate directional signal
        # Implementation would call predictor.predict_next_day()
        pass

    def calculate_position_size(self,
                               signal: Signal,
                               portfolio_value: float,
                               current_positions: List[Position]) -> float:
        # Similar to momentum but potentially more aggressive
        # if model shows high confidence
        pass
```

---

## 4. Objective Functions

### Primary Objective: Risk-Adjusted Returns

**Sharpe Ratio** (default objective):
```
SR = (E[R_p] - R_f) / sigma_p

Where:
- E[R_p] = Expected portfolio return (annualized)
- R_f = Risk-free rate (typically 0 or T-bill rate)
- sigma_p = Portfolio volatility (annualized)
```

**Sortino Ratio** (penalizes only downside risk):
```
Sortino = (E[R_p] - R_f) / sigma_d

Where:
- sigma_d = Downside deviation (std of returns below target)
```

**Calmar Ratio** (return per unit of drawdown risk):
```
Calmar = CAGR / |Max Drawdown|
```

### Secondary Objectives (Constraints)

**Drawdown Constraints:**
```python
@dataclass
class DrawdownConstraints:
    max_drawdown: float = 0.20        # Maximum acceptable drawdown
    max_drawdown_duration: int = 60   # Max days in drawdown

    def is_violated(self, cumulative_returns: pd.Series) -> bool:
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak

        if drawdown.min() < -self.max_drawdown:
            return True

        # Check duration
        in_drawdown = drawdown < 0
        drawdown_periods = in_drawdown.groupby((~in_drawdown).cumsum()).cumsum()

        if drawdown_periods.max() > self.max_drawdown_duration:
            return True

        return False
```

**Turnover Constraints:**
```python
@dataclass
class TurnoverConstraints:
    max_annual_turnover: float = 10.0   # 1000%
    min_holding_period: int = 1         # Minimum days between trades

    def calculate_turnover(self, positions: pd.DataFrame) -> float:
        """
        Calculate annualized turnover.

        Turnover = sum(|position_changes|) / avg_portfolio_value
        """
        position_changes = positions.diff().abs().sum(axis=1)
        portfolio_value = positions.abs().sum(axis=1)

        daily_turnover = position_changes / portfolio_value
        annual_turnover = daily_turnover.mean() * 252

        return annual_turnover
```

**Information Ratio** (relative to benchmark):
```
IR = (R_p - R_b) / TE

Where:
- R_p = Portfolio return
- R_b = Benchmark return
- TE = Tracking error (std of excess returns)
```

### Multi-Objective Optimization

```python
def objective_function(returns: pd.Series,
                       benchmark_returns: pd.Series,
                       weights: Dict[str, float] = None) -> float:
    """
    Combined objective function for strategy optimization.

    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns
        weights: Weights for each component (default: equal)

    Returns:
        Combined objective score (higher is better)
    """
    if weights is None:
        weights = {
            'sharpe': 0.4,
            'sortino': 0.2,
            'calmar': 0.2,
            'information_ratio': 0.2
        }

    # Calculate metrics
    excess_returns = returns - benchmark_returns

    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    downside = returns[returns < 0].std()
    sortino = returns.mean() / downside * np.sqrt(252) if downside > 0 else 0

    cum_returns = (1 + returns).cumprod()
    max_dd = (cum_returns / cum_returns.expanding().max() - 1).min()
    calmar = (cum_returns.iloc[-1] ** (252/len(returns)) - 1) / abs(max_dd) if max_dd != 0 else 0

    tracking_error = excess_returns.std() * np.sqrt(252)
    ir = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0

    # Combine with weights
    score = (
        weights['sharpe'] * sharpe +
        weights['sortino'] * sortino +
        weights['calmar'] * calmar +
        weights['information_ratio'] * ir
    )

    return score
```

### Penalty Functions

```python
def apply_penalties(base_score: float,
                    returns: pd.Series,
                    turnover: float,
                    max_turnover: float = 10.0,
                    max_drawdown: float = 0.20) -> float:
    """
    Apply penalties for constraint violations.
    """
    penalty = 0.0

    # Turnover penalty (quadratic beyond threshold)
    if turnover > max_turnover:
        excess_turnover = turnover - max_turnover
        penalty += 0.1 * excess_turnover ** 2

    # Drawdown penalty (severe)
    cum_returns = (1 + returns).cumprod()
    actual_max_dd = abs((cum_returns / cum_returns.expanding().max() - 1).min())

    if actual_max_dd > max_drawdown:
        excess_dd = actual_max_dd - max_drawdown
        penalty += 1.0 * excess_dd ** 2  # Harsh penalty

    return base_score - penalty
```

---

## 5. Risk Controls

### Position Sizing Methods

#### 1. Fixed Fractional

```python
def fixed_fractional_size(portfolio_value: float,
                          risk_fraction: float = 0.02,
                          stop_loss_pct: float = 0.05) -> float:
    """
    Risk a fixed fraction of portfolio on each trade.

    Position = (Portfolio * Risk_Fraction) / Stop_Loss_Pct
    """
    dollar_risk = portfolio_value * risk_fraction
    position_size = dollar_risk / stop_loss_pct
    return min(position_size, portfolio_value)  # Cap at 100%
```

#### 2. Volatility Targeting

```python
def volatility_target_size(portfolio_value: float,
                          target_vol: float,
                          asset_vol: float,
                          signal: float) -> float:
    """
    Size position to achieve target portfolio volatility.

    Position = (Target_Vol / Asset_Vol) * Portfolio * Signal
    """
    if asset_vol <= 0:
        return 0.0

    vol_scalar = target_vol / asset_vol
    vol_scalar = min(vol_scalar, 3.0)  # Cap leverage at 3x

    return portfolio_value * vol_scalar * signal
```

#### 3. Kelly Criterion

```python
def kelly_position_size(win_rate: float,
                       avg_win: float,
                       avg_loss: float,
                       kelly_fraction: float = 0.5) -> float:
    """
    Kelly criterion position sizing.

    Full Kelly: f* = (p * b - q) / b
    Where:
        p = probability of winning
        q = probability of losing (1 - p)
        b = win/loss ratio

    Args:
        win_rate: Historical win rate
        avg_win: Average winning trade return
        avg_loss: Average losing trade return (positive)
        kelly_fraction: Fraction of full Kelly (0.5 = half Kelly)
    """
    p = win_rate
    q = 1 - p
    b = avg_win / avg_loss if avg_loss > 0 else 1.0

    full_kelly = (p * b - q) / b if b > 0 else 0

    # Apply fraction and bounds
    kelly = full_kelly * kelly_fraction
    kelly = max(0, min(kelly, 1.0))  # Bound to [0, 1]

    return kelly
```

### Exposure Limits

```python
@dataclass
class ExposureLimits:
    """Portfolio exposure constraints."""

    # Single name limits
    max_single_position: float = 0.10      # 10% per position
    max_single_sector: float = 0.30        # 30% per sector

    # Gross/Net exposure
    max_gross_exposure: float = 2.0        # 200% (enables 2x leverage)
    max_net_exposure: float = 0.50         # 50% long/short bias
    min_net_exposure: float = -0.50        # Symmetric short bias limit

    # Concentration
    max_top_5_concentration: float = 0.50  # Top 5 positions <= 50%

    def validate_position(self,
                         new_position: Position,
                         current_positions: List[Position],
                         portfolio_value: float) -> bool:
        """Check if adding position violates limits."""

        # Single position check
        position_weight = abs(new_position.size) / portfolio_value
        if position_weight > self.max_single_position:
            return False

        # Gross exposure check
        current_gross = sum(abs(p.size) for p in current_positions)
        new_gross = current_gross + abs(new_position.size)
        if new_gross / portfolio_value > self.max_gross_exposure:
            return False

        # Net exposure check
        current_net = sum(p.size for p in current_positions)
        new_net = current_net + new_position.size
        net_exposure = new_net / portfolio_value
        if net_exposure > self.max_net_exposure or net_exposure < self.min_net_exposure:
            return False

        return True
```

### Drawdown Controls

```python
class DrawdownManager:
    """Manages position sizing based on drawdown state."""

    def __init__(self,
                 daily_limit: float = 0.03,
                 weekly_limit: float = 0.05,
                 monthly_limit: float = 0.10,
                 circuit_breaker_level: float = 0.15):
        self.daily_limit = daily_limit
        self.weekly_limit = weekly_limit
        self.monthly_limit = monthly_limit
        self.circuit_breaker_level = circuit_breaker_level

        self.high_water_mark = 0.0
        self.last_circuit_breaker = None
        self.recovery_period_days = 5

    def update(self, portfolio_value: float) -> None:
        """Update high water mark."""
        self.high_water_mark = max(self.high_water_mark, portfolio_value)

    def get_current_drawdown(self, portfolio_value: float) -> float:
        """Calculate current drawdown from HWM."""
        if self.high_water_mark == 0:
            return 0.0
        return (self.high_water_mark - portfolio_value) / self.high_water_mark

    def get_position_scalar(self,
                           portfolio_value: float,
                           daily_pnl: float,
                           weekly_pnl: float,
                           monthly_pnl: float) -> float:
        """
        Get position scalar based on recent performance.

        Returns value in [0, 1] to scale down positions.
        """
        # Check circuit breaker
        current_dd = self.get_current_drawdown(portfolio_value)
        if current_dd > self.circuit_breaker_level:
            self.last_circuit_breaker = pd.Timestamp.now()
            return 0.0  # Stop trading

        # Recovery period check
        if self.last_circuit_breaker is not None:
            days_since = (pd.Timestamp.now() - self.last_circuit_breaker).days
            if days_since < self.recovery_period_days:
                return 0.0  # Still in recovery

        # Gradual reduction based on drawdown
        scalar = 1.0

        # Daily drawdown reduction
        if daily_pnl < -self.daily_limit:
            scalar *= 0.5

        # Weekly drawdown reduction
        if weekly_pnl < -self.weekly_limit:
            scalar *= 0.5

        # Monthly drawdown reduction
        if monthly_pnl < -self.monthly_limit:
            scalar *= 0.25

        # Current drawdown reduction (gradual)
        if current_dd > 0.05:
            dd_scalar = 1 - (current_dd - 0.05) / (self.circuit_breaker_level - 0.05)
            scalar *= max(0.25, dd_scalar)

        return scalar

    def should_stop_trading(self, portfolio_value: float) -> bool:
        """Check if circuit breaker is triggered."""
        return self.get_current_drawdown(portfolio_value) > self.circuit_breaker_level
```

### Stop-Loss Mechanisms

```python
@dataclass
class StopLoss:
    """Stop-loss configuration for a position."""

    # Price-based stops
    initial_stop_pct: float = 0.05        # 5% initial stop
    trailing_stop_pct: float = 0.03       # 3% trailing stop

    # ATR-based stops
    atr_multiplier: float = 2.0           # 2x ATR for stop

    # Time-based stops
    max_holding_days: int = 63            # Max holding period

    # Track high/low since entry
    highest_since_entry: float = 0.0
    lowest_since_entry: float = float('inf')
    entry_price: float = 0.0
    entry_date: pd.Timestamp = None

    def update(self, current_price: float) -> None:
        """Update tracking prices."""
        self.highest_since_entry = max(self.highest_since_entry, current_price)
        self.lowest_since_entry = min(self.lowest_since_entry, current_price)

    def check_stop(self,
                  current_price: float,
                  current_date: pd.Timestamp,
                  atr: float,
                  position_direction: int) -> bool:
        """
        Check if stop-loss is triggered.

        Args:
            current_price: Current market price
            current_date: Current date
            atr: Current ATR value
            position_direction: 1 for long, -1 for short

        Returns:
            True if stop is triggered
        """
        # Time-based stop
        if self.entry_date is not None:
            days_held = (current_date - self.entry_date).days
            if days_held > self.max_holding_days:
                return True

        if position_direction > 0:  # Long position
            # Initial stop
            if current_price < self.entry_price * (1 - self.initial_stop_pct):
                return True

            # Trailing stop
            trailing_stop_price = self.highest_since_entry * (1 - self.trailing_stop_pct)
            if current_price < trailing_stop_price:
                return True

            # ATR stop
            atr_stop_price = self.entry_price - (self.atr_multiplier * atr)
            if current_price < atr_stop_price:
                return True

        else:  # Short position
            # Initial stop
            if current_price > self.entry_price * (1 + self.initial_stop_pct):
                return True

            # Trailing stop
            trailing_stop_price = self.lowest_since_entry * (1 + self.trailing_stop_pct)
            if current_price > trailing_stop_price:
                return True

            # ATR stop
            atr_stop_price = self.entry_price + (self.atr_multiplier * atr)
            if current_price > atr_stop_price:
                return True

        return False
```

### Correlation Constraints

```python
class CorrelationManager:
    """Manage portfolio correlation risk."""

    def __init__(self,
                 lookback: int = 60,
                 max_avg_correlation: float = 0.5,
                 max_pairwise_correlation: float = 0.8):
        self.lookback = lookback
        self.max_avg_correlation = max_avg_correlation
        self.max_pairwise_correlation = max_pairwise_correlation

    def calculate_correlation_matrix(self,
                                    returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling correlation matrix."""
        return returns.iloc[-self.lookback:].corr()

    def check_new_position(self,
                          new_asset_returns: pd.Series,
                          existing_returns: pd.DataFrame) -> bool:
        """
        Check if adding new asset violates correlation constraints.
        """
        if existing_returns.empty:
            return True

        # Calculate correlations with existing positions
        correlations = existing_returns.iloc[-self.lookback:].corrwith(
            new_asset_returns.iloc[-self.lookback:]
        )

        # Check max pairwise correlation
        if correlations.abs().max() > self.max_pairwise_correlation:
            return False

        # Check average correlation
        if correlations.abs().mean() > self.max_avg_correlation:
            return False

        return True

    def get_diversification_ratio(self,
                                  returns: pd.DataFrame,
                                  weights: np.ndarray) -> float:
        """
        Calculate diversification ratio.

        DR = sum(w_i * sigma_i) / sigma_portfolio

        Higher DR means better diversification.
        """
        individual_vols = returns.std()
        weighted_vol_sum = (weights * individual_vols).sum()

        cov_matrix = returns.cov()
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)

        return weighted_vol_sum / portfolio_vol if portfolio_vol > 0 else 1.0
```

---

## Summary

This document provides:

1. **Momentum Strategy**: Time-series momentum with volatility targeting, grounded in academic research (Moskowitz et al., 2012). Expected Sharpe 0.4-0.8 with 200-400% turnover.

2. **Mean-Reversion Strategy**: Z-score based with Ornstein-Uhlenbeck half-life estimation. Expected Sharpe 0.5-1.0 with higher turnover.

3. **Unified Interface**: Abstract base class ensuring all strategies (including future LSTM) are comparable and consistently evaluated.

4. **Objective Functions**: Multi-objective optimization combining Sharpe, Sortino, Calmar, and Information Ratio with penalty functions for constraint violations.

5. **Risk Controls**: Comprehensive risk management including:
   - Position sizing (fixed fractional, volatility targeting, Kelly)
   - Exposure limits (single name, sector, gross/net)
   - Drawdown controls (daily/weekly/monthly limits, circuit breakers)
   - Stop-loss mechanisms (fixed, trailing, ATR-based, time-based)
   - Correlation constraints

**Next Steps:**
1. Implement these strategies in `/Users/willhatfield/ML-Stock_Trading/src/models/math_first/`
2. Run walk-forward backtests on historical data
3. Establish performance benchmarks
4. Compare LSTM strategy performance against these baselines
5. Only proceed with ML complexity if it demonstrably outperforms

---

*Document Version: 1.0*
*Created: 2026-01-07*
*Author: QuantResearcher Agent*
