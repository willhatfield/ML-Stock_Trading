"""
Base strategy interface for all trading strategies.

All strategies (momentum, mean-reversion, and ML-based) must implement this interface
to ensure consistent evaluation under the backtest protocol.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
    parameters: Dict[str, Any] = field(default_factory=dict)
    risk_limits: Dict[str, float] = field(default_factory=dict)

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
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Current position information."""
    symbol: str
    size: float               # Signed position size (dollars)
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

        if data[required_columns].isnull().any().any():
            raise ValueError("Data contains NaN values in OHLCV columns")

        return True

    def get_metrics(self,
                   returns: pd.Series,
                   benchmark_returns: Optional[pd.Series] = None,
                   risk_free_rate: float = 0.04) -> Dict[str, float]:
        """
        Calculate strategy performance metrics.

        Args:
            returns: Strategy returns series (daily)
            benchmark_returns: Optional benchmark returns for comparison
            risk_free_rate: Annual risk-free rate (default 4%)

        Returns:
            Dictionary of metric names and values
        """
        metrics = {}

        if len(returns) == 0:
            return metrics

        # Daily risk-free rate
        rf_daily = (1 + risk_free_rate) ** (1/252) - 1

        # Basic metrics
        total_return = (1 + returns).prod() - 1
        n_days = len(returns)
        metrics['total_return'] = total_return
        metrics['cagr'] = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0
        metrics['volatility'] = returns.std() * np.sqrt(252)

        excess_returns = returns - rf_daily
        metrics['sharpe_ratio'] = (excess_returns.mean() * 252) / metrics['volatility'] if metrics['volatility'] > 0 else 0

        # Downside metrics
        downside_returns = returns[returns < 0]
        metrics['downside_vol'] = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        metrics['sortino_ratio'] = (excess_returns.mean() * 252) / metrics['downside_vol'] if metrics['downside_vol'] > 0 else 0

        # Drawdown metrics
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        metrics['max_drawdown'] = drawdowns.min()
        metrics['calmar_ratio'] = metrics['cagr'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0

        # Win/loss metrics
        metrics['win_rate'] = (returns > 0).mean()
        metrics['avg_win'] = returns[returns > 0].mean() if (returns > 0).any() else 0
        metrics['avg_loss'] = returns[returns < 0].mean() if (returns < 0).any() else 0
        metrics['profit_factor'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else np.inf

        # Benchmark comparison
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            # Align series
            aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
            if len(aligned) > 0:
                excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
                tracking_error = excess.std() * np.sqrt(252)
                metrics['information_ratio'] = excess.mean() * 252 / tracking_error if tracking_error > 0 else 0

                # Beta
                cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
                var = aligned.iloc[:, 1].var()
                metrics['beta'] = cov / var if var > 0 else 0
                metrics['alpha'] = metrics['cagr'] - metrics['beta'] * ((1 + aligned.iloc[:, 1]).prod() ** (252/len(aligned)) - 1)

        return metrics

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type={self.strategy_type.value})"
