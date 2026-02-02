"""
Z-score mean reversion strategy implementation.
"""

from typing import Dict, Any, List

import numpy as np
import pandas as pd

from .base import Strategy, StrategyConfig, Signal, Position


class MeanReversionStrategy(Strategy):
    """Contrarian mean-reversion strategy using z-score signals."""

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        params = config.parameters or {}
        limits = config.risk_limits or {}

        self.lookback = int(params.get('lookback', 20))
        self.entry_z = float(params.get('entry_z', 1.5))
        self.exit_z = float(params.get('exit_z', 0.5))
        self.max_half_life = float(params.get('max_half_life', 20.0))

        self.max_position = float(limits.get('max_position', config.max_position))
        self.target_vol = float(limits.get('target_volatility', config.target_volatility))

        self.half_life = None

    @property
    def min_history(self) -> int:
        return max(self.lookback * 3, 60)

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            'lookback': self.lookback,
            'entry_z': self.entry_z,
            'exit_z': self.exit_z,
            'max_half_life': self.max_half_life,
            'half_life': self.half_life,
            'max_position': self.max_position,
        }

    def fit(self, data: pd.DataFrame) -> 'MeanReversionStrategy':
        self.validate_data(data)
        self.half_life = self._estimate_half_life(data['Close'])
        self._is_fitted = True
        return self

    def _estimate_half_life(self, prices: pd.Series) -> float:
        if len(prices) < 2:
            return np.inf

        y = np.log(prices)
        y_lag = y.shift(1)
        delta_y = y - y_lag

        valid = ~(y_lag.isna() | delta_y.isna())
        y_lag = y_lag[valid].values
        delta_y = delta_y[valid].values

        if len(y_lag) < 2:
            return np.inf

        X = np.column_stack([np.ones(len(y_lag)), y_lag])
        coeffs = np.linalg.lstsq(X, delta_y, rcond=None)[0]
        lambda_coef = coeffs[1]

        if lambda_coef >= 0:
            return np.inf

        return float(-np.log(2) / lambda_coef)

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        if not self._is_fitted:
            raise ValueError("Strategy must be fitted before generating signals")

        self.validate_data(data)
        prices = data['Close']

        recent = prices.iloc[-self.lookback:]
        ma = recent.mean()
        std = recent.std()

        if std <= 0:
            z_score = 0.0
        else:
            z_score = float((prices.iloc[-1] - ma) / std)

        if self.half_life is not None and self.half_life > self.max_half_life:
            direction = 0.0
        elif z_score > self.entry_z:
            direction = -1.0
        elif z_score < -self.entry_z:
            direction = 1.0
        elif abs(z_score) < self.exit_z:
            direction = 0.0
        else:
            direction = -z_score / self.entry_z
            direction = float(np.clip(direction, -1.0, 1.0))

        confidence = min(1.0, abs(z_score) / self.entry_z) if self.entry_z > 0 else 0.0

        signal = Signal(
            timestamp=data.index[-1],
            symbol=data.attrs.get('symbol', 'UNKNOWN'),
            direction=float(direction),
            confidence=float(confidence),
            raw_score=float(-z_score),
            metadata={
                'z_score': float(z_score),
                'moving_average': float(ma),
                'std_dev': float(std) if std == std else 0.0,
                'half_life': self.half_life,
                'lookback': self.lookback,
            },
        )

        self._last_signal = signal
        return signal

    def calculate_position_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_positions: List[Position],
    ) -> float:
        z_score = abs(signal.metadata.get('z_score', 0.0))

        if z_score < 1.5:
            size_scalar = 0.25
        elif z_score < 2.0:
            size_scalar = 0.5
        elif z_score < 2.5:
            size_scalar = 0.75
        else:
            size_scalar = 1.0

        base_size = signal.direction * portfolio_value * self.max_position
        return float(base_size * size_scalar * signal.confidence)
