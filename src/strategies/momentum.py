"""
Time-series momentum (TSMOM) strategy implementation.
"""

from typing import Dict, Any, List

import numpy as np
import pandas as pd

from .base import Strategy, StrategyConfig, Signal, Position


class MomentumStrategy(Strategy):
    """Volatility-targeted time-series momentum strategy."""

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        params = config.parameters or {}
        limits = config.risk_limits or {}

        self.formation_period = int(params.get('formation_period', 252))
        self.skip_period = int(params.get('skip_period', 21))
        self.vol_window = int(params.get('vol_window', 60))
        self.entry_threshold = float(params.get('entry_threshold', 0.1))

        self.target_vol = float(limits.get('target_volatility', config.target_volatility))
        self.max_position = float(limits.get('max_position', config.max_position))

    @property
    def min_history(self) -> int:
        return self.formation_period + self.skip_period + self.vol_window

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            'formation_period': self.formation_period,
            'skip_period': self.skip_period,
            'vol_window': self.vol_window,
            'entry_threshold': self.entry_threshold,
            'target_volatility': self.target_vol,
            'max_position': self.max_position,
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

        if len(prices) < self.min_history:
            raise ValueError("Insufficient data for momentum signal")

        # Calculate momentum return excluding skip period
        p_now = prices.iloc[-self.skip_period - 1]
        p_past = prices.iloc[-self.formation_period - self.skip_period]
        momentum_return = (p_now - p_past) / p_past if p_past != 0 else 0.0

        # Realized volatility over vol_window
        returns = prices.pct_change().dropna()
        daily_vol = returns.iloc[-self.vol_window:].std()
        annual_vol = daily_vol * np.sqrt(252) if daily_vol > 0 else 0.0

        if daily_vol > 0:
            raw_score = momentum_return / (daily_vol * np.sqrt(self.formation_period))
            strength = min(1.0, abs(raw_score))
            direction = np.sign(raw_score) * strength
        else:
            raw_score = 0.0
            strength = 0.0
            direction = 0.0

        if abs(direction) < self.entry_threshold:
            direction = 0.0

        confidence = min(1.0, abs(raw_score))

        signal = Signal(
            timestamp=data.index[-1],
            symbol=data.attrs.get('symbol', 'UNKNOWN'),
            direction=float(direction),
            confidence=float(confidence),
            raw_score=float(raw_score),
            metadata={
                'momentum_return': float(momentum_return),
                'daily_vol': float(daily_vol) if daily_vol == daily_vol else 0.0,
                'annual_vol': float(annual_vol),
                'formation_period': self.formation_period,
                'skip_period': self.skip_period,
                'vol_window': self.vol_window,
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
        current_vol = signal.metadata.get('annual_vol', self.target_vol)
        if current_vol and current_vol > 0:
            vol_scalar = self.target_vol / current_vol
        else:
            vol_scalar = 1.0

        vol_scalar = min(vol_scalar, 2.0)

        base_size = signal.direction * portfolio_value * self.max_position
        target_size = base_size * vol_scalar * signal.confidence
        return float(target_size)
