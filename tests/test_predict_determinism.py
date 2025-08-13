import numpy as np
from stock_direction_trader import StockDirectionTrader


class DummyModel:
    def predict(self, features, verbose=0):
        return np.array([[0.7]])


def test_predict_direction_deterministic():
    trader = StockDirectionTrader("TEST")
    trader.model = DummyModel()
    features = np.zeros((trader.lookback_days, 5))

    first = trader.predict_direction(features)
    second = trader.predict_direction(features)

    assert first == second
