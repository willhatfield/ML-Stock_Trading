import types
import sys
from pathlib import Path
import pandas as pd
import pytest

tf_stub = types.SimpleNamespace(random=types.SimpleNamespace(set_seed=lambda x: None))
preprocessing_stub = types.SimpleNamespace(StandardScaler=type('StandardScaler', (), {'fit_transform': lambda self, x: x, 'transform': lambda self, x: x}))
metrics_stub = types.SimpleNamespace(
    accuracy_score=lambda a, b: 0,
    precision_score=lambda a, b, zero_division=0: 0,
    recall_score=lambda a, b, zero_division=0: 0,
    f1_score=lambda a, b, zero_division=0: 0,
    confusion_matrix=lambda a, b: 0,
)

sys.modules.setdefault('tensorflow', tf_stub)
sys.modules.setdefault('sklearn', types.SimpleNamespace(preprocessing=preprocessing_stub, metrics=metrics_stub))
sys.modules.setdefault('sklearn.preprocessing', preprocessing_stub)
sys.modules.setdefault('sklearn.metrics', metrics_stub)
sys.modules.setdefault('joblib', types.SimpleNamespace())
class _StockPredictor:
    def __init__(self, *args, **kwargs):
        pass
sys.modules.setdefault('beta2', types.SimpleNamespace(StockPredictor=_StockPredictor))

sys.path.append(str(Path(__file__).resolve().parents[1]))

from stock_direction_trader import StockDirectionTrader


def test_fee_deduction():
    trader = StockDirectionTrader(symbol='TEST', initial_balance=100, transaction_cost=0.01)
    date1 = pd.Timestamp('2020-01-01')
    trader.execute_trade('BUY', 10, date1)

    assert trader.balance == pytest.approx(0)
    buy_trade = trader.trade_log[-1]
    assert buy_trade['transaction_cost'] == pytest.approx(buy_trade['price'] * buy_trade['shares'] * trader.transaction_cost)

    date2 = pd.Timestamp('2020-01-02')
    trader.execute_trade('SELL', 12, date2)
    sell_trade = trader.trade_log[-1]
    shares = buy_trade['shares']
    expected_balance = shares * 12 * (1 - trader.transaction_cost)
    assert trader.balance == pytest.approx(expected_balance)
    assert sell_trade['transaction_cost'] == pytest.approx(sell_trade['price'] * sell_trade['shares'] * trader.transaction_cost)
