import types
import sys
from pathlib import Path
import pandas as pd

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


def test_next_bar_execution():
    trader = StockDirectionTrader(symbol='TEST', lookback_days=1, initial_balance=100, transaction_cost=0.0)

    dates = pd.date_range('2020-01-01', periods=4, freq='D')
    df = pd.DataFrame(
        {
            'Close': [10, 11, 12, 13],
            'feat': [0, 0, 0, 0],
            'daily_return': [0, 0, 0, 0],
            'target': [1, 1, 1, 1],
        },
        index=dates,
    )

    trader.predict_direction = lambda features: (1, 1.0)
    signals = iter(['BUY', 'SELL'])
    trader.generate_signal = lambda prediction, confidence, position: next(signals, 'HOLD')

    trader.backtest(df)

    assert len(trader.trade_log) == 2
    buy_trade = trader.trade_log[0]
    sell_trade = trader.trade_log[1]

    assert buy_trade['action'] == 'BUY'
    assert buy_trade['price'] == df['Close'].iloc[2]
    assert buy_trade['date'] == dates[2]

    assert sell_trade['action'] == 'SELL'
    assert sell_trade['price'] == df['Close'].iloc[3]
    assert sell_trade['date'] == dates[3]
