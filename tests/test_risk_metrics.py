import numpy as np
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stock_direction_trader import StockDirectionTrader


def create_dummy_trader():
    trader = StockDirectionTrader('TEST', lookback_days=1, initial_balance=1000, max_position=0.5)
    predictions = [1, 0] * 5

    def mock_predict(features):
        return predictions.pop(0), 0.6

    trader.predict_direction = mock_predict
    return trader


def create_dummy_df():
    prices = np.linspace(100, 110, 10)
    df = pd.DataFrame({'Close': prices, 'feat': np.arange(10)})
    df['daily_return'] = df['Close'].pct_change() * 100
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    return df


def test_metrics_and_position_cap():
    trader = create_dummy_trader()
    df = create_dummy_df()
    trader.backtest(df)

    for key in ['Sharpe', 'Sortino', 'MDD']:
        assert key in trader.metrics

    for trade in trader.trade_log:
        if trade['action'] == 'BUY':
            trade_value = trade['shares'] * trade['price']
            equity_before = trade['previous_balance'] + trade.get('previous_position', 0) * trade['price']
            assert trade_value <= trader.max_position * equity_before + 1e-8
