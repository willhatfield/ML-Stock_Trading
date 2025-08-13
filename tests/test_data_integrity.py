import os
import sys

import pandas as pd

# Ensure the project root is on the Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stock_direction_trader import StockDirectionTrader

def test_ohlcv_from_source():
    trader = StockDirectionTrader('AAPL', lookback_days=2)
    # Create sample OHLCV data with distinct values
    dates = pd.date_range('2023-01-03', periods=5, freq='B')
    sample = pd.DataFrame(
        {
            'Open': [10, 11, 12, 13, 14],
            'High': [11, 12, 13, 14, 15],
            'Low': [9, 10, 11, 12, 13],
            'Close': [10.5, 11.5, 12.5, 13.5, 14.5],
            'Volume': [1000, 2000, 1500, 1800, 1600],
        },
        index=dates,
    )
    # Patch fetch_data to return our sample dataframe
    trader.stock_predictor.fetch_data = lambda start, end, symbol: sample.copy()

    df = trader.prepare_data(dates.min(), dates.max())
    subset = df.head(3)

    assert (subset['Open'] != subset['Close']).all()
    assert (subset['High'] != subset['Close']).all()
    assert (subset['Low'] != subset['Close']).all()
    assert (subset['Volume'] > 0).all()

