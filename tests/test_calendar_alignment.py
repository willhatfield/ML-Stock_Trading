import os
import sys
from datetime import datetime

import pandas as pd
import yfinance as yf

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from stock_direction_trader import StockDirectionTrader


def _load_local_data():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'SPY_5yr.csv')
    cols = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    df = pd.read_csv(path, skiprows=3, names=cols, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    return df


def test_calendar_alignment(monkeypatch):
    # Use local CSV data to avoid network dependency
    base_df = _load_local_data()

    def fake_download(symbol, start=None, end=None, progress=False):
        df = base_df.copy()
        if start:
            df = df[df.index >= pd.to_datetime(start)]
        if end:
            df = df[df.index <= pd.to_datetime(end)]
        return df

    monkeypatch.setattr(yf, 'download', fake_download)

    symbol = 'SPY'
    start = datetime(2020, 1, 1)
    end = datetime(2020, 12, 31)
    trader = StockDirectionTrader(symbol)
    prepared = trader.prepare_data(start, end)

    fetched = yf.download(symbol, start=start, end=end, progress=False)
    intersection_len = len(prepared.index.intersection(fetched.index))
    assert intersection_len == len(prepared.index)
