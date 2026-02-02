"""Walk-forward backtest engine."""

from dataclasses import dataclass, field, replace
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from .metrics import compute_metrics, returns_from_prices
from ..strategies.base import Strategy, StrategyConfig, Position


REQUIRED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']


@dataclass
class WalkForwardConfig:
    train_years: int = 2
    val_months: int = 3
    test_months: int = 3
    step_months: int = 3
    embargo_days: int = 5
    initial_capital: float = 100000.0
    cost_bps: float = 30.0
    risk_free_rate: float = 0.04
    min_train_days: int = 252
    min_val_days: int = 63
    min_test_days: int = 63
    min_folds: int = 4
    cost_sensitivity_bps: Tuple[float, float, float] = (5.0, 30.0, 100.0)
    enable_cost_sensitivity: bool = True
    enable_robustness: bool = True
    enable_parameter_sensitivity: bool = True
    parameter_sensitivity_pct: float = 0.20
    regime_ma_window: int = 200
    enable_shorts: bool = True
    max_single_long_pct: float = 0.20
    max_single_short_pct: float = 0.10
    target_gross_exposure: float = 1.40
    max_gross_exposure: float = 1.90
    max_long_exposure: float = 1.50
    max_short_exposure: float = 0.5
    max_net_exposure: float = 1.0
    min_net_exposure: float = -1.0
    net_long_bias: float = 0.20
    short_penalty: float = 0.70


@dataclass
class FoldResult:
    fold_id: int
    train_range: Tuple[pd.Timestamp, pd.Timestamp]
    val_range: Tuple[pd.Timestamp, pd.Timestamp]
    test_range: Tuple[pd.Timestamp, pd.Timestamp]
    metrics: Dict[str, float]
    returns: pd.Series
    equity_curve: pd.DataFrame
    trades: pd.DataFrame


@dataclass
class BacktestResult:
    symbol: str
    config: WalkForwardConfig
    fold_results: List[FoldResult]
    aggregate_metrics: Dict[str, float]
    aggregate_equity_curve: pd.DataFrame
    symbols: Optional[List[str]] = None
    benchmark_returns: Dict[str, pd.Series] = field(default_factory=dict)
    benchmarks: Optional[Dict[str, Any]] = None
    cost_sensitivity: Optional[Dict[str, Any]] = None
    robustness: Optional[Dict[str, Any]] = None


def _load_csv_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=[0, 1])
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c[0] != 'Price' else 'Date' for c in df.columns]
    if 'Date' not in df.columns:
        raise ValueError(f"CSV missing Date column: {path}")

    df = df[df['Date'] != 'Date']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    df = df.rename(columns={'Close': 'Close', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Volume': 'Volume'})
    df = df[REQUIRED_COLUMNS]
    return df


def _normalize_ohlcv(
    df: pd.DataFrame,
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError(f"No data available for {symbol}")

    df = df[REQUIRED_COLUMNS].dropna()
    df = df.sort_index()
    df = df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
    df.attrs['symbol'] = symbol
    if df.empty:
        raise ValueError(f"No data available for {symbol} in requested date range")
    return df


def _extract_symbol_frame(df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    if not isinstance(df.columns, pd.MultiIndex):
        return df.copy()

    level0 = df.columns.get_level_values(0)
    level1 = df.columns.get_level_values(1)
    if symbol in level0:
        return df.xs(symbol, axis=1, level=0)
    if symbol in level1:
        return df.xs(symbol, axis=1, level=1)
    return None


def _chunked(values: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(values), size):
        yield values[i:i + size]


def load_ohlcv_data(
    symbol: str,
    start_date: str,
    end_date: str,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load OHLCV data from local CSV if available, else yfinance."""
    data_dir = data_dir or Path('data')
    csv_path = data_dir / f"{symbol}_5yr.csv"

    if csv_path.exists():
        df = _load_csv_data(csv_path)
    else:
        try:
            import yfinance as yf
        except Exception as exc:
            raise ImportError("yfinance is required to fetch data") from exc

        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )

    return _normalize_ohlcv(df, symbol, start_date, end_date)


def load_ohlcv_data_multi(
    symbols: List[str],
    start_date: str,
    end_date: str,
    data_dir: Optional[Path] = None,
) -> Tuple[Dict[str, pd.DataFrame], pd.DatetimeIndex]:
    data_map: Dict[str, pd.DataFrame] = {}
    failures: List[str] = []
    pending: List[str] = []
    data_dir = data_dir or Path('data')

    for symbol in symbols:
        csv_path = data_dir / f"{symbol}_5yr.csv"
        if csv_path.exists():
            try:
                df = _load_csv_data(csv_path)
                data_map[symbol] = _normalize_ohlcv(df, symbol, start_date, end_date)
            except Exception:
                failures.append(symbol)
        else:
            pending.append(symbol)

    if pending:
        try:
            import yfinance as yf
        except Exception as exc:
            raise ImportError("yfinance is required to fetch data") from exc

        batch_size = 1
        retries = 2
        backoff_seconds = 5.0
        pause_seconds = 0.5

        for batch in _chunked(pending, batch_size):
            df = None
            for attempt in range(retries + 1):
                try:
                    df = yf.download(
                        batch,
                        start=start_date,
                        end=end_date,
                        auto_adjust=True,
                        progress=False,
                        group_by="ticker",
                        threads=False,
                    )
                    if df is not None and not df.empty:
                        break
                except Exception:
                    if attempt >= retries:
                        df = None
                    else:
                        time.sleep(backoff_seconds * (2 ** attempt))

            if df is None or df.empty:
                failures.extend(batch)
                time.sleep(pause_seconds)
                continue

            for symbol in batch:
                symbol_df = _extract_symbol_frame(df, symbol)
                if symbol_df is None or symbol_df.empty:
                    failures.append(symbol)
                    continue
                try:
                    data_map[symbol] = _normalize_ohlcv(symbol_df, symbol, start_date, end_date)
                except Exception:
                    failures.append(symbol)
            time.sleep(pause_seconds)

    if not data_map:
        raise ValueError("No data available across symbols")

    common_index: Optional[pd.DatetimeIndex] = None
    for df in data_map.values():
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)

    if common_index is None or common_index.empty:
        raise ValueError("No overlapping dates across symbols")

    aligned_map: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        if symbol not in data_map:
            continue
        df = data_map[symbol]
        aligned = df.reindex(common_index).dropna()
        if aligned.empty:
            failures.append(symbol)
            continue
        aligned.attrs['symbol'] = symbol
        aligned_map[symbol] = aligned

    if failures:
        preview = ", ".join(failures[:10])
        print(f"WARNING: Skipped {len(set(failures))} symbols due to missing data or download errors. First: {preview}")

    if not aligned_map:
        raise ValueError("No aligned data available across symbols")

    return aligned_map, common_index


def _generate_folds(index: pd.DatetimeIndex, config: WalkForwardConfig) -> Iterable[Tuple[int, int, int, int, int]]:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex")

    start_date = index.min()
    end_date = index.max()

    cursor = start_date
    while True:
        train_start = cursor
        train_end = train_start + pd.DateOffset(years=config.train_years)
        val_end = train_end + pd.DateOffset(months=config.val_months)

        train_start_idx = index.searchsorted(train_start, side='left')
        train_end_idx = index.searchsorted(train_end, side='right') - 1
        val_end_idx = index.searchsorted(val_end, side='right') - 1

        if train_end_idx <= train_start_idx or val_end_idx <= train_end_idx:
            break

        test_start_idx = val_end_idx + 1 + config.embargo_days
        if test_start_idx >= len(index):
            break

        test_start_date = index[test_start_idx]
        test_end = test_start_date + pd.DateOffset(months=config.test_months)
        if test_end > end_date:
            break

        test_end_idx = index.searchsorted(test_end, side='right') - 1

        yield train_start_idx, train_end_idx, val_end_idx, test_start_idx, test_end_idx

        cursor = cursor + pd.DateOffset(months=config.step_months)


def _apply_portfolio_constraints(
    targets: Dict[str, float],
    portfolio_value: float,
    config: WalkForwardConfig,
) -> Dict[str, float]:
    if portfolio_value <= 0:
        return {symbol: 0.0 for symbol in targets}

    def _apply_single_name_caps(values: Dict[str, float]) -> Dict[str, float]:
        capped: Dict[str, float] = {}
        for symbol, value in values.items():
            if not config.enable_shorts and value < 0:
                value = 0.0
            cap = config.max_single_long_pct if value >= 0 else config.max_single_short_pct
            max_value = cap * portfolio_value
            if max_value > 0:
                value = float(np.clip(value, -max_value, max_value))
            capped[symbol] = value
        return capped

    constrained = _apply_single_name_caps(targets)

    if config.short_penalty != 1.0:
        for symbol, value in constrained.items():
            if value < 0:
                constrained[symbol] = value * config.short_penalty

    long_exposure = sum(max(v, 0.0) for v in constrained.values())
    short_exposure = sum(abs(min(v, 0.0)) for v in constrained.values())
    net_exposure = long_exposure - short_exposure

    if config.net_long_bias > 0 and short_exposure > 0:
        desired_net = config.net_long_bias * portfolio_value
        if net_exposure < desired_net:
            reduction = min(1.0, (desired_net - net_exposure) / short_exposure)
            if reduction > 0:
                for symbol, value in constrained.items():
                    if value < 0:
                        constrained[symbol] = value * (1.0 - reduction)

    long_exposure = sum(max(v, 0.0) for v in constrained.values())
    short_exposure = sum(abs(min(v, 0.0)) for v in constrained.values())
    gross_exposure = long_exposure + short_exposure

    if gross_exposure > 0 and config.target_gross_exposure > 0:
        target_gross = config.target_gross_exposure
        if config.max_gross_exposure > 0:
            target_gross = min(target_gross, config.max_gross_exposure)
        target_value = target_gross * portfolio_value
        if gross_exposure < target_value:
            gross_scale = target_value / gross_exposure
            for symbol, value in constrained.items():
                constrained[symbol] = value * gross_scale
            constrained = _apply_single_name_caps(constrained)

    long_exposure = sum(max(v, 0.0) for v in constrained.values())
    short_exposure = sum(abs(min(v, 0.0)) for v in constrained.values())

    if long_exposure > 0 and config.max_long_exposure > 0:
        long_scale = min(1.0, (config.max_long_exposure * portfolio_value) / long_exposure)
    else:
        long_scale = 1.0

    if short_exposure > 0 and config.max_short_exposure > 0:
        short_scale = min(1.0, (config.max_short_exposure * portfolio_value) / short_exposure)
    else:
        short_scale = 1.0

    if long_scale < 1.0 or short_scale < 1.0:
        for symbol, value in constrained.items():
            if value >= 0:
                constrained[symbol] = value * long_scale
            else:
                constrained[symbol] = value * short_scale

    long_exposure = sum(max(v, 0.0) for v in constrained.values())
    short_exposure = sum(abs(min(v, 0.0)) for v in constrained.values())
    gross_exposure = long_exposure + short_exposure

    if gross_exposure > 0 and config.max_gross_exposure > 0:
        gross_scale = min(1.0, (config.max_gross_exposure * portfolio_value) / gross_exposure)
        if gross_scale < 1.0:
            for symbol, value in constrained.items():
                constrained[symbol] = value * gross_scale

    long_exposure = sum(max(v, 0.0) for v in constrained.values())
    short_exposure = sum(abs(min(v, 0.0)) for v in constrained.values())
    net_exposure = long_exposure - short_exposure

    if net_exposure > 0 and config.max_net_exposure is not None:
        net_scale = min(1.0, (config.max_net_exposure * portfolio_value) / net_exposure)
    elif net_exposure < 0 and config.min_net_exposure is not None:
        net_scale = min(1.0, (config.min_net_exposure * portfolio_value) / net_exposure)
    else:
        net_scale = 1.0

    if net_scale < 1.0:
        for symbol, value in constrained.items():
            constrained[symbol] = value * net_scale

    return constrained


def _run_single_fold(
    data: pd.DataFrame,
    symbol: str,
    strategy_cls: Type[Strategy],
    strategy_config: StrategyConfig,
    fold_id: int,
    train_slice: slice,
    val_slice: slice,
    test_slice: slice,
    config: WalkForwardConfig,
) -> FoldResult:
    train_df = data.iloc[train_slice]
    val_df = data.iloc[val_slice]
    test_df = data.iloc[test_slice]

    strategy = strategy_cls(strategy_config)
    strategy.fit(train_df)

    cost_rate = (config.cost_bps / 10000.0) / 2.0

    cash = config.initial_capital
    shares = 0.0
    entry_price = None
    entry_time = None
    holding_days = 0

    returns = []
    equity = []
    trades = []

    test_index = test_df.index

    for idx in range(0, len(test_index) - 1):
        current_idx = test_df.index[idx]
        next_idx = test_df.index[idx + 1]

        current_price = test_df['Close'].iloc[idx]
        next_price = test_df['Close'].iloc[idx + 1]

        portfolio_value = cash + shares * current_price

        history = data.loc[:current_idx]
        signal = strategy.generate_signal(history)

        current_positions = []
        if shares != 0:
            current_positions.append(
                Position(
                    symbol=symbol,
                    size=shares * current_price,
                    entry_price=entry_price or current_price,
                    entry_time=entry_time or current_idx,
                    current_price=current_price,
                    unrealized_pnl=(current_price - (entry_price or current_price)) * shares,
                    holding_days=holding_days,
                )
            )

        target_value = strategy.calculate_position_size(signal, portfolio_value, current_positions)
        target_shares = target_value / next_price if next_price != 0 else 0.0

        delta_shares = target_shares - shares
        transaction_cost = abs(delta_shares) * next_price * cost_rate

        if delta_shares != 0:
            trades.append(
                {
                    'date': next_idx,
                    'symbol': symbol,
                    'price': float(next_price),
                    'shares': float(delta_shares),
                    'direction': 'BUY' if delta_shares > 0 else 'SELL',
                    'notional': float(abs(delta_shares) * next_price),
                    'transaction_cost': float(transaction_cost),
                }
            )

        prev_shares = shares
        cash = cash - delta_shares * next_price - transaction_cost
        shares = target_shares

        if shares == 0:
            entry_price = None
            entry_time = None
            holding_days = 0
        else:
            if entry_price is None or prev_shares == 0 or (prev_shares * shares < 0):
                entry_price = next_price
                entry_time = next_idx
                holding_days = 0
            holding_days += 1

        new_value = cash + shares * next_price
        position_value = shares * next_price
        daily_return = (new_value / portfolio_value) - 1 if portfolio_value != 0 else 0.0

        returns.append((next_idx, daily_return))
        equity.append((next_idx, new_value, position_value))

    returns_series = pd.Series({d: r for d, r in returns}).sort_index()
    equity_curve = pd.DataFrame(
        equity,
        columns=['date', 'portfolio_value', 'position_value'],
    ).set_index('date')

    train_range = (train_df.index.min(), train_df.index.max())
    val_range = (val_df.index.min(), val_df.index.max())
    test_range = (test_df.index.min(), test_df.index.max())

    trades_df = pd.DataFrame(trades)
    benchmark_returns = returns_from_prices(test_df['Close'])
    metrics = compute_metrics(
        returns_series,
        risk_free_rate=config.risk_free_rate,
        benchmark_returns=benchmark_returns,
        trades=trades_df,
        equity_curve=equity_curve,
    )

    return FoldResult(
        fold_id=fold_id,
        train_range=train_range,
        val_range=val_range,
        test_range=test_range,
        metrics=metrics,
        returns=returns_series,
        equity_curve=equity_curve,
        trades=trades_df,
    )


def _run_single_fold_portfolio(
    data_map: Dict[str, pd.DataFrame],
    symbols: List[str],
    strategy_cls: Type[Strategy],
    strategy_config: StrategyConfig,
    fold_id: int,
    train_slice: slice,
    val_slice: slice,
    test_slice: slice,
    config: WalkForwardConfig,
    common_index: pd.DatetimeIndex,
) -> FoldResult:
    strategies: Dict[str, Strategy] = {}
    for symbol in symbols:
        strategy = strategy_cls(strategy_config)
        strategy.fit(data_map[symbol].iloc[train_slice])
        strategies[symbol] = strategy

    cost_rate = (config.cost_bps / 10000.0) / 2.0
    cash = config.initial_capital
    shares: Dict[str, float] = {symbol: 0.0 for symbol in symbols}
    entry_price: Dict[str, Optional[float]] = {symbol: None for symbol in symbols}
    entry_time: Dict[str, Optional[pd.Timestamp]] = {symbol: None for symbol in symbols}
    holding_days: Dict[str, int] = {symbol: 0 for symbol in symbols}

    returns: List[Tuple[pd.Timestamp, float]] = []
    equity: List[Tuple[pd.Timestamp, float, float]] = []
    trades: List[Dict[str, Any]] = []

    test_index = common_index[test_slice]

    for idx in range(0, len(test_index) - 1):
        current_idx = test_index[idx]
        next_idx = test_index[idx + 1]

        prices_now = {symbol: data_map[symbol].loc[current_idx, 'Close'] for symbol in symbols}
        prices_next = {symbol: data_map[symbol].loc[next_idx, 'Close'] for symbol in symbols}

        portfolio_value = cash + sum(shares[symbol] * prices_now[symbol] for symbol in symbols)

        current_positions: List[Position] = []
        for symbol in symbols:
            if shares[symbol] == 0:
                continue
            current_positions.append(
                Position(
                    symbol=symbol,
                    size=shares[symbol] * prices_now[symbol],
                    entry_price=entry_price[symbol] or prices_now[symbol],
                    entry_time=entry_time[symbol] or current_idx,
                    current_price=prices_now[symbol],
                    unrealized_pnl=(prices_now[symbol] - (entry_price[symbol] or prices_now[symbol])) * shares[symbol],
                    holding_days=holding_days[symbol],
                )
            )

        target_values: Dict[str, float] = {}
        for symbol in symbols:
            history = data_map[symbol].loc[:current_idx]
            signal = strategies[symbol].generate_signal(history)
            target_values[symbol] = strategies[symbol].calculate_position_size(
                signal,
                portfolio_value,
                current_positions,
            )

        target_values = _apply_portfolio_constraints(target_values, portfolio_value, config)

        gross_position_value = 0.0
        for symbol in symbols:
            next_price = prices_next[symbol]
            target_value = target_values.get(symbol, 0.0)
            target_shares = target_value / next_price if next_price != 0 else 0.0
            delta_shares = target_shares - shares[symbol]
            transaction_cost = abs(delta_shares) * next_price * cost_rate

            if delta_shares != 0:
                trades.append(
                    {
                        'date': next_idx,
                        'symbol': symbol,
                        'price': float(next_price),
                        'shares': float(delta_shares),
                        'direction': 'BUY' if delta_shares > 0 else 'SELL',
                        'notional': float(abs(delta_shares) * next_price),
                        'transaction_cost': float(transaction_cost),
                    }
                )

            prev_shares = shares[symbol]
            cash = cash - delta_shares * next_price - transaction_cost
            shares[symbol] = target_shares

            if shares[symbol] == 0:
                entry_price[symbol] = None
                entry_time[symbol] = None
                holding_days[symbol] = 0
            else:
                if entry_price[symbol] is None or prev_shares == 0 or (prev_shares * shares[symbol] < 0):
                    entry_price[symbol] = next_price
                    entry_time[symbol] = next_idx
                    holding_days[symbol] = 0
                holding_days[symbol] += 1

            gross_position_value += abs(shares[symbol] * next_price)

        new_value = cash + sum(shares[symbol] * prices_next[symbol] for symbol in symbols)
        daily_return = (new_value / portfolio_value) - 1 if portfolio_value != 0 else 0.0

        returns.append((next_idx, daily_return))
        equity.append((next_idx, new_value, gross_position_value))

    returns_series = pd.Series({d: r for d, r in returns}).sort_index()
    equity_curve = pd.DataFrame(
        equity,
        columns=['date', 'portfolio_value', 'position_value'],
    ).set_index('date')

    train_range = (common_index[train_slice.start], common_index[train_slice.stop - 1])
    val_range = (common_index[val_slice.start], common_index[val_slice.stop - 1])
    test_range = (common_index[test_slice.start], common_index[test_slice.stop - 1])

    trades_df = pd.DataFrame(trades)
    metrics = compute_metrics(
        returns_series,
        risk_free_rate=config.risk_free_rate,
        trades=trades_df,
        equity_curve=equity_curve,
    )

    return FoldResult(
        fold_id=fold_id,
        train_range=train_range,
        val_range=val_range,
        test_range=test_range,
        metrics=metrics,
        returns=returns_series,
        equity_curve=equity_curve,
        trades=trades_df,
    )


def _run_folds_with_config_portfolio(
    data_map: Dict[str, pd.DataFrame],
    symbols: List[str],
    strategy_cls: Type[Strategy],
    strategy_config: StrategyConfig,
    config: WalkForwardConfig,
    folds: List[Tuple[int, int, int, int, int]],
    common_index: pd.DatetimeIndex,
    require_min_folds: bool = True,
) -> Tuple[List[FoldResult], pd.Series, pd.DataFrame, pd.DataFrame]:
    fold_results: List[FoldResult] = []

    for fold_id, (train_start_idx, train_end_idx, val_end_idx, test_start_idx, test_end_idx) in enumerate(folds, start=1):
        train_slice = slice(train_start_idx, train_end_idx + 1)
        val_slice = slice(train_end_idx + 1, val_end_idx + 1)
        test_slice = slice(test_start_idx, test_end_idx + 1)

        train_len = train_slice.stop - train_slice.start
        val_len = val_slice.stop - val_slice.start
        test_len = test_slice.stop - test_slice.start

        if train_len < config.min_train_days or val_len < config.min_val_days or test_len < config.min_test_days:
            continue

        fold_result = _run_single_fold_portfolio(
            data_map,
            symbols,
            strategy_cls,
            strategy_config,
            fold_id,
            train_slice,
            val_slice,
            test_slice,
            config,
            common_index,
        )
        fold_results.append(fold_result)

    if require_min_folds and len(fold_results) < config.min_folds:
        raise ValueError(
            f"Insufficient folds for walk-forward backtest: {len(fold_results)} < {config.min_folds}"
        )

    if fold_results:
        aggregate_returns = pd.concat([f.returns for f in fold_results]).sort_index()
        aggregate_equity = pd.concat([f.equity_curve for f in fold_results]).sort_index()
        trades_list = [f.trades for f in fold_results if f.trades is not None and not f.trades.empty]
        aggregate_trades = pd.concat(trades_list, ignore_index=True) if trades_list else pd.DataFrame()
    else:
        aggregate_returns = pd.Series(dtype=float)
        aggregate_equity = pd.DataFrame()
        aggregate_trades = pd.DataFrame()

    return fold_results, aggregate_returns, aggregate_equity, aggregate_trades


def _run_folds_with_config(
    data: pd.DataFrame,
    symbol: str,
    strategy_cls: Type[Strategy],
    strategy_config: StrategyConfig,
    config: WalkForwardConfig,
    folds: List[Tuple[int, int, int, int, int]],
    require_min_folds: bool = True,
) -> Tuple[List[FoldResult], pd.Series, pd.DataFrame, pd.DataFrame]:
    fold_results: List[FoldResult] = []

    for fold_id, (train_start_idx, train_end_idx, val_end_idx, test_start_idx, test_end_idx) in enumerate(folds, start=1):
        train_slice = slice(train_start_idx, train_end_idx + 1)
        val_slice = slice(train_end_idx + 1, val_end_idx + 1)
        test_slice = slice(test_start_idx, test_end_idx + 1)

        if test_slice.start >= len(data):
            break

        train_len = train_slice.stop - train_slice.start
        val_len = val_slice.stop - val_slice.start
        test_len = test_slice.stop - test_slice.start

        if train_len < config.min_train_days or val_len < config.min_val_days or test_len < config.min_test_days:
            continue

        fold_result = _run_single_fold(
            data,
            symbol,
            strategy_cls,
            strategy_config,
            fold_id,
            train_slice,
            val_slice,
            test_slice,
            config,
        )
        fold_results.append(fold_result)

    if require_min_folds and len(fold_results) < config.min_folds:
        raise ValueError(
            f"Insufficient folds for walk-forward backtest: {len(fold_results)} < {config.min_folds}"
        )

    if fold_results:
        aggregate_returns = pd.concat([f.returns for f in fold_results]).sort_index()
        aggregate_equity = pd.concat([f.equity_curve for f in fold_results]).sort_index()
        trades_list = [f.trades for f in fold_results if f.trades is not None and not f.trades.empty]
        aggregate_trades = pd.concat(trades_list, ignore_index=True) if trades_list else pd.DataFrame()
    else:
        aggregate_returns = pd.Series(dtype=float)
        aggregate_equity = pd.DataFrame()
        aggregate_trades = pd.DataFrame()

    return fold_results, aggregate_returns, aggregate_equity, aggregate_trades


def _slice_trades_by_index(trades: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    if trades is None or trades.empty or 'date' not in trades.columns or index is None or len(index) == 0:
        return pd.DataFrame()
    start = index.min()
    end = index.max()
    mask = (trades['date'] >= start) & (trades['date'] <= end)
    return trades.loc[mask]


def _slice_equity_by_index(equity_curve: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    if equity_curve is None or equity_curve.empty or index is None or len(index) == 0:
        return pd.DataFrame()
    if isinstance(equity_curve.index, pd.DatetimeIndex):
        equity = equity_curve
        if equity.index.has_duplicates:
            equity = equity.groupby(level=0).mean()
        return equity.reindex(index).dropna()
    return equity_curve.copy()


def _estimate_breakeven_cost(cost_levels: Dict[str, Dict[str, float]], metric_key: str = 'cagr') -> Optional[float]:
    points: List[Tuple[float, float]] = []
    for cost_str, metrics in cost_levels.items():
        try:
            cost = float(cost_str)
        except (TypeError, ValueError):
            continue
        if metrics is None or metric_key not in metrics:
            continue
        try:
            value = float(metrics[metric_key])
        except (TypeError, ValueError):
            continue
        points.append((cost, value))

    if len(points) < 2:
        return None

    points.sort(key=lambda x: x[0])
    for (cost_a, val_a), (cost_b, val_b) in zip(points, points[1:]):
        if val_a == 0:
            return float(cost_a)
        if (val_a > 0 and val_b < 0) or (val_a < 0 and val_b > 0):
            if val_b == val_a:
                return None
            return float(cost_a + (0 - val_a) * (cost_b - cost_a) / (val_b - val_a))
    return None


def _subperiod_metrics(
    returns: pd.Series,
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    risk_free_rate: float,
    benchmark_returns: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    if returns is None or len(returns) < 2:
        return {"error": "insufficient returns"}

    split_idx = len(returns) // 2
    first_returns = returns.iloc[:split_idx]
    second_returns = returns.iloc[split_idx:]

    first_trades = _slice_trades_by_index(trades, first_returns.index)
    second_trades = _slice_trades_by_index(trades, second_returns.index)

    first_equity = _slice_equity_by_index(equity_curve, first_returns.index)
    second_equity = _slice_equity_by_index(equity_curve, second_returns.index)

    first_metrics = compute_metrics(
        first_returns,
        risk_free_rate=risk_free_rate,
        benchmark_returns=benchmark_returns,
        trades=first_trades,
        equity_curve=first_equity,
    )
    second_metrics = compute_metrics(
        second_returns,
        risk_free_rate=risk_free_rate,
        benchmark_returns=benchmark_returns,
        trades=second_trades,
        equity_curve=second_equity,
    )

    cagr_baseline = first_metrics.get('cagr', 0.0)
    sharpe_baseline = first_metrics.get('sharpe_ratio', 0.0)
    cagr_drop = None if cagr_baseline == 0 else float((second_metrics.get('cagr', 0.0) - cagr_baseline) / abs(cagr_baseline))
    sharpe_drop = None if sharpe_baseline == 0 else float(
        (second_metrics.get('sharpe_ratio', 0.0) - sharpe_baseline) / abs(sharpe_baseline)
    )

    flag_overfitting = False
    if cagr_baseline != 0:
        flag_overfitting = second_metrics.get('cagr', 0.0) < cagr_baseline * 0.5
    if sharpe_baseline != 0:
        flag_overfitting = flag_overfitting or second_metrics.get('sharpe_ratio', 0.0) < sharpe_baseline * 0.5

    return {
        'first_half': {
            'start': first_returns.index.min(),
            'end': first_returns.index.max(),
            'metrics': first_metrics,
        },
        'second_half': {
            'start': second_returns.index.min(),
            'end': second_returns.index.max(),
            'metrics': second_metrics,
        },
        'degradation': {
            'cagr_pct': cagr_drop,
            'sharpe_pct': sharpe_drop,
        },
        'flag_overfitting': flag_overfitting,
    }


def _regime_metrics(
    returns: pd.Series,
    data: pd.DataFrame,
    ma_window: int,
    risk_free_rate: float,
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    benchmark_returns: Optional[pd.Series] = None,
    source_label: str = 'symbol',
) -> Dict[str, Any]:
    if returns is None or returns.empty:
        return {"error": "insufficient returns"}
    if 'Close' not in data.columns:
        return {"error": "missing Close prices for regime analysis"}
    if returns.index.has_duplicates:
        returns = returns.groupby(level=0).mean()

    ma = data['Close'].rolling(ma_window).mean()
    aligned = pd.concat([returns, data['Close'], ma], axis=1).dropna()
    returns_aligned = aligned.iloc[:, 0]
    price = aligned.iloc[:, 1]
    ma_aligned = aligned.iloc[:, 2]
    bull_mask = price > ma_aligned
    bear_mask = price <= ma_aligned

    bull_returns = returns_aligned[bull_mask]
    bear_returns = returns_aligned[bear_mask]

    bull_trades = _slice_trades_by_index(trades, bull_returns.index)
    bear_trades = _slice_trades_by_index(trades, bear_returns.index)
    bull_equity = _slice_equity_by_index(equity_curve, bull_returns.index)
    bear_equity = _slice_equity_by_index(equity_curve, bear_returns.index)

    return {
        'source': source_label,
        'ma_window': ma_window,
        'bull': {
            'days': int(len(bull_returns)),
            'metrics': compute_metrics(
                bull_returns,
                risk_free_rate=risk_free_rate,
                benchmark_returns=benchmark_returns,
                trades=bull_trades,
                equity_curve=bull_equity,
            ),
        },
        'bear': {
            'days': int(len(bear_returns)),
            'metrics': compute_metrics(
                bear_returns,
                risk_free_rate=risk_free_rate,
                benchmark_returns=benchmark_returns,
                trades=bear_trades,
                equity_curve=bear_equity,
            ),
        },
    }


def _parameter_sensitivity_metrics(
    data: pd.DataFrame,
    symbol: str,
    strategy_cls: Type[Strategy],
    strategy_config: StrategyConfig,
    config: WalkForwardConfig,
    folds: List[Tuple[int, int, int, int, int]],
    base_metrics: Dict[str, float],
    benchmark_returns: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for param_name, param_value in strategy_config.parameters.items():
        if isinstance(param_value, bool) or not isinstance(param_value, (int, float)) or param_value == 0:
            continue
        results[param_name] = {
            'base_value': param_value,
            'variants': {},
        }
        for scale, label in [(1 - config.parameter_sensitivity_pct, 'down_20'),
                             (1 + config.parameter_sensitivity_pct, 'up_20')]:
            if isinstance(param_value, int):
                new_value = max(1, int(round(param_value * scale)))
            else:
                new_value = param_value * scale
            new_params = dict(strategy_config.parameters)
            new_params[param_name] = new_value
            variant_config = replace(strategy_config, parameters=new_params)
            try:
                _, variant_returns, variant_equity, variant_trades = _run_folds_with_config(
                    data,
                    symbol,
                    strategy_cls,
                    variant_config,
                    config,
                    folds,
                    require_min_folds=False,
                )
                variant_metrics = compute_metrics(
                    variant_returns,
                    risk_free_rate=config.risk_free_rate,
                    benchmark_returns=benchmark_returns,
                    trades=variant_trades,
                    equity_curve=variant_equity,
                )
                results[param_name]['variants'][label] = {
                    'value': new_value,
                    'metrics': variant_metrics,
                    'delta': {
                        'cagr_pct': None if base_metrics.get('cagr', 0.0) == 0 else float(
                            (variant_metrics.get('cagr', 0.0) - base_metrics.get('cagr', 0.0))
                            / abs(base_metrics.get('cagr', 0.0))
                        ),
                        'sharpe_pct': None if base_metrics.get('sharpe_ratio', 0.0) == 0 else float(
                            (variant_metrics.get('sharpe_ratio', 0.0) - base_metrics.get('sharpe_ratio', 0.0))
                            / abs(base_metrics.get('sharpe_ratio', 0.0))
                        ),
                    },
                }
            except Exception as exc:
                results[param_name]['variants'][label] = {'value': new_value, 'error': str(exc)}
        cliff_flag = False
        base_cagr = base_metrics.get('cagr', 0.0)
        base_sharpe = base_metrics.get('sharpe_ratio', 0.0)
        for variant in results[param_name]['variants'].values():
            metrics = variant.get('metrics')
            if not metrics:
                continue
            if base_cagr != 0 and metrics.get('cagr', 0.0) < base_cagr * 0.5:
                cliff_flag = True
            if base_sharpe != 0 and metrics.get('sharpe_ratio', 0.0) < base_sharpe * 0.5:
                cliff_flag = True
        results[param_name]['flag_performance_cliff'] = cliff_flag
    return results


def _parameter_sensitivity_metrics_portfolio(
    data_map: Dict[str, pd.DataFrame],
    symbols: List[str],
    strategy_cls: Type[Strategy],
    strategy_config: StrategyConfig,
    config: WalkForwardConfig,
    folds: List[Tuple[int, int, int, int, int]],
    common_index: pd.DatetimeIndex,
    base_metrics: Dict[str, float],
    benchmark_returns: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for param_name, param_value in strategy_config.parameters.items():
        if isinstance(param_value, bool) or not isinstance(param_value, (int, float)) or param_value == 0:
            continue
        results[param_name] = {
            'base_value': param_value,
            'variants': {},
        }
        for scale, label in [(1 - config.parameter_sensitivity_pct, 'down_20'),
                             (1 + config.parameter_sensitivity_pct, 'up_20')]:
            if isinstance(param_value, int):
                new_value = max(1, int(round(param_value * scale)))
            else:
                new_value = param_value * scale
            new_params = dict(strategy_config.parameters)
            new_params[param_name] = new_value
            variant_config = replace(strategy_config, parameters=new_params)
            try:
                _, variant_returns, variant_equity, variant_trades = _run_folds_with_config_portfolio(
                    data_map,
                    symbols,
                    strategy_cls,
                    variant_config,
                    config,
                    folds,
                    common_index,
                    require_min_folds=False,
                )
                variant_metrics = compute_metrics(
                    variant_returns,
                    risk_free_rate=config.risk_free_rate,
                    benchmark_returns=benchmark_returns,
                    trades=variant_trades,
                    equity_curve=variant_equity,
                )
                results[param_name]['variants'][label] = {
                    'value': new_value,
                    'metrics': variant_metrics,
                    'delta': {
                        'cagr_pct': None if base_metrics.get('cagr', 0.0) == 0 else float(
                            (variant_metrics.get('cagr', 0.0) - base_metrics.get('cagr', 0.0))
                            / abs(base_metrics.get('cagr', 0.0))
                        ),
                        'sharpe_pct': None if base_metrics.get('sharpe_ratio', 0.0) == 0 else float(
                            (variant_metrics.get('sharpe_ratio', 0.0) - base_metrics.get('sharpe_ratio', 0.0))
                            / abs(base_metrics.get('sharpe_ratio', 0.0))
                        ),
                    },
                }
            except Exception as exc:
                results[param_name]['variants'][label] = {'value': new_value, 'error': str(exc)}
        cliff_flag = False
        base_cagr = base_metrics.get('cagr', 0.0)
        base_sharpe = base_metrics.get('sharpe_ratio', 0.0)
        for variant in results[param_name]['variants'].values():
            metrics = variant.get('metrics')
            if not metrics:
                continue
            if base_cagr != 0 and metrics.get('cagr', 0.0) < base_cagr * 0.5:
                cliff_flag = True
            if base_sharpe != 0 and metrics.get('sharpe_ratio', 0.0) < base_sharpe * 0.5:
                cliff_flag = True
        results[param_name]['flag_performance_cliff'] = cliff_flag
    return results


def run_walk_forward(
    data: pd.DataFrame,
    symbol: str,
    strategy_cls: Type[Strategy],
    strategy_config: StrategyConfig,
    config: Optional[WalkForwardConfig] = None,
) -> BacktestResult:
    """Run walk-forward backtest for a given strategy and data."""
    config = config or WalkForwardConfig()

    data = data.copy()
    data = data.sort_index()
    data = data[REQUIRED_COLUMNS].dropna()
    data.attrs['symbol'] = symbol

    folds = list(_generate_folds(data.index, config))

    fold_results, aggregate_returns, aggregate_equity, aggregate_trades = _run_folds_with_config(
        data,
        symbol,
        strategy_cls,
        strategy_config,
        config,
        folds,
    )

    benchmark_returns: Dict[str, pd.Series] = {
        'buy_hold': returns_from_prices(data['Close']),
    }
    spy_data: Optional[pd.DataFrame] = None
    try:
        start_str = data.index.min().strftime('%Y-%m-%d')
        end_str = data.index.max().strftime('%Y-%m-%d')
        spy_data = load_ohlcv_data('SPY', start_str, end_str)
        benchmark_returns['spy'] = returns_from_prices(spy_data['Close'])
    except Exception:
        pass

    primary_benchmark = benchmark_returns.get('spy', benchmark_returns.get('buy_hold'))

    aggregate_metrics = compute_metrics(
        aggregate_returns,
        risk_free_rate=config.risk_free_rate,
        benchmark_returns=primary_benchmark,
        trades=aggregate_trades,
        equity_curve=aggregate_equity,
    )

    benchmarks: Dict[str, Any] = {}
    for name, stream in benchmark_returns.items():
        if stream is not None and not stream.empty:
            aligned_stream = stream.reindex(aggregate_returns.index).dropna()
            benchmarks[name] = compute_metrics(aligned_stream, risk_free_rate=config.risk_free_rate)

    if not aggregate_returns.empty:
        rf_daily = (1 + config.risk_free_rate) ** (1 / 252) - 1
        rf_stream = pd.Series(rf_daily, index=aggregate_returns.index)
        benchmarks['risk_free'] = compute_metrics(rf_stream, risk_free_rate=config.risk_free_rate)

    cost_levels: Dict[str, Dict[str, float]] = {}
    if config.enable_cost_sensitivity:
        for cost_bps in sorted({float(x) for x in config.cost_sensitivity_bps}):
            if cost_bps == float(config.cost_bps):
                cost_levels[str(cost_bps)] = aggregate_metrics
                continue
            config_override = replace(config, cost_bps=cost_bps)
            _, cost_returns, cost_equity, cost_trades = _run_folds_with_config(
                data,
                symbol,
                strategy_cls,
                strategy_config,
                config_override,
                folds,
            )
            cost_metrics = compute_metrics(
                cost_returns,
                risk_free_rate=config_override.risk_free_rate,
                benchmark_returns=primary_benchmark,
                trades=cost_trades,
                equity_curve=cost_equity,
            )
            cost_levels[str(cost_bps)] = cost_metrics

    cost_sensitivity = None
    if cost_levels:
        cost_sensitivity = {
            'levels': cost_levels,
            'breakeven_cost_bps': _estimate_breakeven_cost(cost_levels),
        }

    robustness: Optional[Dict[str, Any]] = None
    if config.enable_robustness and not aggregate_returns.empty:
        robustness = {
            'subperiod': _subperiod_metrics(
                aggregate_returns,
                aggregate_trades,
                aggregate_equity,
                config.risk_free_rate,
                benchmark_returns=primary_benchmark,
            ),
            'regime': _regime_metrics(
                aggregate_returns,
                spy_data if spy_data is not None else data,
                config.regime_ma_window,
                config.risk_free_rate,
                aggregate_trades,
                aggregate_equity,
                benchmark_returns=primary_benchmark,
                source_label='spy' if spy_data is not None else 'symbol',
            ),
        }

        if config.enable_parameter_sensitivity:
            param_sensitivity = _parameter_sensitivity_metrics(
                data,
                symbol,
                strategy_cls,
                strategy_config,
                config,
                folds,
                aggregate_metrics,
                benchmark_returns=primary_benchmark,
            )
            if param_sensitivity:
                robustness['parameter_sensitivity'] = param_sensitivity

    return BacktestResult(
        symbol=symbol,
        config=config,
        fold_results=fold_results,
        aggregate_metrics=aggregate_metrics,
        aggregate_equity_curve=aggregate_equity,
        symbols=[symbol],
        benchmark_returns=benchmark_returns,
        benchmarks=benchmarks or None,
        cost_sensitivity=cost_sensitivity,
        robustness=robustness,
    )


def run_walk_forward_portfolio(
    data_map: Dict[str, pd.DataFrame],
    symbols: List[str],
    strategy_cls: Type[Strategy],
    strategy_config: StrategyConfig,
    config: Optional[WalkForwardConfig] = None,
) -> BacktestResult:
    config = config or WalkForwardConfig()

    if not symbols:
        raise ValueError("Portfolio backtest requires at least one symbol")

    aligned_map: Dict[str, pd.DataFrame] = {}
    common_index: Optional[pd.DatetimeIndex] = None
    for symbol in symbols:
        if symbol not in data_map:
            raise ValueError(f"Missing data for symbol {symbol}")
        df = data_map[symbol].copy().sort_index()
        df = df[REQUIRED_COLUMNS].dropna()
        df.attrs['symbol'] = symbol
        aligned_map[symbol] = df
        common_index = df.index if common_index is None else common_index.intersection(df.index)

    if common_index is None or common_index.empty:
        raise ValueError("No overlapping dates across portfolio symbols")

    for symbol, df in aligned_map.items():
        aligned_map[symbol] = df.reindex(common_index).dropna()

    folds = list(_generate_folds(common_index, config))

    fold_results, aggregate_returns, aggregate_equity, aggregate_trades = _run_folds_with_config_portfolio(
        aligned_map,
        symbols,
        strategy_cls,
        strategy_config,
        config,
        folds,
        common_index,
    )

    benchmark_returns: Dict[str, pd.Series] = {}
    if symbols:
        returns_list = []
        for symbol in symbols:
            returns_list.append(returns_from_prices(aligned_map[symbol]['Close']))
        aligned_returns = pd.concat(returns_list, axis=1).dropna()
        if not aligned_returns.empty:
            benchmark_returns['buy_hold'] = aligned_returns.mean(axis=1)

    spy_data = None
    try:
        start_str = common_index.min().strftime('%Y-%m-%d')
        end_str = common_index.max().strftime('%Y-%m-%d')
        spy_data = load_ohlcv_data('SPY', start_str, end_str)
        benchmark_returns['spy'] = returns_from_prices(spy_data['Close'])
    except Exception:
        spy_data = None

    primary_benchmark = benchmark_returns.get('spy', benchmark_returns.get('buy_hold'))

    aggregate_metrics = compute_metrics(
        aggregate_returns,
        risk_free_rate=config.risk_free_rate,
        benchmark_returns=primary_benchmark,
        trades=aggregate_trades,
        equity_curve=aggregate_equity,
    )

    benchmarks: Dict[str, Any] = {}
    for name, stream in benchmark_returns.items():
        if stream is not None and not stream.empty:
            benchmarks[name] = compute_metrics(stream, risk_free_rate=config.risk_free_rate)

    if not aggregate_returns.empty:
        rf_daily = (1 + config.risk_free_rate) ** (1 / 252) - 1
        rf_stream = pd.Series(rf_daily, index=aggregate_returns.index)
        benchmarks['risk_free'] = compute_metrics(rf_stream, risk_free_rate=config.risk_free_rate)

    cost_levels: Dict[str, Dict[str, float]] = {}
    if config.enable_cost_sensitivity:
        for cost_bps in sorted({float(x) for x in config.cost_sensitivity_bps}):
            if cost_bps == float(config.cost_bps):
                cost_levels[str(cost_bps)] = aggregate_metrics
                continue
            config_override = replace(config, cost_bps=cost_bps)
            _, cost_returns, cost_equity, cost_trades = _run_folds_with_config_portfolio(
                aligned_map,
                symbols,
                strategy_cls,
                strategy_config,
                config_override,
                folds,
                common_index,
            )
            cost_metrics = compute_metrics(
                cost_returns,
                risk_free_rate=config_override.risk_free_rate,
                benchmark_returns=primary_benchmark,
                trades=cost_trades,
                equity_curve=cost_equity,
            )
            cost_levels[str(cost_bps)] = cost_metrics

    cost_sensitivity = None
    if cost_levels:
        cost_sensitivity = {
            'levels': cost_levels,
            'breakeven_cost_bps': _estimate_breakeven_cost(cost_levels),
        }

    robustness: Optional[Dict[str, Any]] = None
    if config.enable_robustness and not aggregate_returns.empty:
        if spy_data is not None:
            regime_data = spy_data
        else:
            price_df = pd.concat([aligned_map[s]['Close'] for s in symbols], axis=1).dropna()
            normalized = price_df / price_df.iloc[0]
            proxy_close = normalized.mean(axis=1)
            regime_data = pd.DataFrame({'Close': proxy_close})

        robustness = {
            'subperiod': _subperiod_metrics(
                aggregate_returns,
                aggregate_trades,
                aggregate_equity,
                config.risk_free_rate,
                benchmark_returns=primary_benchmark,
            ),
            'regime': _regime_metrics(
                aggregate_returns,
                regime_data,
                config.regime_ma_window,
                config.risk_free_rate,
                aggregate_trades,
                aggregate_equity,
                benchmark_returns=primary_benchmark,
            ),
        }

        if config.enable_parameter_sensitivity:
            param_sensitivity = _parameter_sensitivity_metrics_portfolio(
                aligned_map,
                symbols,
                strategy_cls,
                strategy_config,
                config,
                folds,
                common_index,
                aggregate_metrics,
                benchmark_returns=primary_benchmark,
            )
            if param_sensitivity:
                robustness['parameter_sensitivity'] = param_sensitivity

    return BacktestResult(
        symbol="PORTFOLIO",
        config=config,
        fold_results=fold_results,
        aggregate_metrics=aggregate_metrics,
        aggregate_equity_curve=aggregate_equity,
        symbols=symbols,
        benchmark_returns=benchmark_returns,
        benchmarks=benchmarks or None,
        cost_sensitivity=cost_sensitivity,
        robustness=robustness,
    )
