"""Performance metrics for backtesting."""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd


def _to_series(returns) -> pd.Series:
    if isinstance(returns, pd.Series):
        return returns.dropna()
    return pd.Series(returns).dropna()


def _to_price_series(prices) -> pd.Series:
    if isinstance(prices, pd.Series):
        series = prices
    elif isinstance(prices, pd.DataFrame):
        if 'Close' in prices.columns:
            series = prices['Close']
        elif len(prices.columns) == 1:
            series = prices.iloc[:, 0]
        else:
            raise ValueError("Price DataFrame must contain 'Close' or a single column")
    else:
        series = pd.Series(prices)
    return series.astype(float).dropna().sort_index()


def _to_equity_series(equity_curve) -> pd.Series:
    if equity_curve is None:
        return pd.Series(dtype=float)
    if isinstance(equity_curve, pd.Series):
        series = equity_curve
    elif isinstance(equity_curve, pd.DataFrame):
        if 'portfolio_value' in equity_curve.columns:
            series = equity_curve['portfolio_value']
        elif len(equity_curve.columns) == 1:
            series = equity_curve.iloc[:, 0]
        else:
            raise ValueError("Equity curve must contain 'portfolio_value' or a single column")
    else:
        series = pd.Series(equity_curve)
    series = series.astype(float).dropna().sort_index()
    if series.index.has_duplicates:
        series = series.groupby(level=0).mean()
    return series


def returns_from_prices(prices) -> pd.Series:
    series = _to_price_series(prices)
    if len(series) < 2:
        return pd.Series(dtype=float)
    return series.pct_change().dropna()


def compute_buy_hold_returns(prices) -> pd.Series:
    """Compute buy-and-hold returns from price data."""
    return returns_from_prices(prices)


def align_returns(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    if strategy_returns.index.has_duplicates:
        strategy_returns = strategy_returns.groupby(level=0).mean()
    if benchmark_returns.index.has_duplicates:
        benchmark_returns = benchmark_returns.groupby(level=0).mean()
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    if aligned.empty:
        return strategy_returns.iloc[:0], benchmark_returns.iloc[:0]
    return aligned.iloc[:, 0], aligned.iloc[:, 1]


def cagr(returns, periods_per_year: int = 252) -> float:
    series = _to_series(returns)
    if len(series) == 0:
        return 0.0
    total_return = (1 + series).prod() - 1
    years = len(series) / periods_per_year
    if years <= 0:
        return 0.0
    return float((1 + total_return) ** (1 / years) - 1)


def volatility(returns, periods_per_year: int = 252) -> float:
    series = _to_series(returns)
    if len(series) == 0:
        return 0.0
    return float(series.std() * np.sqrt(periods_per_year))


def sharpe_ratio(returns, risk_free_rate: float = 0.04, periods_per_year: int = 252) -> float:
    vol = volatility(returns, periods_per_year)
    if vol == 0:
        return 0.0
    return float((cagr(returns, periods_per_year) - risk_free_rate) / vol)


def sortino_ratio(returns, risk_free_rate: float = 0.04, periods_per_year: int = 252) -> float:
    series = _to_series(returns)
    if len(series) == 0:
        return 0.0
    downside = series[series < 0]
    if len(downside) == 0:
        return float('inf')
    downside_vol = downside.std() * np.sqrt(periods_per_year)
    if downside_vol == 0:
        return 0.0
    return float((cagr(series, periods_per_year) - risk_free_rate) / downside_vol)


def max_drawdown(returns) -> float:
    series = _to_series(returns)
    if len(series) == 0:
        return 0.0
    cumulative = (1 + series).cumprod()
    rolling_max = cumulative.cummax()
    drawdowns = cumulative / rolling_max - 1
    return float(drawdowns.min())


def calmar_ratio(returns, risk_free_rate: float = 0.04, periods_per_year: int = 252) -> float:
    dd = max_drawdown(returns)
    if dd == 0:
        return 0.0
    return float(cagr(returns, periods_per_year) / abs(dd))


def information_ratio(returns, benchmark_returns, periods_per_year: int = 252) -> float:
    strat, bench = align_returns(_to_series(returns), _to_series(benchmark_returns))
    if len(strat) == 0:
        return 0.0
    excess = strat - bench
    tracking_error = excess.std() * np.sqrt(periods_per_year)
    if tracking_error == 0:
        return 0.0
    return float(excess.mean() * periods_per_year / tracking_error)


def beta_alpha(returns, benchmark_returns, periods_per_year: int = 252) -> Tuple[float, float]:
    strat, bench = align_returns(_to_series(returns), _to_series(benchmark_returns))
    if len(strat) == 0:
        return 0.0, 0.0
    cov = strat.cov(bench)
    var = bench.var()
    beta = cov / var if var > 0 else 0.0
    alpha = cagr(strat, periods_per_year) - beta * cagr(bench, periods_per_year)
    return float(beta), float(alpha)


def _normalize_trades(trades: Optional[pd.DataFrame]) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    trades_df = trades.copy()
    if 'date' in trades_df.columns:
        trades_df['date'] = pd.to_datetime(trades_df['date'])
        trades_df = trades_df.set_index('date')
    if not isinstance(trades_df.index, pd.DatetimeIndex):
        trades_df.index = pd.to_datetime(trades_df.index)
    return trades_df.sort_index()


def _infer_position_values(trades: Optional[pd.DataFrame], equity_curve) -> pd.Series:
    equity = _to_equity_series(equity_curve)
    if equity.empty:
        return pd.Series(dtype=float)
    trades_df = _normalize_trades(trades)
    if trades_df.empty:
        return pd.Series(0.0, index=equity.index)

    if 'shares' in trades_df.columns and 'price' in trades_df.columns:
        signed_notional = trades_df['shares'].astype(float) * trades_df['price'].astype(float)
    elif 'notional' in trades_df.columns and 'direction' in trades_df.columns:
        direction = trades_df['direction'].astype(str).str.upper().map({'BUY': 1.0, 'SELL': -1.0}).fillna(0.0)
        signed_notional = trades_df['notional'].astype(float) * direction
    elif 'notional' in trades_df.columns:
        signed_notional = trades_df['notional'].astype(float)
    else:
        return pd.Series(0.0, index=equity.index)

    transaction_costs = (
        trades_df['transaction_cost'].astype(float) if 'transaction_cost' in trades_df.columns else 0.0
    )
    cash_flow = -signed_notional - transaction_costs

    daily_cash_flow = cash_flow.groupby(level=0).sum()
    daily_costs = (
        transaction_costs.groupby(level=0).sum() if isinstance(transaction_costs, pd.Series) else pd.Series(dtype=float)
    )

    first_date = equity.index.min()
    initial_cash = float(equity.iloc[0]) + float(daily_costs.get(first_date, 0.0))

    cash_flow_aligned = daily_cash_flow.reindex(equity.index, fill_value=0.0)
    cash_series = initial_cash + cash_flow_aligned.cumsum()

    return equity - cash_series


def turnover_from_trades(
    trades: Optional[pd.DataFrame],
    equity_curve: Optional[pd.DataFrame],
    periods_per_year: int = 252,
) -> float:
    equity = _to_equity_series(equity_curve)
    if equity.empty:
        return 0.0
    trades_df = _normalize_trades(trades)
    if trades_df.empty:
        return 0.0

    if 'notional' in trades_df.columns:
        daily_notional = trades_df['notional'].astype(float).abs()
    elif 'shares' in trades_df.columns and 'price' in trades_df.columns:
        daily_notional = (trades_df['shares'].astype(float) * trades_df['price'].astype(float)).abs()
    else:
        return 0.0

    daily_notional = daily_notional.groupby(level=0).sum()
    daily_notional = daily_notional.reindex(equity.index, fill_value=0.0)
    daily_turnover = (daily_notional / (2.0 * equity)).replace([np.inf, -np.inf], np.nan).dropna()
    if len(daily_turnover) == 0:
        return 0.0
    return float(daily_turnover.mean() * periods_per_year)


def exposure_from_equity_curve(
    equity_curve: Optional[pd.DataFrame],
    trades: Optional[pd.DataFrame] = None,
    position_values: Optional[pd.Series] = None,
) -> float:
    equity = _to_equity_series(equity_curve)
    if equity.empty:
        return 0.0

    if position_values is not None:
        position_series = _to_series(position_values)
        if position_series.index.has_duplicates:
            position_series = position_series.groupby(level=0).mean()
        position = position_series.reindex(equity.index).fillna(0.0)
    elif isinstance(equity_curve, pd.DataFrame) and 'position_value' in equity_curve.columns:
        position_series = equity_curve['position_value'].astype(float)
        if position_series.index.has_duplicates:
            position_series = position_series.groupby(level=0).mean()
        position = position_series.reindex(equity.index).fillna(0.0)
    else:
        position = _infer_position_values(trades, equity_curve)

    exposure_series = (position.abs() / equity).replace([np.inf, -np.inf], np.nan).dropna()
    if len(exposure_series) == 0:
        return 0.0
    return float(exposure_series.mean())


def benchmark_return_streams(
    price_series,
    benchmark_prices: Optional[Union[pd.Series, Dict[str, pd.Series]]] = None,
) -> Dict[str, pd.Series]:
    streams = {
        'buy_hold': returns_from_prices(price_series),
    }
    if benchmark_prices is None:
        return streams
    if isinstance(benchmark_prices, dict):
        for name, prices in benchmark_prices.items():
            streams[name] = returns_from_prices(prices)
    else:
        streams['benchmark'] = returns_from_prices(benchmark_prices)
    return streams


def compute_metrics(
    returns,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252,
    benchmark_returns: Optional[pd.Series] = None,
    trades: Optional[pd.DataFrame] = None,
    equity_curve: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Compute required backtest metrics."""
    series = _to_series(returns)

    metrics = {
        'cagr': cagr(series, periods_per_year),
        'volatility': volatility(series, periods_per_year),
        'sharpe_ratio': sharpe_ratio(series, risk_free_rate, periods_per_year),
        'sortino_ratio': sortino_ratio(series, risk_free_rate, periods_per_year),
        'max_drawdown': max_drawdown(series),
        'calmar_ratio': calmar_ratio(series, risk_free_rate, periods_per_year),
    }

    if benchmark_returns is not None:
        benchmark_series = _to_series(benchmark_returns)
        strat_aligned, bench_aligned = align_returns(series, benchmark_series)
        if len(bench_aligned) > 0:
            metrics['benchmark_cagr'] = cagr(bench_aligned, periods_per_year)
            metrics['benchmark_volatility'] = volatility(bench_aligned, periods_per_year)
            metrics['information_ratio'] = information_ratio(strat_aligned, bench_aligned, periods_per_year)
            beta, alpha = beta_alpha(strat_aligned, bench_aligned, periods_per_year)
            metrics['beta'] = beta
            metrics['alpha'] = alpha

    if trades is not None and equity_curve is not None:
        metrics['turnover'] = turnover_from_trades(trades, equity_curve, periods_per_year)
        metrics['exposure'] = exposure_from_equity_curve(equity_curve)

    return metrics
