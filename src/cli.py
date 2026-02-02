#!/usr/bin/env python3
"""
Unified CLI entrypoint for the ML-Stock_Trading framework.

Provides command-line interface for:
- Running backtests with different strategies
- Listing available strategies
- Viewing results and reports
"""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Type

import pandas as pd

from src.backtest.engine import (
    WalkForwardConfig,
    load_ohlcv_data,
    load_ohlcv_data_multi,
    run_walk_forward,
    run_walk_forward_portfolio,
)
from src.strategies.base import Strategy, StrategyConfig, StrategyType
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy


STRATEGY_REGISTRY: Dict[str, Tuple[Type[Strategy], StrategyConfig]] = {
    "momentum": (
        MomentumStrategy,
        StrategyConfig(
            name="tsmom_baseline",
            strategy_type=StrategyType.MOMENTUM,
            parameters={
                "formation_period": 252,
                "skip_period": 21,
                "vol_window": 60,
                "entry_threshold": 0.1,
            },
            risk_limits={"target_volatility": 0.15, "max_position": 2.0},
        ),
    ),
    "mean_reversion": (
        MeanReversionStrategy,
        StrategyConfig(
            name="mean_reversion_baseline",
            strategy_type=StrategyType.MEAN_REVERSION,
            parameters={
                "lookback": 20,
                "entry_z": 1.5,
                "exit_z": 0.5,
                "max_half_life": 20.0,
            },
            risk_limits={"target_volatility": 0.10, "max_position": 2.0},
        ),
    ),
}


def list_strategies() -> None:
    """List available trading strategies."""
    print("\n" + "=" * 70)
    print("Available Trading Strategies")
    print("=" * 70)

    for key, (_, config) in STRATEGY_REGISTRY.items():
        print(f"{key:.<20} {config.name}")
        print(f"{'':20} Type: {config.strategy_type.value}")
        print(f"{'':20} Parameters: {config.parameters}")

    print("\n" + "=" * 70)
    print("Usage: python run.py backtest --strategy <strategy_name> [options]")
    print("=" * 70 + "\n")


def _save_results(output_path: Path, result, strategy_name: str, args: argparse.Namespace) -> None:
    output_path.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "strategy": strategy_name,
        "symbol": result.symbol,
        "symbols": result.symbols if getattr(result, "symbols", None) else [result.symbol],
        "start_date": args.start_date,
        "end_date": args.end_date,
        "walk_forward": asdict(result.config),
    }

    with open(output_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    metrics_payload = {
        "aggregate": result.aggregate_metrics,
        "folds": {
            fold.fold_id: {
                "metrics": fold.metrics,
                "train_range": [fold.train_range[0].isoformat(), fold.train_range[1].isoformat()],
                "val_range": [fold.val_range[0].isoformat(), fold.val_range[1].isoformat()],
                "test_range": [fold.test_range[0].isoformat(), fold.test_range[1].isoformat()],
            }
            for fold in result.fold_results
        },
    }
    if getattr(result, "cost_sensitivity", None):
        metrics_payload["cost_sensitivity"] = result.cost_sensitivity
    if getattr(result, "robustness", None):
        metrics_payload["robustness"] = result.robustness

    with open(output_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, default=str)

    result.aggregate_equity_curve.to_csv(output_path / "equity_curve.csv")

    trades = []
    for fold in result.fold_results:
        if fold.trades is not None and not fold.trades.empty:
            fold_trades = fold.trades.copy()
            fold_trades["fold_id"] = fold.fold_id
            trades.append(fold_trades)
    if trades:
        trades_df = pd.concat(trades)
        trades_df.to_json(output_path / "trades.json", orient="records")

    report_lines = [
        "Walk-Forward Backtest Summary",
        "=" * 32,
        f"Strategy: {strategy_name}",
        f"Symbol: {result.symbol}",
        f"Period: {args.start_date} to {args.end_date}",
        "",
        "Aggregate Metrics:",
    ]
    if getattr(result, "symbols", None):
        report_lines.insert(4, f"Symbols: {', '.join(result.symbols)}")
    for key, value in result.aggregate_metrics.items():
        report_lines.append(f"  {key}: {value:.4f}")

    with open(output_path / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


def backtest(args: argparse.Namespace) -> None:
    """
    Execute a backtest with the specified strategy.

    Args:
        args: Parsed command-line arguments
    """
    print("\n" + "=" * 70)
    print("Walk-Forward Backtest")
    print("=" * 70)

    if args.strategy not in STRATEGY_REGISTRY:
        print(f"ERROR: Unknown strategy '{args.strategy}'")
        print(f"Available strategies: {', '.join(STRATEGY_REGISTRY.keys())}")
        sys.exit(1)

    try:
        start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
        if start_dt >= end_dt:
            raise ValueError("start-date must be before end-date")
    except ValueError as exc:
        print(f"ERROR: Invalid date format or range: {exc}")
        sys.exit(1)

    output_path = Path(args.output)

    strategy_cls, strategy_config = STRATEGY_REGISTRY[args.strategy]

    wf_config = WalkForwardConfig(
        train_years=args.train_years,
        val_months=args.val_months,
        test_months=args.test_months,
        step_months=args.step_months,
        embargo_days=args.embargo_days,
        cost_bps=args.cost_bps,
    )

    data_dir = Path(args.data_dir) if args.data_dir else None

    symbols: list[str] = []
    if getattr(args, "symbols_file", None):
        symbols_path = Path(args.symbols_file)
        if not symbols_path.exists():
            print(f"ERROR: symbols file not found: {symbols_path}")
            sys.exit(1)
        raw_lines = symbols_path.read_text(encoding="utf-8").splitlines()
        for line in raw_lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",") if p.strip()]
            symbols.extend(parts)
    elif getattr(args, "symbols", None):
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    elif args.symbol:
        if "," in args.symbol:
            symbols = [s.strip() for s in args.symbol.split(",") if s.strip()]
        else:
            symbols = [args.symbol]

    if not symbols:
        print("ERROR: Provide at least one symbol via --symbol or --symbols")
        sys.exit(1)

    try:
        if len(symbols) == 1:
            data = load_ohlcv_data(symbols[0], args.start_date, args.end_date, data_dir=data_dir)
            result = run_walk_forward(
                data=data,
                symbol=symbols[0],
                strategy_cls=strategy_cls,
                strategy_config=strategy_config,
                config=wf_config,
            )
        else:
            data_map, _ = load_ohlcv_data_multi(symbols, args.start_date, args.end_date, data_dir=data_dir)
            symbols = list(data_map.keys())
            result = run_walk_forward_portfolio(
                data_map=data_map,
                symbols=symbols,
                strategy_cls=strategy_cls,
                strategy_config=strategy_config,
                config=wf_config,
            )
    except Exception as exc:
        print(f"ERROR: Backtest failed: {exc}")
        sys.exit(1)

    _save_results(output_path, result, args.strategy, args)

    print("\nAggregate Metrics:")
    for key, value in result.aggregate_metrics.items():
        print(f"  {key:<15} {value:.4f}")

    print(f"\nResults saved to: {output_path.resolve()}")


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="ML-Stock_Trading Framework CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available strategies
  python run.py list-strategies

  # Run a backtest
  python run.py backtest \
    --strategy momentum \
    --symbol AAPL \
    --start-date 2020-01-01 \
    --end-date 2024-01-01 \
    --train-years 2 \
    --test-months 3 \
    --cost-bps 30 \
    --output ./results/

  # Portfolio backtest from a symbols file
  python run.py backtest \
    --strategy momentum \
    --symbols-file ./top500.txt \
    --start-date 2018-01-01 \
    --end-date 2023-01-01 \
    --output ./results_portfolio/
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    subparsers.add_parser(
        "list-strategies",
        help="List available trading strategies",
    )

    backtest_parser = subparsers.add_parser(
        "backtest",
        help="Run a backtest with specified strategy",
    )

    backtest_parser.add_argument(
        "--strategy",
        required=True,
        choices=list(STRATEGY_REGISTRY.keys()),
        help="Strategy to backtest",
    )

    symbol_group = backtest_parser.add_mutually_exclusive_group(required=True)
    symbol_group.add_argument(
        "--symbol",
        help="Stock symbol to trade (e.g., AAPL, SPY). Comma-separated for multiple.",
    )
    symbol_group.add_argument(
        "--symbols",
        help="Comma-separated list of symbols for portfolio backtest",
    )
    symbol_group.add_argument(
        "--symbols-file",
        help="Path to newline-delimited symbols file for portfolio backtest",
    )

    backtest_parser.add_argument(
        "--start-date",
        required=True,
        help="Start date for backtest (YYYY-MM-DD)",
    )

    backtest_parser.add_argument(
        "--end-date",
        required=True,
        help="End date for backtest (YYYY-MM-DD)",
    )

    backtest_parser.add_argument(
        "--train-years",
        type=int,
        default=2,
        help="Years of training data per fold (default: 2)",
    )

    backtest_parser.add_argument(
        "--val-months",
        type=int,
        default=3,
        help="Months of validation data per fold (default: 3)",
    )

    backtest_parser.add_argument(
        "--test-months",
        type=int,
        default=3,
        help="Months of test data per fold (default: 3)",
    )

    backtest_parser.add_argument(
        "--step-months",
        type=int,
        default=3,
        help="Months to slide forward each fold (default: 3)",
    )

    backtest_parser.add_argument(
        "--embargo-days",
        type=int,
        default=5,
        help="Embargo days between validation and test (default: 5)",
    )

    backtest_parser.add_argument(
        "--cost-bps",
        type=float,
        default=30.0,
        help="Transaction cost in basis points round-trip (default: 30)",
    )

    backtest_parser.add_argument(
        "--data-dir",
        default=None,
        help="Optional directory for local CSV data (default: ./data)",
    )

    backtest_parser.add_argument(
        "--output",
        default="./backtest_results/",
        help="Output directory for results (default: ./backtest_results/)",
    )

    args = parser.parse_args()

    if args.command == "list-strategies":
        list_strategies()
    elif args.command == "backtest":
        backtest(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
