#\!/usr/bin/env python3
"""
Entry point for the ML-Stock_Trading framework CLI.

Run this script to execute framework commands:
    python run.py --help
    python run.py list-strategies
    python run.py backtest --strategy momentum --symbol AAPL [options]
"""

from src.cli import main

if __name__ == "__main__":
    main()
