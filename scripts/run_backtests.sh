#!/bin/bash

# Create required directories
mkdir -p backtests/logs
mkdir -p backtests/graphs
mkdir -p backtests/summary

echo "Starting backtesting process..."

# Set of stocks to test (major tech, finance, retail, healthcare)
STOCKS=("AAPL" "MSFT" "GOOGL" "AMZN" "META" "NVDA" "TSLA" "JPM" "BAC" "V" "WMT" "TGT" "JNJ" "PFE" "UNH" "VOO")

# Model types
MODELS=("lstm" "gru" "bilstm" "cnn_lstm" "hybrid")

# Run a test for each model type on all stocks
echo "Running comprehensive next-day prediction tests..."
for model in "${MODELS[@]}"
do
    # Test the first 5 stocks with each model type
    for stock in "${STOCKS[@]:0:5}"
    do
        echo "Backtesting $stock with $model model for next-day prediction..."
        python models/backtest.py --stock "$stock" --model "$model" --days 60
    done
done

# Run some targeted tests on NVDA with different models
echo "Running focused tests on NVDA..."
for model in "${MODELS[@]}"
do
    echo "Backtesting NVDA with $model model (specific timeframe)..."
    python models/backtest.py --stock "NVDA" --model "$model" --days 60
done

# Run 20 random tests to increase sample size
echo "Running random stock/model combinations for statistical significance..."
python models/backtest.py --count 20

# Generate summary 
echo "Generating backtest summary with next-day prediction analysis..."
python models/backtest_summary.py

echo "Backtest process completed."
echo "Results are available in the backtests directory:"
echo "  - Individual backtest graphs: backtests/graphs/"
echo "  - Regression plots showing prediction errors: backtests/graphs/*_regression.png"
echo "  - Backtest logs: backtests/logs/"
echo "  - Summary results: backtests/summary/"

# Display top performing models for next-day prediction
echo "Top performing models for next-day prediction (from summary):"
if [ -f "backtests/summary/backtest_summary.txt" ]; then
    grep -A 5 "NEXT-DAY PREDICTION CHAMPIONS" backtests/summary/backtest_summary.txt
    echo ""
    grep -A 7 "MODEL TYPE COMPARISON" backtests/summary/backtest_summary.txt
fi 