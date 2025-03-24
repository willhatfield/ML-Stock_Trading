import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt
import os
import logging
import random
import sys
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import the StockPredictor from beta2.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from beta2 import StockPredictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create backtest directories
BACKTEST_DIR = "backtests"
BACKTEST_LOGS_DIR = os.path.join(BACKTEST_DIR, "logs")
BACKTEST_GRAPHS_DIR = os.path.join(BACKTEST_DIR, "graphs")

os.makedirs(BACKTEST_DIR, exist_ok=True)
os.makedirs(BACKTEST_LOGS_DIR, exist_ok=True)
os.makedirs(BACKTEST_GRAPHS_DIR, exist_ok=True)

def get_random_date_range(start_year=2018, end_year=2025, period_days=60):
    """Generate a random date range for backtesting"""
    start_date = dt.datetime(start_year, 1, 1)
    end_date = dt.datetime(2025, 3, 23)  # Fixed current date: March 23, 2025
    
    max_start = end_date - dt.timedelta(days=period_days + 120)  # Add buffer for training
    
    random_start = start_date + dt.timedelta(days=random.randint(0, (max_start - start_date).days))
    random_end = random_start + dt.timedelta(days=period_days)
    
    # Training period (1 year before random start)
    train_start = random_start - dt.timedelta(days=365)
    train_end = random_start
    
    return {
        'train_start': train_start,
        'train_end': train_end,
        'test_start': random_start,
        'test_end': random_end
    }

def plot_regression_error(predicted_prices, actual_prices, symbol, model_type, backtest_id):
    """Create a regression plot showing prediction error for next-day predictions"""
    # Create a figure with two subplots - one for next-day predictions only, one for all days
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # First subplot: Next-day prediction only
    if len(predicted_prices) > 0 and len(actual_prices) > 0:
        next_day_predicted = predicted_prices[0]
        next_day_actual = actual_prices[0]
        
        ax1.scatter([next_day_actual], [next_day_predicted], s=120, color='red', edgecolor='black', zorder=5)
        
        # Add perfect prediction line
        value_range = [min(next_day_actual, next_day_predicted) * 0.95, max(next_day_actual, next_day_predicted) * 1.05]
        ax1.plot(value_range, value_range, 'r--', label='Perfect Prediction')
        
        # Calculate error
        error = next_day_predicted - next_day_actual
        error_pct = (error / next_day_actual) * 100
        
        ax1.set_title(f"{symbol} Next-Day Prediction ({model_type} model)")
        ax1.set_xlabel("Actual Price")
        ax1.set_ylabel("Predicted Price")
        ax1.grid(True, alpha=0.3)
        
        # Add error annotation
        ax1.annotate(f"Actual: ${next_day_actual:.2f}\nPredicted: ${next_day_predicted:.2f}\nError: ${abs(error):.2f} ({abs(error_pct):.2f}%)\nDirection: {'Correct' if (error > 0 and next_day_actual > actual_prices[1] if len(actual_prices) > 1 else True) or (error < 0 and next_day_actual < actual_prices[1] if len(actual_prices) > 1 else True) else 'Incorrect'}", 
                   xy=(0.05, 0.05), xycoords='axes fraction',
                   bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        # Add up/down arrow based on prediction direction
        if len(actual_prices) > 1:
            actual_direction = next_day_actual > actual_prices[1]
            pred_direction = next_day_predicted > actual_prices[1]
            
            direction_correct = actual_direction == pred_direction
            arrow_color = 'green' if direction_correct else 'red'
            
            if actual_direction:
                ax1.annotate('', xy=(next_day_actual, next_day_predicted*1.1), 
                          xytext=(next_day_actual, next_day_predicted*0.95), 
                          arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))
            else:
                ax1.annotate('', xy=(next_day_actual, next_day_predicted*0.95), 
                          xytext=(next_day_actual, next_day_predicted*1.1), 
                          arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))
    
    # Second subplot: All days regression
    ax2.scatter(actual_prices, predicted_prices, alpha=0.6)
    
    # Add perfect prediction line
    min_val = min(min(actual_prices), min(predicted_prices))
    max_val = max(max(actual_prices), max(predicted_prices))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    # Calculate errors
    errors = predicted_prices - actual_prices
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(np.square(errors)))
    
    # Add error statistics
    ax2.set_title(f"{symbol} All Predictions Error ({model_type} model)")
    ax2.set_xlabel("Actual Price")
    ax2.set_ylabel("Predicted Price")
    ax2.annotate(f"MAE: ${mae:.2f}\nRMSE: ${rmse:.2f}", 
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
    
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    regression_file = os.path.join(BACKTEST_GRAPHS_DIR, f"{backtest_id}_regression.png")
    plt.savefig(regression_file)
    plt.close()
    
    logger.info(f"Regression plot saved to {regression_file}")
    return regression_file

def backtest_stock(symbol, model_type, prediction_days=60, feature_days=30, random_period=True, date_range=None):
    """Backtest a stock over a specific period or random period"""
    try:
        # Create a unique ID for this backtest
        backtest_id = f"{symbol}_{model_type}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting backtest {backtest_id}")
        
        # Set date range
        if random_period:
            date_range = get_random_date_range(period_days=prediction_days+60)  # Add 60 days to ensure enough test data
        elif not date_range:
            # Default date range (last year from March 23, 2025)
            today = dt.datetime(2025, 3, 23).replace(tzinfo=None)
            date_range = {
                'train_start': today - dt.timedelta(days=365*2),
                'train_end': today - dt.timedelta(days=365),
                'test_start': today - dt.timedelta(days=365),
                'test_end': today
            }
            
        # Log the selected date range
        date_info = {
            'train_start': date_range['train_start'].strftime('%Y-%m-%d'),
            'train_end': date_range['train_end'].strftime('%Y-%m-%d'),
            'test_start': date_range['test_start'].strftime('%Y-%m-%d'),
            'test_end': date_range['test_end'].strftime('%Y-%m-%d')
        }
        logger.info(f"Date range: {date_info}")
        
        # Initialize the predictor
        predictor = StockPredictor(prediction_days, feature_days, model_type, company=symbol)
        
        # Fetch and prepare training data
        logger.info(f"Fetching training data for {symbol}...")
        train_data = predictor.fetch_data(date_range['train_start'], date_range['train_end'])
        
        if train_data.empty or len(train_data) < prediction_days * 2:
            raise ValueError(f"Not enough training data for {symbol}. Got {len(train_data)} rows, need at least {prediction_days * 2}.")
            
        logger.info(f"Training data shape: {train_data.shape}")
        logger.info(f"Training data range: {train_data.index.min()} to {train_data.index.max()}")
        
        # Prepare data for training
        x_price_train, x_features_train, y_train = predictor.prepare_data(train_data)
        
        # Build and train model
        logger.info(f"Building and training {model_type} model...")
        if model_type == 'hybrid' and x_features_train is not None and x_features_train.size > 0:
            feature_input_shape = (x_features_train.shape[1], x_features_train.shape[2]) if x_features_train.ndim == 3 else None
            predictor.model = predictor.build_model((x_price_train.shape[1], 1), feature_input_shape)
            history = predictor.train_model(x_price_train, x_features_train, y_train)
        else:
            # Fallback to simpler model if no features or not hybrid model
            logger.info(f"Using price-only model architecture.")
            predictor.model = predictor.build_model((x_price_train.shape[1], 1), None)
            history = predictor.train_model(x_price_train, None, y_train)
        
        # Fetch test data
        logger.info(f"Fetching test data for {symbol}...")
        test_data = predictor.fetch_data(date_range['test_start'], date_range['test_end'])
        
        if test_data.empty:
            raise ValueError(f"No test data available for {symbol}")
            
        logger.info(f"Test data shape: {test_data.shape}")
        logger.info(f"Test data range: {test_data.index.min()} to {test_data.index.max()}")
        
        # Add technical indicators for test data
        processed_test_data = predictor.add_technical_indicators(test_data)
        
        # Extract features
        features_list = [
            'Open', 'High', 'Low', 'Volume', 
            'Returns', 'MA5', 'MA20', 'RSI', 'MACD', 
            'BBWidth', '%K', '%D', 'ATR', 'OBV',
            'SP500_Close', 'RelPerf'
        ]
        
        # Ensure all selected features exist in the dataframe
        available_features = [col for col in features_list if col in processed_test_data.columns]
        
        if not available_features:
            logger.warning("No technical features available. Using only price data.")
            test_feature_data = None
            model_type = 'lstm'  # Fallback to simple model if no features available
        else:
            logger.info(f"Using {len(available_features)} features: {', '.join(available_features)}")
            test_feature_data = processed_test_data[available_features].values
        
        # Make sure test_feature_data doesn't have any NaN or infinite values
        if test_feature_data is not None:
            test_feature_data = np.nan_to_num(test_feature_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Combine data for predictions (we need extra days for the prediction window)
        try:
            total_price_data = pd.concat((train_data['Close'], test_data['Close']), axis=0)
            # Remove duplicates if any
            total_price_data = total_price_data[~total_price_data.index.duplicated(keep='first')]
            price_inputs = total_price_data[len(total_price_data) - len(test_data) - predictor.prediction_days:].values
        except Exception as e:
            logger.error(f"Error preparing prediction inputs: {e}")
            # Fallback approach if concatenation fails
            price_inputs = test_data['Close'].values
            if len(price_inputs) < predictor.prediction_days:
                logger.warning(f"Not enough data points for prediction. Need at least {predictor.prediction_days}")
                # Pad with the earliest value if needed
                padding = np.full(predictor.prediction_days - len(price_inputs), price_inputs[0])
                price_inputs = np.concatenate([padding, price_inputs])
        
        # Make predictions on test data
        logger.info("Making predictions on test data...")
        if model_type == 'hybrid' and test_feature_data is not None:
            predicted_prices = predictor.make_predictions(price_inputs, test_feature_data)
        else:
            predicted_prices = predictor.make_predictions(price_inputs, None)
            
        if len(predicted_prices) == 0:
            raise ValueError("Failed to generate predictions.")
        
        # Evaluate model performance
        # Get actual prices, adjusting for the prediction window
        actual_prices = test_data['Close'].values[prediction_days:]
        
        # Fix for insufficient test data
        if len(actual_prices) == 0:
            logger.warning(f"Test period too short. Need at least {prediction_days} days of data plus additional days for evaluation.")
            # Create a shorter window for evaluation if possible
            if len(test_data) > prediction_days + 5:  # At least 5 days for evaluation
                actual_prices = test_data['Close'].values[prediction_days:] 
                logger.info(f"Using {len(actual_prices)} days for evaluation")
            else:
                logger.warning("Not enough data for meaningful evaluation")
                metrics = {
                    'rmse': 0,
                    'mae': 0,
                    'directional_accuracy': 0,
                    'insufficient_data': True
                }
                
                # Plot and save the results even without metrics
                plot_backtest_results(
                    test_data=test_data,
                    predicted_prices=predicted_prices,
                    symbol=symbol,
                    backtest_id=backtest_id,
                    prediction_days=prediction_days,
                    metrics=metrics,
                    model_type=model_type
                )
                
                # Save backtest results to log file
                backtest_results = {
                    'backtest_id': backtest_id,
                    'symbol': symbol,
                    'model_type': model_type,
                    'date_range': date_info,
                    'metrics': metrics,
                    'prediction_days': prediction_days,
                    'feature_days': feature_days,
                    'error': 'Insufficient data for evaluation'
                }
                
                log_file = os.path.join(BACKTEST_LOGS_DIR, f"{backtest_id}.json")
                with open(log_file, 'w') as f:
                    json.dump(backtest_results, f, indent=4)
                    
                logger.info(f"Backtest complete. Results saved to {log_file}")
                
                return backtest_results
        
        # If there's a mismatch in lengths, truncate the longer one
        if len(predicted_prices) > len(actual_prices):
            predicted_prices = predicted_prices[:len(actual_prices)]
        elif len(predicted_prices) < len(actual_prices):
            actual_prices = actual_prices[:len(predicted_prices)]
            
        if len(predicted_prices) == 0 or len(actual_prices) == 0:
            logger.warning("No data available for evaluation after adjustment.")
            metrics = None
        else:
            # Calculate metrics
            mse = mean_squared_error(actual_prices, predicted_prices)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual_prices, predicted_prices)
            
            # Calculate directional accuracy
            if len(actual_prices) > 1:
                actual_direction = np.diff(actual_prices) > 0
                pred_direction = np.diff(predicted_prices) > 0
                
                if len(actual_direction) > 0:
                    directional_accuracy = np.mean(actual_direction == pred_direction)
                else:
                    directional_accuracy = 0
            else:
                directional_accuracy = 0
            
            # Calculate specific next-day prediction accuracy
            next_day_actual = actual_prices[0] if len(actual_prices) > 0 else None
            next_day_predicted = predicted_prices[0] if len(predicted_prices) > 0 else None
            
            if next_day_actual is not None and next_day_predicted is not None:
                next_day_error = abs(next_day_predicted - next_day_actual)
                next_day_error_pct = (next_day_error / next_day_actual) * 100
            else:
                next_day_error = None
                next_day_error_pct = None
                
            metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'directional_accuracy': float(directional_accuracy),
                'next_day_error': float(next_day_error) if next_day_error is not None else None,
                'next_day_error_pct': float(next_day_error_pct) if next_day_error_pct is not None else None
            }
            logger.info(f"Metrics: {metrics}")
            
            # Create regression plot for next-day predictions
            plot_regression_error(predicted_prices, actual_prices, symbol, model_type, backtest_id)
        
        # Plot and save the results
        plot_backtest_results(
            test_data=test_data,
            predicted_prices=predicted_prices,
            symbol=symbol,
            backtest_id=backtest_id,
            prediction_days=prediction_days,
            metrics=metrics,
            model_type=model_type
        )
        
        # Save backtest results to log file
        backtest_results = {
            'backtest_id': backtest_id,
            'symbol': symbol,
            'model_type': model_type,
            'date_range': date_info,
            'metrics': metrics,
            'prediction_days': prediction_days,
            'feature_days': feature_days,
        }
        
        log_file = os.path.join(BACKTEST_LOGS_DIR, f"{backtest_id}.json")
        with open(log_file, 'w') as f:
            json.dump(backtest_results, f, indent=4)
            
        logger.info(f"Backtest complete. Results saved to {log_file}")
        
        return backtest_results
        
    except Exception as e:
        logger.error(f"Error in backtest: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_backtest_results(test_data, predicted_prices, symbol, backtest_id, prediction_days, metrics, model_type):
    """Plot and save the backtest results"""
    plt.figure(figsize=(14, 7))
    
    # Plot actual test data
    if len(test_data) <= prediction_days:
        # Not enough test data to show meaningful comparison
        plt.title(f"{symbol} Backtest ({model_type} model) - Insufficient Test Data")
        plt.xlabel("Date")
        plt.ylabel(f"{symbol} Share Price")
        plt.text(0.5, 0.5, "Insufficient test data for meaningful comparison", 
                 ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
    else:
        # Get the date range where we have both actual and predicted values
        actual_dates = test_data.index[prediction_days:prediction_days+len(predicted_prices)]
        
        if len(actual_dates) > 0:
            actual_prices = test_data['Close'].values[prediction_days:prediction_days+len(predicted_prices)]
            
            plt.plot(actual_dates, actual_prices, label="Actual Price", color="black")
            plt.plot(actual_dates, predicted_prices, label="Predicted Price", color="blue")
            
            # Add metrics annotation if provided
            if metrics and not metrics.get('insufficient_data', False):
                try:
                    info_text = (
                        f"RMSE: {metrics.get('rmse', 0):.2f}\n"
                        f"MAE: {metrics.get('mae', 0):.2f}\n"
                        f"Dir Acc: {metrics.get('directional_accuracy', 0):.2f}"
                    )
                    plt.annotate(info_text, xy=(0.02, 0.95), xycoords='axes fraction',
                                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
                except (TypeError, KeyError) as e:
                    logger.warning(f"Could not add metrics to plot: {e}")
            
            # Format plot
            plt.title(f"{symbol} Backtest ({model_type} model)")
            plt.xlabel("Date")
            plt.ylabel(f"{symbol} Share Price")
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
        else:
            # No overlapping date range for comparison
            plt.title(f"{symbol} Backtest ({model_type} model) - No Matching Dates")
            plt.xlabel("Date")
            plt.ylabel(f"{symbol} Share Price")
            plt.text(0.5, 0.5, "No matching date range for comparison", 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
    
    plt.tight_layout()
    
    # Save the plot
    graph_file = os.path.join(BACKTEST_GRAPHS_DIR, f"{backtest_id}.png")
    plt.savefig(graph_file)
    plt.close()
    
    logger.info(f"Backtest graph saved to {graph_file}")

def run_multiple_backtests(stocks=None, model_types=None, num_tests=5):
    """Run multiple backtests on different stocks and models"""
    if stocks is None:
        stocks = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'TSLA', 'META', 'JPM', 'V', 'WMT']
    
    if model_types is None:
        model_types = ['lstm', 'bilstm', 'cnn_lstm', 'hybrid', 'gru']
    
    # Record start time
    start_time = dt.datetime.now()
    
    # Track results
    results = []
    
    for _ in range(num_tests):
        # Randomly select a stock and model type
        symbol = random.choice(stocks)
        model_type = random.choice(model_types)
        
        # Run the backtest
        logger.info(f"Running backtest for {symbol} with {model_type} model")
        result = backtest_stock(symbol, model_type, random_period=True)
        
        if result:
            results.append(result)
    
    # Calculate total time
    end_time = dt.datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Summarize results
    print(f"\n===== Backtest Summary =====")
    print(f"Completed {len(results)} backtests in {duration:.1f} seconds")
    
    if results:
        # Calculate average metrics
        avg_rmse = np.mean([r['metrics'].get('rmse', 0) for r in results if r.get('metrics')])
        avg_dir_acc = np.mean([r['metrics'].get('directional_accuracy', 0) for r in results if r.get('metrics')])
        
        print(f"Average RMSE: {avg_rmse:.2f}")
        print(f"Average Directional Accuracy: {avg_dir_acc:.2f}")
        
        # Best model by RMSE
        best_rmse = min([r for r in results if r.get('metrics')], key=lambda x: x['metrics'].get('rmse', float('inf')))
        print(f"Best RMSE: {best_rmse['metrics']['rmse']:.2f} - {best_rmse['symbol']} with {best_rmse['model_type']} model")
        
        # Best model by directional accuracy
        best_dir = max([r for r in results if r.get('metrics')], key=lambda x: x['metrics'].get('directional_accuracy', 0))
        print(f"Best Dir. Accuracy: {best_dir['metrics']['directional_accuracy']:.2f} - {best_dir['symbol']} with {best_dir['model_type']} model")
    
    print(f"\nBacktest graphs saved to {BACKTEST_GRAPHS_DIR}")
    print(f"Backtest logs saved to {BACKTEST_LOGS_DIR}")
    
    return results

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Run backtests on stock prediction models')
    parser.add_argument('--stock', type=str, help='Stock symbol to backtest')
    parser.add_argument('--model', type=str, help='Model type to use', choices=['lstm', 'bilstm', 'cnn_lstm', 'hybrid', 'gru'])
    parser.add_argument('--count', type=int, default=5, help='Number of backtests to run')
    parser.add_argument('--days', type=int, default=60, help='Number of days to predict')
    
    args = parser.parse_args()
    
    if args.stock and args.model:
        # Run a single backtest
        backtest_stock(args.stock, args.model, prediction_days=args.days)
    else:
        # Run multiple backtests
        stocks = [args.stock] if args.stock else None
        models = [args.model] if args.model else None
        run_multiple_backtests(stocks=stocks, model_types=models, num_tests=args.count) 