import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
import traceback
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.dates as mdates

# Import the StockPredictor from beta2.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
from beta2 import StockPredictor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def predict_single_stock(symbol, model_type='lstm', prediction_days=30, feature_days=30):
    """
    Run a complete prediction cycle for a single stock
    
    Args:
        symbol: Stock symbol to predict
        model_type: Model type to use ('lstm', 'gru', 'bilstm', 'cnn_lstm', 'hybrid')
        prediction_days: Number of days to use in the prediction window
        feature_days: Number of days to use for technical features
        
    Returns:
        dict: Prediction results
    """
    try:
        logger.info(f"Starting prediction for {symbol} using {model_type} model")
        
        # Create output directory
        os.makedirs("predictions", exist_ok=True)
        
        # Initialize the predictor
        predictor = StockPredictor(
            prediction_days=prediction_days, 
            feature_days=feature_days, 
            model_type=model_type, 
            company=symbol
        )
        
        # Define date ranges
        end_date = datetime.now()
        train_start_date = end_date - timedelta(days=730)  # 2 years for training
        train_end_date = end_date - timedelta(days=60)     # Use recent 60 days for testing
        
        # Fetch and prepare training data
        logger.info(f"Fetching training data for {symbol}...")
        price_data, feature_data = predictor.fetch_data(train_start_date, train_end_date, symbol)
        
        if len(price_data) < prediction_days * 2:
            raise ValueError(f"Not enough data for {symbol}. Got {len(price_data)} days, need at least {prediction_days * 2}.")
            
        logger.info(f"Preparing data for model training...")
        x_price_train, x_features_train, y_train = predictor.prepare_data(price_data, feature_data)
        
        # Build and train model
        logger.info(f"Building and training {model_type} model...")
        feature_input_shape = None
        
        if model_type == 'hybrid' and x_features_train is not None and x_features_train.size > 0:
            feature_input_shape = (x_features_train.shape[1], x_features_train.shape[2]) if x_features_train.ndim == 3 else None
        
        predictor.model = predictor.build_model(feature_input_shape)
        
        if model_type == 'hybrid' and x_features_train is not None and x_features_train.size > 0:
            history = predictor.train_model(x_price_train, x_features_train, y_train)
        else:
            history = predictor.train_model(x_price_train, None, y_train)
        
        if history is None:
            raise ValueError("Model training failed")
            
        # Make next-day prediction
        try:
            logger.info("Making next-day prediction...")
            # Get the most recent window of data
            recent_price_data, _ = predictor.fetch_data(
                start_date=datetime.now() - timedelta(days=predictor.feature_days * 2),
                end_date=datetime.now(),
                symbol=symbol
            )
            
            # Transform the recent data
            # Check if recent_price_data is a numpy array or DataFrame
            if isinstance(recent_price_data, np.ndarray):
                most_recent_data = predictor.scaler_price.transform(recent_price_data)
            else:
                most_recent_data = predictor.scaler_price.transform(recent_price_data.values)
            
            most_recent_window = most_recent_data[-predictor.feature_days:].reshape(1, predictor.feature_days, 1)
            
            # Make prediction
            next_day_prediction = predictor.model.predict(most_recent_window)
            
            # Convert back to original scale
            if hasattr(next_day_prediction, 'reshape'):
                next_day_prediction = next_day_prediction.reshape(-1, 1)
            next_day_prediction = predictor.scaler_price.inverse_transform(next_day_prediction)
            
            # Get the current price (last known price)
            if isinstance(recent_price_data, np.ndarray):
                current_price = recent_price_data[-1]
                recent_trend = recent_price_data[-1] - recent_price_data[-2]
            else:
                current_price = recent_price_data.values[-1]
                recent_trend = recent_price_data.values[-1] - recent_price_data.values[-2]
            
            # Log the prediction - extract scalar value from arrays if needed
            logger.info(f"Current price: ${float(current_price):.2f}")
            logger.info(f"Recent trend: {'upward' if float(recent_trend) > 0 else 'downward'} movement of {float(recent_trend):.4f}")
            logger.info(f"Next-day prediction: ${float(next_day_prediction[0][0]):.2f}")
        except Exception as e:
            logger.error(f"Error in next-day prediction: {str(e)}")
            traceback.print_exc()
        
        # Make future predictions
        try:
            # Get most recent window if not already defined
            if 'most_recent_window' not in locals() or most_recent_window is None:
                recent_price_data, _ = predictor.fetch_data(
                    start_date=datetime.now() - timedelta(days=predictor.feature_days * 2),
                    end_date=datetime.now(),
                    symbol=symbol
                )
                
                # Check if recent_price_data is a numpy array or DataFrame
                if isinstance(recent_price_data, np.ndarray):
                    most_recent_data = predictor.scaler_price.transform(recent_price_data)
                else:
                    most_recent_data = predictor.scaler_price.transform(recent_price_data.values)
                
                most_recent_window = most_recent_data[-predictor.feature_days:].reshape(1, predictor.feature_days, 1)
            
            # Get the most recent shape correctly
            if tf.is_tensor(most_recent_window):
                window_numpy = most_recent_window.numpy()
            else:
                window_numpy = most_recent_window.copy()
            
            # Calculate historical volatility and mean return
            if 'scaled_prices' in locals() and len(predictor.scaled_prices) > 30:
                historical_volatility = np.std(np.diff(predictor.scaled_prices[-30:]))
                mean_return = np.mean(np.diff(predictor.scaled_prices[-30:]))
            else:
                historical_volatility = 0.01
                mean_return = 0.0005
            
            # Initialize array for future predictions
            # Use the most_recent_window from the next-day prediction
            # or create it if not available
            prediction_input = None
            
            if 'most_recent_window' in locals() and most_recent_window is not None:
                # Use numpy to handle the predictions
                if tf.is_tensor(most_recent_window):
                    prediction_input = most_recent_window.numpy()
                else:
                    prediction_input = np.array(most_recent_window)
            else:
                # If most_recent_window is not available, create it
                if isinstance(recent_price_data, np.ndarray):
                    most_recent_data = predictor.scaler_price.transform(recent_price_data)
                else:
                    most_recent_data = predictor.scaler_price.transform(recent_price_data.values)
                most_recent_window = most_recent_data[-predictor.feature_days:].reshape(1, predictor.feature_days, 1)
                prediction_input = np.array(most_recent_window)
            
            # Calculate historical volatility for constraints
            if hasattr(predictor, 'scaled_prices') and len(predictor.scaled_prices) >= 20:  # Need enough data to calculate volatility
                historical_volatility = np.std(predictor.scaled_prices[-20:]) 
                mean_return = np.mean(np.diff(predictor.scaled_prices[-20:])) 
                
                # Set constraints based on historical volatility
                max_daily_change = historical_volatility * 2.5  # Allow for 2.5x standard deviation moves
                logger.info(f"Historical volatility: {historical_volatility:.4f}, Mean return: {mean_return:.4f}")
                logger.info(f"Max daily change constraint: {max_daily_change:.4f}")
            else:
                # Default constraints if not enough data
                historical_volatility = 0.02
                mean_return = 0.001
                max_daily_change = 0.05
                logger.info("Using default constraints due to insufficient historical data")
            
            # Generate future predictions
            future_predictions = []
            future_window = prediction_input.copy()
            
            # Generate prediction dates (excluding weekends)
            prediction_dates = []
            current_date = end_date - timedelta(days=60)
            
            # First prediction (next day) already calculated
            future_predictions.append(next_day_prediction[0][0])
            prediction_dates.append(current_date)
            
            # Generate remaining predictions
            for i in range(1, prediction_days):
                # Move to next business day
                current_date = current_date + timedelta(days=1)
                while current_date.weekday() >= 5:  # Skip weekends
                    current_date = current_date + timedelta(days=1)
                
                # Update the prediction window by removing the oldest value and adding the latest prediction
                if len(future_predictions) > 0:
                    # Get the last prediction in scaled form for the window
                    # We need to keep predictions in scaled form for the model input
                    last_prediction_scaled = predictor.scaler_price.transform(np.array([[future_predictions[-1]]]))
                    
                    # Ensure consistent dimensions for reshaping
                    future_window_reshaped = future_window.reshape(predictor.feature_days, 1)
                    # Remove the first value and append the new prediction
                    future_window_reshaped = np.concatenate((future_window_reshaped[1:], last_prediction_scaled), axis=0)
                    # Reshape back to the expected input shape for the model
                    future_window = future_window_reshaped.reshape(1, predictor.feature_days, 1)
                
                # Make prediction (output is in scaled form)
                raw_pred = predictor.model.predict(future_window, verbose=0)[0][0]
                
                # Convert to unscaled value for constraints and calculations
                unscaled_raw_pred = float(predictor.scaler_price.inverse_transform([[raw_pred]])[0][0])
                
                # Apply momentum and mean reversion factors
                momentum_factor = 0.7  # Weight for momentum
                reversion_factor = 0.3  # Weight for mean reversion
                
                prev_return = 0
                if len(future_predictions) > 0:
                    prev_value = future_predictions[-1]
                    # Get the second last prediction from the window (already scaled)
                    prev_window_value = float(predictor.scaler_price.inverse_transform([[future_window.reshape(predictor.feature_days, 1)[-2][0]]])[0][0])
                    if prev_window_value > 0:  # Avoid division by zero
                        prev_return = (prev_value - prev_window_value) / prev_window_value
                
                # Blend prediction with momentum and mean reversion
                momentum_component = future_predictions[-1] * (1 + prev_return)
                reversion_component = future_predictions[-1] * (1 + mean_return)
                blended_pred = unscaled_raw_pred * 0.6 + momentum_component * momentum_factor * 0.2 + reversion_component * reversion_factor * 0.2
                
                # Apply constraints to ensure realistic predictions
                prev_price = future_predictions[-1]
                max_change = prev_price * max_daily_change
                
                # Ensure prediction doesn't change too much from previous day
                if blended_pred > prev_price + max_change:
                    blended_pred = prev_price + max_change
                    logger.debug(f"Prediction constrained (upper bound): {blended_pred:.4f}")
                elif blended_pred < prev_price - max_change:
                    blended_pred = prev_price - max_change
                    logger.debug(f"Prediction constrained (lower bound): {blended_pred:.4f}")
                
                future_predictions.append(blended_pred)
                prediction_dates.append(current_date)
            
            # The future_predictions array already contains unscaled values, so DO NOT inverse transform again
            # Just convert to numpy array for plotting
            future_predictions = np.array(future_predictions)
            
            # Ensure prediction_dates and future_predictions are the same length
            min_length = min(len(prediction_dates), len(future_predictions))
            prediction_dates = prediction_dates[:min_length]
            future_predictions = future_predictions[:min_length]
            
            logger.info(f"Generated {len(future_predictions)} future predictions")
            
            # Prepare for plotting
            if isinstance(recent_price_data, np.ndarray):
                plot_prices = np.array(recent_price_data).flatten()
            else:
                plot_prices = np.array(recent_price_data.values).flatten()
            plot_dates = np.array(range(len(plot_prices)))  # Use integer indices for x-axis
            
            future_indices = np.array(range(len(plot_prices), len(plot_prices) + len(future_predictions)))
            
            # Create the plot
            plt.figure(figsize=(12, 6))
            
            # Plot historical prices
            plt.plot(plot_dates, plot_prices, label='Historical Prices', color='blue')
            
            # Add vertical line at the last known price
            plt.axvline(x=len(plot_prices)-1, color='gray', linestyle='--')
            
            # Add text annotation for the current price
            plt.annotate(f'Current: ${plot_prices[-1]:.2f}', 
                        xy=(len(plot_prices)-1, plot_prices[-1]),
                        xytext=(10, 20),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->'))
            
            # Plot next day prediction
            plt.scatter(len(plot_prices), future_predictions[0], color='green', s=100, zorder=5)
            plt.annotate(f'Next day: ${future_predictions[0]:.2f}', 
                        xy=(len(plot_prices), future_predictions[0]),
                        xytext=(10, -30),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->'))
            
            # Plot future predictions
            if len(future_predictions) > 1:
                plt.plot(future_indices, future_predictions, label='Future Predictions', color='red', linestyle='--')
            
            # Format x-axis with dates
            try:
                # Create tick positions at regular intervals
                num_ticks = 10  # Adjust as needed
                
                # Calculate positions for historical ticks and prediction ticks
                hist_tick_interval = max(1, len(plot_prices) // (num_ticks // 2))
                pred_tick_interval = max(1, len(future_predictions) // (num_ticks // 2))
                
                # Create positions for historical ticks
                historical_tick_positions = list(range(0, len(plot_prices), hist_tick_interval))
                if len(plot_prices) - 1 not in historical_tick_positions:
                    historical_tick_positions.append(len(plot_prices) - 1)  # Add the last historical point
                
                # Create positions for prediction ticks
                prediction_tick_positions = list(range(len(plot_prices), len(plot_prices) + len(future_predictions), pred_tick_interval))
                if len(plot_prices) + len(future_predictions) - 1 not in prediction_tick_positions and len(future_predictions) > 0:
                    prediction_tick_positions.append(len(plot_prices) + len(future_predictions) - 1)  # Add the last prediction point
                
                # Combine positions
                tick_positions = historical_tick_positions + prediction_tick_positions
                
                # Create tick labels
                tick_labels = []
                
                # Add historical date labels
                historical_dates = []
                
                # Try to extract dates from index if it's a DatetimeIndex
                try:
                    if isinstance(recent_price_data, pd.DataFrame) and hasattr(recent_price_data.index, 'date'):
                        for date_idx in historical_tick_positions:
                            if 0 <= date_idx < len(recent_price_data.index):
                                historical_dates.append(recent_price_data.index[date_idx].date())
                            else:
                                historical_dates.append(None)
                    else:
                        # If index is not DatetimeIndex, create generic labels
                        historical_dates = [None] * len(historical_tick_positions)
                except Exception as e:
                    logger.warning(f"Could not extract dates from index: {e}")
                    historical_dates = [None] * len(historical_tick_positions)
                
                # Format historical date labels
                for i, date in enumerate(historical_dates):
                    if date is not None:
                        tick_labels.append(date.strftime('%Y-%m-%d'))
                    else:
                        # Use a generic label if date is not available
                        tick_labels.append(f"Hist-{i}")
                
                # Add prediction date labels
                for i, pos in enumerate(prediction_tick_positions):
                    pred_idx = pos - len(plot_prices)
                    if 0 <= pred_idx < len(prediction_dates):
                        tick_labels.append(prediction_dates[pred_idx].strftime('%Y-%m-%d'))
                    else:
                        tick_labels.append(f"Pred-{i}")
                
                # Set the ticks
                plt.xticks(tick_positions, tick_labels, rotation=45)
                
            except Exception as e:
                logger.warning(f"Error formatting x-axis dates: {e}")
                # Fall back to numeric x-axis if date formatting fails
                plt.xlabel('Days')
            
            # Add labels and title
            plt.xlabel('Date')
            plt.ylabel('Price ($)')
            plt.title(f'{symbol} Stock Price Prediction ({model_type.upper()} model)')
            plt.legend()
            plt.tight_layout()
            
            # Save the plot
            os.makedirs('predictions', exist_ok=True)
            plt.savefig(f'predictions/{symbol}_{model_type}_prediction.png')
            plt.close()
            
            # Print prediction summary
            logger.info("\n" + "="*50)
            logger.info(f"PREDICTION SUMMARY FOR {symbol}")
            logger.info("="*50)
            logger.info(f"Current price: ${float(plot_prices[-1]):.2f}")
            logger.info(f"Next day prediction: ${float(future_predictions[0]):.2f} ({(float(future_predictions[0])/float(plot_prices[-1]) - 1)*100:.2f}%)")
            
            if len(future_predictions) > 5:
                logger.info(f"5-day prediction: ${float(future_predictions[5]):.2f} ({(float(future_predictions[5])/float(plot_prices[-1]) - 1)*100:.2f}%)")
            
            if len(future_predictions) > 20:
                logger.info(f"20-day prediction: ${float(future_predictions[20]):.2f} ({(float(future_predictions[20])/float(plot_prices[-1]) - 1)*100:.2f}%)")
            
            # Calculate prediction statistics
            pred_min = np.min(future_predictions)
            pred_max = np.max(future_predictions)
            pred_mean = np.mean(future_predictions)
            
            logger.info(f"Prediction range: ${float(pred_min):.2f} to ${float(pred_max):.2f}")
            logger.info(f"Mean prediction: ${float(pred_mean):.2f}")
            logger.info("="*50)
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            traceback.print_exc()
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict stock prices for a single stock')
    parser.add_argument('symbol', type=str, help='Stock symbol to predict')
    parser.add_argument('--model', type=str, default='lstm', 
                        choices=['lstm', 'gru', 'bilstm', 'cnn_lstm', 'hybrid'],
                        help='Model type to use')
    parser.add_argument('--days', type=int, default=30, help='Number of days in prediction window')
    parser.add_argument('--features', type=int, default=30, help='Number of days for technical features')
    
    args = parser.parse_args()
    
    predict_single_stock(args.symbol, args.model, args.days, args.features)
    
    print("\nAvailable model types: lstm, gru, bilstm, cnn_lstm, hybrid") 