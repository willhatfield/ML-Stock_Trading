import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Import the StockPredictor from beta2.py for data fetching and indicators
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
from beta2 import StockPredictor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockDirectionTrader:
    """
    A machine learning system that predicts stock price direction (up/down)
    and simulates trading based on those predictions.
    """
    
    def __init__(self, symbol, lookback_days=30, test_size=0.2, initial_balance=100, transaction_cost=0.0):
        """
        Initialize the trader with the given parameters.
        
        Args:
            symbol: Stock symbol to trade
            lookback_days: Number of days to use for prediction features
            test_size: Proportion of data to use for testing
            initial_balance: Starting balance for backtesting
        """
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.test_size = test_size
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0  # Number of shares held
        self.transaction_cost = transaction_cost  # Proportional transaction cost
        
        # For feature extraction and data preparation
        self.stock_predictor = StockPredictor(
            prediction_days=lookback_days,
            feature_days=lookback_days,
            model_type='lstm',
            company=symbol
        )
        
        # Initialize scalers
        self.feature_scaler = StandardScaler()
        
        # Initialize model
        self.model = None
        
        # Trade log
        self.trade_log = []
        
        # Performance metrics
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_profit': 0.0,
            'final_balance': 0.0,
            'max_drawdown': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }

    def prepare_data(self, start_date, end_date):
        """
        Prepare data for training and testing.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            DataFrame with price data and features
        """
        # Fetch historical data
        logger.info(f"Fetching data for {self.symbol} from {start_date} to {end_date}")
        price_data, feature_data = self.stock_predictor.fetch_data(start_date, end_date, self.symbol)
        
        if len(price_data) == 0:
            raise ValueError(f"No data retrieved for {self.symbol}. Check the symbol and date range.")
        
        # Handle different shapes of price_data
        logger.info(f"Price data shape: {price_data.shape}")
        
        # Ensure price_data is 1-dimensional
        if len(price_data.shape) > 1:
            # If it's a 2D array, assume the first column is the close price
            price_values = price_data[:, 0] if price_data.shape[1] > 0 else price_data.flatten()
        else:
            price_values = price_data
        
        # Create a DataFrame with dates as index
        dates = pd.date_range(start=start_date, periods=len(price_values), freq='B')
        
        # Create a DataFrame with the Close prices
        df = pd.DataFrame({
            'Close': price_values
        }, index=dates)
        
        # Generate simple OHLC data (approximation since we only have Close)
        # This is just for features that might expect this format
        df['Open'] = df['Close'].values
        df['High'] = df['Close'].values
        df['Low'] = df['Close'].values
        df['Volume'] = 0  # Placeholder
        
        # Add technical indicators from feature_data
        if feature_data is not None and feature_data.size > 0:
            logger.info(f"Feature data shape: {feature_data.shape}")
            
            # Get feature names based on beta2.py structure
            feature_names = [
                'Volume', 'Returns', 'MA5', 'MA20', 'RSI', 'MACD', 
                'BBWidth', '%K', '%D', 'ATR', 'OBV', 'SP500_Close', 'RelPerf'
            ]
            
            # Ensure feature_names length matches feature_data columns
            if len(feature_data.shape) > 1:
                num_features = min(len(feature_names), feature_data.shape[1])
                for i in range(num_features):
                    df[feature_names[i]] = feature_data[:, i]
            else:
                # Single feature column
                df['feature_0'] = feature_data
        
        # Calculate daily returns (percentage change)
        df['daily_return'] = df['Close'].pct_change() * 100
        
        # Create target variable: 1 if price goes up tomorrow, 0 if down
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        # Ensure we have enough data points
        if len(df) < self.lookback_days * 2:
            raise ValueError(f"Not enough data points after processing: {len(df)}. Need at least {self.lookback_days * 2}.")
        
        logger.info(f"Prepared data with {len(df)} days and {len(df.columns)} features")
        
        return df

    def extract_features(self, df):
        """
        Extract features for the model.
        
        Args:
            df: DataFrame with price data and indicators
            
        Returns:
            X: Feature matrix
            y: Target labels
        """
        # Select all numeric columns except target and date-related columns
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in feature_columns if col not in ['target', 'daily_return']]
        
        # Create sequences of lookback_days
        X = []
        y = []
        
        for i in range(self.lookback_days, len(df)):
            X.append(df[feature_columns].iloc[i-self.lookback_days:i].values)
            y.append(df['target'].iloc[i])
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y

    def build_model(self, input_shape):
        """
        Build a deep learning model for binary classification.
        
        Args:
            input_shape: Shape of input features
            
        Returns:
            Compiled TensorFlow model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Use binary crossentropy for binary classification
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train_model(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """
        Train the deep learning model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        # Build the model
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
        ]
        
        # Train the model
        if X_val is not None and y_val is not None:
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Use a portion of training data for validation
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
        
        return history

    def predict_direction(self, features):
        """
        Predict whether the stock price will go up or down.
        
        Args:
            features: Input features for prediction
            
        Returns:
            1 for up, 0 for down
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Reshape for a single prediction if needed
        if len(features.shape) == 2:
            features = np.expand_dims(features, axis=0)
        
        # Get prediction probability and use it directly without random noise
        pred_prob = self.model.predict(features, verbose=0)
        confidence = float(pred_prob[0][0])

        # Determine signal based on raw model confidence
        signal = int(confidence > 0.5)

        logger.debug(
            f"Confidence: {confidence:.4f}, Prediction: {signal}"
        )

        return signal, confidence

    def generate_signal(self, prediction, confidence, current_position):
        """
        Generate a trading signal based on prediction.
        
        Args:
            prediction: Binary prediction (1 for up, 0 for down)
            confidence: Prediction probability
            current_position: Current position (0 for no position, >0 for long)
            
        Returns:
            'BUY', 'SELL', or 'HOLD'
        """
        # Very low confidence threshold to encourage trading - for testing purposes
        confidence_threshold = 0.51  # Was 0.55
        
        # Log the prediction and confidence
        logger.debug(f"Prediction: {'UP' if prediction == 1 else 'DOWN'}, Confidence: {confidence:.4f}, "
                    f"Position: {current_position:.4f}, Threshold: {confidence_threshold}")
        
        # More aggressive trading strategy:
        # Buy if predicted up with no position, regardless of confidence (model already has trained confidence)
        # Sell if predicted down with position
        # Hold only in specific cases
        if prediction == 1 and current_position == 0:
            logger.debug(f"Generating BUY signal (confidence: {confidence:.4f})")
            return 'BUY'
        elif prediction == 0 and current_position > 0:
            logger.debug(f"Generating SELL signal (confidence: {confidence:.4f})")
            return 'SELL'
        else:
            # Log why we're holding
            if current_position > 0 and prediction == 1:
                logger.debug("Holding: Already have position and predicting UP")
            elif current_position == 0 and prediction == 0:
                logger.debug("Holding: No position and predicting DOWN")
            else:
                logger.debug("Holding: General case")
            return 'HOLD'

    def execute_trade(self, signal, execution_price, date):
        """
        Execute a trade based on the signal.
        
        Args:
            signal: Trading signal ('BUY', 'SELL', or 'HOLD')
            execution_price: Price at which the trade is executed
            date: Execution date
            
        Returns:
            Updated balance and position
        """
        # Simplified trading logic with proportional transaction costs
        if signal == 'BUY' and self.position == 0:
            # Calculate number of shares to buy accounting for transaction cost
            shares_to_buy = self.balance / (execution_price * (1 + self.transaction_cost))
            cost = shares_to_buy * execution_price
            fee = cost * self.transaction_cost
            previous_balance = self.balance
            self.position = shares_to_buy
            self.balance -= cost + fee

            # Log the trade
            self.trade_log.append({
                'date': date,
                'action': 'BUY',
                'price': execution_price,
                'shares': shares_to_buy,
                'previous_balance': previous_balance,
                'new_balance': self.balance,
                'portfolio_value': self.position * execution_price,
                'transaction_cost': fee
            })

            self.metrics['total_trades'] += 1

        elif signal == 'SELL' and self.position > 0:
            # Calculate proceeds from selling all shares
            proceeds = self.position * execution_price
            fee = proceeds * self.transaction_cost
            previous_balance = self.balance
            self.balance += proceeds - fee

            # Determine if this was a winning trade
            last_buy_price = None
            for trade in reversed(self.trade_log):
                if trade['action'] == 'BUY':
                    last_buy_price = trade['price']
                    break

            if last_buy_price is not None:
                trade_profit = (execution_price - last_buy_price) * self.position
                if trade_profit > 0:
                    self.metrics['winning_trades'] += 1
                else:
                    self.metrics['losing_trades'] += 1

            # Log the trade
            self.trade_log.append({
                'date': date,
                'action': 'SELL',
                'price': execution_price,
                'shares': self.position,
                'previous_balance': previous_balance,
                'new_balance': self.balance,
                'portfolio_value': 0,
                'transaction_cost': fee
            })

            self.position = 0
            self.metrics['total_trades'] += 1
        
    def calculate_portfolio_value(self, current_price):
        """
        Calculate current portfolio value (cash + shares).
        
        Args:
            current_price: Current stock price
            
        Returns:
            Total portfolio value
        """
        return self.balance + (self.position * current_price)

    def backtest(self, df, start_idx=None, end_idx=None):
        """
        Perform backtesting on the given data.
        
        Args:
            df: DataFrame with price data and features
            start_idx: Starting index for backtesting
            end_idx: Ending index for backtesting
        """
        if start_idx is None:
            start_idx = self.lookback_days
        if end_idx is None:
            end_idx = len(df)
        
        # Reset backtesting state
        self.balance = self.initial_balance
        self.position = 0
        self.trade_log = []
        
        # Track portfolio value over time
        portfolio_values = []
        dates = []
        predictions = []
        actual_movements = []
        
        # Keep track of the highest portfolio value seen
        highest_value = self.initial_balance
        max_drawdown = 0
        
        logger.info("Starting backtesting...")
        
        # Iterate through each day
        for i in range(start_idx, end_idx - 1):
            current_date = df.index[i]
            current_price = df['Close'].iloc[i]
            next_date = df.index[i + 1]
            next_price = df['Close'].iloc[i + 1]
            
            # Extract features for the lookback period
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in feature_columns if col not in ['target', 'daily_return']]
            features = df[feature_columns].iloc[i-self.lookback_days:i].values
            
            # Reshape for model input
            features = np.expand_dims(features, axis=0)
            
            # Get prediction for tomorrow's direction
            prediction, confidence = self.predict_direction(features)
            
            # Enhanced logging
            logger.debug(f"Date: {current_date}, Price: ${current_price:.2f}")
            logger.debug(f"  Prediction: {'UP' if prediction == 1 else 'DOWN'}, Confidence: {confidence:.4f}")
            
            # Record the actual price movement for performance evaluation
            actual_movement = 1 if next_price > current_price else 0
            actual_movements.append(actual_movement)
            # Store prediction for later evaluation
            predictions.append(prediction)
            logger.debug(f"  Actual movement: {'UP' if actual_movement == 1 else 'DOWN'}")
            
            # Generate trading signal
            signal = self.generate_signal(prediction, confidence, self.position)
            
            # Execute trade on next bar
            self.execute_trade(signal, next_price, next_date)

            # Calculate and record portfolio value using execution price
            portfolio_value = self.calculate_portfolio_value(next_price)
            portfolio_values.append(portfolio_value)
            dates.append(next_date)
            
            # Update max drawdown
            highest_value = max(highest_value, portfolio_value)
            drawdown = (highest_value - portfolio_value) / highest_value if highest_value > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
            
            # Logging for all significant signals
            if signal != 'HOLD':
                logger.info(
                    f"Date: {next_date}, Price: ${next_price:.2f}, "
                    f"Prediction: {'UP' if prediction == 1 else 'DOWN'} (conf: {confidence:.4f}), "
                    f"Signal: {signal}, Portfolio Value: ${portfolio_value:.2f}"
                )
        
        # Calculate accuracy metrics
        if len(predictions) > 0 and len(actual_movements) > 0:
            accuracy = accuracy_score(actual_movements, predictions)
            precision = precision_score(actual_movements, predictions, zero_division=0)
            recall = recall_score(actual_movements, predictions, zero_division=0)
            f1 = f1_score(actual_movements, predictions, zero_division=0)
            conf_matrix = confusion_matrix(actual_movements, predictions)
            
            logger.info(f"Model prediction performance:")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")
            logger.info(f"Confusion Matrix:\n{conf_matrix}")
            
            # Add metrics to the metrics dictionary
            self.metrics['accuracy'] = accuracy
            self.metrics['precision'] = precision
            self.metrics['recall'] = recall
            self.metrics['f1_score'] = f1
        
        # Update final metrics
        final_value = portfolio_values[-1] if portfolio_values else self.initial_balance
        self.metrics['final_balance'] = final_value
        self.metrics['total_profit'] = final_value - self.initial_balance
        self.metrics['max_drawdown'] = max_drawdown
        
        if self.metrics['total_trades'] > 0:
            self.metrics['win_rate'] = self.metrics['winning_trades'] / self.metrics['total_trades']
        
        # Create a dataframe for visualization
        performance_df = pd.DataFrame({
            'date': dates,
            'portfolio_value': portfolio_values
        })
        
        return performance_df

    def plot_performance(self, performance_df, price_df):
        """
        Plot backtest performance and trade points.
        
        Args:
            performance_df: DataFrame with portfolio values
            price_df: DataFrame with price data
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot price data
        ax1.plot(price_df.index, price_df['Close'], label='Stock Price', color='blue')
        ax1.set_title(f'{self.symbol} Stock Price and Trades')
        ax1.set_ylabel('Price ($)')
        
        # Mark buy and sell points
        for trade in self.trade_log:
            if trade['action'] == 'BUY':
                ax1.scatter(trade['date'], trade['price'], color='green', marker='^', s=100)
            elif trade['action'] == 'SELL':
                ax1.scatter(trade['date'], trade['price'], color='red', marker='v', s=100)
        
        # Plot portfolio value
        ax2.plot(performance_df['date'], performance_df['portfolio_value'], label='Portfolio Value', color='purple')
        ax2.set_title('Portfolio Value Over Time')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Value ($)')
        
        # Format dates on x-axis
        plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        plt.savefig(f'trading_performance_{self.symbol}.png')
        plt.close()
        
        logger.info(f"Performance plot saved as trading_performance_{self.symbol}.png")

    def run(self, start_date=None, end_date=None, train_ratio=0.7):
        """
        Run the complete trading system workflow.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            train_ratio: Ratio of data to use for training
        """
        # Set logging level to DEBUG for more detailed logs
        logging.getLogger(__name__).setLevel(logging.DEBUG)
        
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365*3)  # 3 years of data
        
        # Prepare data
        df = self.prepare_data(start_date, end_date)
        
        # Split data into training and testing sets
        train_size = int(len(df) * train_ratio)
        df_train = df.iloc[:train_size]
        df_test = df.iloc[train_size:]
        
        logger.info(f"Training data: {len(df_train)} days, Testing data: {len(df_test)} days")
        
        # Extract features
        X_train, y_train = self.extract_features(df_train)
        
        # Train the model
        logger.info("Training model...")
        self.train_model(X_train, y_train)
        
        # Backtest on test data
        logger.info("Backtesting on test data...")
        performance_df = self.backtest(df_test)
        
        # Plot results
        self.plot_performance(performance_df, df_test)
        
        # Display performance metrics
        self.print_performance_metrics()
        
        return self.metrics

    def print_performance_metrics(self):
        """Print performance metrics from backtesting."""
        logger.info("\n" + "="*50)
        logger.info(f"TRADING PERFORMANCE METRICS FOR {self.symbol}")
        logger.info("="*50)
        logger.info(f"Initial balance: ${self.initial_balance:.2f}")
        logger.info(f"Final balance: ${self.metrics['final_balance']:.2f}")
        logger.info(f"Total profit/loss: ${self.metrics['total_profit']:.2f} ({self.metrics['total_profit']/self.initial_balance*100:.2f}%)")
        logger.info(f"Total trades: {self.metrics['total_trades']}")
        logger.info(f"Winning trades: {self.metrics['winning_trades']} ({self.metrics['win_rate']*100:.2f}% win rate)")
        logger.info(f"Losing trades: {self.metrics['losing_trades']}")
        logger.info(f"Maximum drawdown: {self.metrics['max_drawdown']*100:.2f}%")
        
        # Add model performance metrics if available
        if 'accuracy' in self.metrics:
            logger.info("\nMODEL PERFORMANCE METRICS")
            logger.info(f"Prediction accuracy: {self.metrics['accuracy']*100:.2f}%")
            logger.info(f"Precision: {self.metrics['precision']:.4f}")
            logger.info(f"Recall: {self.metrics['recall']:.4f}")
            logger.info(f"F1 Score: {self.metrics['f1_score']:.4f}")
        
        logger.info("="*50)

    def save_model(self, path="saved_models"):
        """Save the trained model and scalers."""
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, f"{self.symbol}_direction_model")
        scaler_path = os.path.join(path, f"{self.symbol}_feature_scaler.joblib")
        
        # Save the Keras model
        self.model.save(model_path)
        
        # Save the feature scaler
        joblib.dump(self.feature_scaler, scaler_path)
        
        logger.info(f"Model and scaler saved to {path}")

    def load_model(self, path="saved_models"):
        """Load a previously trained model and scalers."""
        model_path = os.path.join(path, f"{self.symbol}_direction_model")
        scaler_path = os.path.join(path, f"{self.symbol}_feature_scaler.joblib")
        
        # Load the Keras model
        self.model = tf.keras.models.load_model(model_path)
        
        # Load the feature scaler
        self.feature_scaler = joblib.load(scaler_path)
        
        logger.info(f"Model and scaler loaded from {path}")


def main():
    parser = argparse.ArgumentParser(description='Stock price direction prediction and trading simulation')
    parser.add_argument('symbol', type=str, help='Stock symbol to trade')
    parser.add_argument('--lookback', type=int, default=30, help='Number of days to use for prediction')
    parser.add_argument('--balance', type=float, default=100.0, help='Initial balance for backtesting')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of data to use for training')
    parser.add_argument('--days', type=int, default=1095, help='Number of days to fetch (default: 3 years)')
    
    args = parser.parse_args()
    
    # Initialize the trader
    trader = StockDirectionTrader(
        symbol=args.symbol,
        lookback_days=args.lookback,
        initial_balance=args.balance
    )
    
    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # Run the trading system
    trader.run(start_date, end_date, args.train_ratio)


if __name__ == "__main__":
    main() 