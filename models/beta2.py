import os
import sys
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.model_selection import TimeSeriesSplit, train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
except ImportError:  # pragma: no cover
    MinMaxScaler = StandardScaler = TimeSeriesSplit = train_test_split = None
    mean_squared_error = mean_absolute_error = r2_score = None
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model  # type: ignore
    from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional  # type: ignore
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization  # type: ignore
    from tensorflow.keras.layers import Input, concatenate  # type: ignore
    from tensorflow.keras.optimizers import Adam  # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # type: ignore
except ImportError:  # pragma: no cover
    tf = None
    Sequential = Model = load_model = None
    Dense = LSTM = Dropout = GRU = Bidirectional = None
    Conv1D = MaxPooling1D = BatchNormalization = None
    Input = concatenate = None
    Adam = None
    EarlyStopping = ModelCheckpoint = ReduceLROnPlateau = None
try:
    import ta
except ImportError:  # pragma: no cover
    ta = None
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self, prediction_days=60, feature_days=1, model_type='hybrid', model_path=None, scaler_path=None, company=None):
        """
        Initialize the StockPredictor with given parameters
        
        Args:
            prediction_days: Number of days to use for prediction window
            feature_days: Number of days to use for feature calculation
            model_type: Type of model to use ('lstm', 'gru', 'cnn_lstm', 'bilstm', 'hybrid')
            model_path: Path to save/load model
            scaler_path: Path to save/load scalers
            company: Stock symbol for predictions
        """
        self.prediction_days = prediction_days
        self.feature_days = feature_days if feature_days else prediction_days // 2
        self.model_type = model_type.lower()
        self.model = None
        self.company = company
        
        # Setup model and scaler paths
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = f"models/saved/{company}_{self.model_type}_model.keras" if company else f"models/saved/stock_{self.model_type}_model.keras"
            
        if scaler_path:
            self.scaler_path = scaler_path
        else:
            self.scaler_path = f"models/saved/{company}_scalers.joblib" if company else "models/saved/stock_scalers.joblib"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Initialize scalers with improved parameters
        self.scaler_price = MinMaxScaler(feature_range=(0, 1)) if MinMaxScaler else None
        self.scaler_features = StandardScaler() if StandardScaler else None  # Z-score normalization for features
        
        logger.info(f"Initialized StockPredictor with {self.model_type} model type, {self.prediction_days} prediction days")
        
    def fetch_data(self, start_date=None, end_date=None, symbol=None):
        """
        Fetch historical stock data and calculate technical indicators
        
        Args:
            start_date: Start date for data fetching (defaults to 2000 days ago)
            end_date: End date for data fetching (defaults to today)
            symbol: Stock symbol for fetching data (required)
            
        Returns:
            tuple: (price_data, feature_data) arrays for model input
        """
        try:
            # Set default dates if not provided
            if end_date is None:
                end_date = datetime.now()
                
            if start_date is None:
                start_date = end_date - timedelta(days=2000)  # Use 2000 days of historical data by default
            
            # Format dates for yfinance
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Use the instance company if no symbol provided
            symbol_to_use = symbol if symbol else self.company
            
            if not symbol_to_use:
                logger.error("No stock symbol provided for fetching data")
                return np.array([]), np.array([])
            
            logger.info(f"Fetching data for {symbol_to_use} from {start_str} to {end_str}")
            
            # Get S&P 500 data for market comparison
            try:
                sp500_data = yf.download('^GSPC', start=start_str, end=end_str)
                sp500_available = True
            except Exception as e:
                logger.warning(f"Could not fetch S&P 500 data: {e}")
                sp500_available = False
            
            # Fetch stock data using yfinance
            try:
                stock_data = yf.download(symbol_to_use, start=start_str, end=end_str)
                
                if stock_data.empty or len(stock_data) < 5:
                    logger.error(f"No data found for {symbol_to_use} or insufficient data points")
                    return np.array([]), np.array([])
                
                # Clean up the data - handle missing values
                stock_data = stock_data.dropna()
                
                # If we have very few data points, return error
                if len(stock_data) < 30:  # Need at least 30 days for meaningful analysis
                    logger.warning(f"Insufficient data points for {symbol_to_use}: {len(stock_data)} days")
                    return np.array([]), np.array([])
                
                logger.info(f"Successfully fetched {len(stock_data)} days of data")
                
                # Calculate technical indicators
                indicators_df = self.add_technical_indicators(stock_data)
                
                # Add market comparison if available
                if sp500_available:
                    try:
                        # Ensure same date range by aligning indexes
                        common_dates = indicators_df.index.intersection(sp500_data.index)
                        if len(common_dates) > 0:
                            # Add S&P 500 Close price
                            indicators_df['SP500_Close'] = np.nan  # Initialize with NaN
                            for date in common_dates:
                                indicators_df.loc[date, 'SP500_Close'] = sp500_data.loc[date, 'Close']
                                
                            # Calculate relative performance (normalized)
                            first_valid_idx = indicators_df['SP500_Close'].first_valid_index()
                            if first_valid_idx:
                                stock_start = indicators_df.loc[first_valid_idx, 'Close']
                                sp500_start = indicators_df.loc[first_valid_idx, 'SP500_Close']
                                
                                if stock_start > 0 and sp500_start > 0:
                                    indicators_df['RelPerf'] = (indicators_df['Close'] / stock_start) / (
                                        indicators_df['SP500_Close'] / sp500_start)
                                else:
                                    indicators_df['RelPerf'] = 1.0
                            else:
                                indicators_df['RelPerf'] = 1.0
                        else:
                            logger.warning("No overlapping dates between stock and S&P 500 data")
                            indicators_df['SP500_Close'] = 0
                            indicators_df['RelPerf'] = 1.0
                    except Exception as e:
                        logger.warning(f"Error calculating relative performance: {e}")
                        indicators_df['SP500_Close'] = 0
                        indicators_df['RelPerf'] = 1.0
                else:
                    # Add dummy columns if S&P data not available
                    indicators_df['SP500_Close'] = 0
                    indicators_df['RelPerf'] = 1.0
                
                # Fill any remaining NaN values with 0
                indicators_df = indicators_df.fillna(0)
                
                # Extract price and feature data
                price_data = indicators_df['Close'].values
                
                # Extract technical indicators (excluding date, open, high, low, close)
                feature_columns = [
                    'Volume', 'Returns', 'MA5', 'MA20', 'RSI', 'MACD', 
                    'BBWidth', '%K', '%D', 'ATR', 'OBV', 'SP500_Close', 'RelPerf'
                ]
                
                # Keep only features that exist in the dataframe
                available_features = [col for col in feature_columns if col in indicators_df.columns]
                if available_features:
                    feature_data = indicators_df[available_features].values
                else:
                    logger.warning("No technical features available")
                    feature_data = np.array([])
                
                return price_data, feature_data
                
            except Exception as e:
                logger.error(f"Error fetching stock data for {symbol_to_use}: {e}")
                import traceback
                traceback.print_exc()
                return np.array([]), np.array([])
            
        except Exception as e:
            logger.error(f"Error in fetch_data: {e}")
            import traceback
            traceback.print_exc()
            return np.array([]), np.array([])
            
    def add_technical_indicators(self, data):
        """
        Calculate technical indicators for the given data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        try:
            # Make a copy to avoid modifying the original
            df = data.copy()
            
            # Basic price data checks
            if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                missing = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col not in df.columns]
                logger.warning(f"Missing columns for technical indicators: {missing}")
                # Fill in missing columns with Close
                for col in missing:
                    if col != 'Volume':
                        df[col] = df['Close'] if 'Close' in df.columns else 0
                    else:
                        df[col] = 0
            
            # Calculate daily returns
            df['Returns'] = df['Close'].pct_change() * 100
            
            # Calculate moving averages
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            
            # Calculate RSI
            try:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                
                rs = avg_gain / avg_loss
                df['RSI'] = 100 - (100 / (1 + rs))
            except Exception as e:
                logger.warning(f"Error calculating RSI: {e}")
                df['RSI'] = 50  # Default value
            
            # Calculate MACD
            try:
                exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = exp1 - exp2
            except Exception as e:
                logger.warning(f"Error calculating MACD: {e}")
                df['MACD'] = 0  # Default value
            
            # Calculate Bollinger Bands
            try:
                df['MA20_std'] = df['Close'].rolling(window=20).std()
                df['BBUpper'] = df['MA20'] + (df['MA20_std'] * 2)
                df['BBLower'] = df['MA20'] - (df['MA20_std'] * 2)
                df['BBWidth'] = (df['BBUpper'] - df['BBLower']) / df['MA20']
            except Exception as e:
                logger.warning(f"Error calculating Bollinger Bands: {e}")
                df['BBWidth'] = 0.1  # Default value
            
            # Calculate Stochastic Oscillator
            try:
                high_14 = df['High'].rolling(window=14).max()
                low_14 = df['Low'].rolling(window=14).min()
                df['%K'] = ((df['Close'] - low_14) / (high_14 - low_14)) * 100
                df['%D'] = df['%K'].rolling(window=3).mean()
            except Exception as e:
                logger.warning(f"Error calculating Stochastic Oscillator: {e}")
                df['%K'] = 50  # Default value
                df['%D'] = 50  # Default value
            
            # Calculate Average True Range
            try:
                df['TR1'] = abs(df['High'] - df['Low'])
                df['TR2'] = abs(df['High'] - df['Close'].shift())
                df['TR3'] = abs(df['Low'] - df['Close'].shift())
                df['TR'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
                df['ATR'] = df['TR'].rolling(window=14).mean()
                df.drop(['TR1', 'TR2', 'TR3', 'TR'], axis=1, inplace=True)
            except Exception as e:
                logger.warning(f"Error calculating ATR: {e}")
                df['ATR'] = df['Close'] * 0.02  # Default 2% of price
            
            # Calculate On-Balance Volume
            try:
                df['OBV'] = (df['Volume'] * (np.sign(df['Close'].diff()))).fillna(0).cumsum()
            except Exception as e:
                logger.warning(f"Error calculating OBV: {e}")
                df['OBV'] = df['Volume'].cumsum()  # Default cumulative volume
            
            # Fill NaN values with appropriate methods
            for col in df.columns:
                if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            # Return original data if calculation fails
            return data

    def prepare_data(self, price_data, feature_data=None):
        """Prepare the price and feature data for model training
        
        Args:
            price_data: Historical price data array
            feature_data: Technical indicator data array (optional)
            
        Returns:
            x_price: Price input sequences
            x_features: Feature input sequences (or None if not available)
            y: Target values
        """
        try:
            # Ensure price_data is 1-dimensional
            if len(price_data.shape) > 1:
                price_data = price_data.flatten()
            
            # Check if there's enough data
            if len(price_data) < self.prediction_days:
                logger.warning(f"Not enough price data for model training. Need at least {self.prediction_days} points.")
                return None, None, None
            
            # Scale the price data
            price_data_2d = price_data.reshape(-1, 1)
            scaled_prices = self.scaler_price.fit_transform(price_data_2d).flatten()
            
            # Create sequences for price data
            x_price = []
            y = []
            
            for i in range(len(scaled_prices) - self.prediction_days):
                x_price.append(scaled_prices[i:i + self.prediction_days])
                y.append(scaled_prices[i + self.prediction_days])
            
            # Convert to numpy arrays
            x_price = np.array(x_price)
            y = np.array(y)
            
            # Reshape for LSTM input [samples, time steps, features]
            x_price = x_price.reshape(x_price.shape[0], x_price.shape[1], 1)
            
            # Process feature data if available
            x_features = None
            if feature_data is not None and self.model_type == 'hybrid':
                try:
                    # Handle different feature data shapes
                    if isinstance(feature_data, np.ndarray) and feature_data.size > 0:
                        # Fit scaler on feature data
                        self.scaler_features.fit(feature_data)
                        
                        # Scale the feature data
                        scaled_features = self.scaler_features.transform(feature_data)
                        
                        # Check for NaN or infinite values
                        if np.isnan(scaled_features).any() or np.isinf(scaled_features).any():
                            logger.warning("Found NaN or infinite values in feature data. Replacing with zeros.")
                            scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=1.0, neginf=-1.0)
                        
                        # Create sequences for feature data (use feature_days)
                        feature_window = min(self.feature_days, self.prediction_days)
                        x_features = []
                        
                        for i in range(len(scaled_features) - feature_window):
                            x_features.append(scaled_features[i:i + feature_window])
                        
                        x_features = np.array(x_features)
                        
                        # Make sure feature sequences align with price sequences
                        # If we have more feature sequences than price, trim from the beginning
                        if len(x_features) > len(x_price):
                            x_features = x_features[-len(x_price):]
                        # If we have fewer feature sequences than price, trim price from the end
                        elif len(x_features) < len(x_price):
                            x_price = x_price[:len(x_features)]
                            y = y[:len(x_features)]
                        
                        logger.info(f"Prepared {len(x_features)} sequences with {x_features.shape[2]} features")
                    else:
                        logger.warning("Feature data is empty or not in the correct format. Using price-only model.")
                except Exception as e:
                    logger.error(f"Error processing feature data: {e}")
                    logger.warning("Falling back to price-only model")
                
            return x_price, x_features, y
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return None, None, None

    def build_model(self, feature_input_shape=None):
        """Build the model with advanced architecture for stock price prediction
        
        Args:
            feature_input_shape: Shape of feature input (optional)
            
        Returns:
            A compiled Keras model
        """
        try:
            # Define price input
            price_input = Input(shape=(self.prediction_days, 1), name='price_input')
            
            # Ensure at least 1 LSTM layer for price processing
            price_x = LSTM(units=128, return_sequences=True, name='price_lstm_1')(price_input)
            price_x = Dropout(0.2)(price_x)
            price_x = LSTM(units=128, return_sequences=False, name='price_lstm_2')(price_x)
            price_x = Dropout(0.2)(price_x)
            
            # If hybrid model and feature input shape is provided
            if self.model_type == 'hybrid' and feature_input_shape is not None:
                # Technical indicator input
                feature_input = Input(shape=feature_input_shape, name='feature_input')
                
                # Process technical indicators with LSTM layers
                feature_x = LSTM(units=64, return_sequences=True, name='feature_lstm_1')(feature_input)
                feature_x = Dropout(0.2)(feature_x)
                feature_x = LSTM(units=64, return_sequences=False, name='feature_lstm_2')(feature_x)
                feature_x = Dropout(0.2)(feature_x)
                
                # Combine price and feature data
                combined = concatenate([price_x, feature_x], name='combined_features')
                
                # Add dense layers
                x = Dense(128, activation='relu', name='dense_1')(combined)
                x = Dropout(0.2)(x)
                x = Dense(64, activation='relu', name='dense_2')(x)
                x = Dropout(0.1)(x)
                
                # Output layer
                outputs = Dense(1, name='output')(x)
                
                # Build model with dual inputs
                model = Model(inputs=[price_input, feature_input], outputs=outputs)
                
            else:
                # For price-only model (LSTM)
                x = Dense(64, activation='relu', name='dense_1')(price_x)
                x = Dropout(0.1)(x)
                outputs = Dense(1, name='output')(x)
                
                # Build model with single input
                model = Model(inputs=price_input, outputs=outputs)
            
            # Compile with Huber loss for robustness to outliers
            try:
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss=tf.keras.losses.Huber(delta=1.0),  # Huber loss is more robust for price predictions
                    metrics=['mean_absolute_error', 'mean_absolute_percentage_error']
                )
            except Exception as e:
                logger.warning(f"Error using Adam optimizer with Huber loss: {e}. Falling back to default configuration.")
                model.compile(
                    optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['mean_absolute_error']
                )
                
            # Print model summary
            model.summary()
            
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {e}")
            # Return a very basic model as fallback
            price_input = Input(shape=(self.prediction_days, 1))
            x = LSTM(units=50, return_sequences=False)(price_input)
            outputs = Dense(1)(x)
            model = Model(inputs=price_input, outputs=outputs)
            model.compile(optimizer='adam', loss='mean_squared_error')
            logger.warning("Using fallback model due to build error")
            return model

    def train_model(self, x_price, x_features, y, epochs=50, batch_size=32, validation_split=0.2):
        """Train the model with enhanced training parameters and callbacks"""
        try:
            # Create callbacks for better training
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001,
                verbose=1
            )
            
            # Fix: Update checkpoint path for Keras 3 compatibility
            model_checkpoint = ModelCheckpoint(
                filepath=self.model_path,
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            )
            
            callbacks = [early_stopping, reduce_lr, model_checkpoint]
            
            # Set up the batch size based on data size
            batch_size = min(batch_size, len(y) // 4)  # Ensure batch size isn't too large
            batch_size = max(batch_size, 8)  # But not too small either
            
            # Add cosine annealing learning rate schedule for better convergence
            lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
                lambda epoch, lr: 0.001 * (0.5 * (1 + np.cos(epoch / epochs * np.pi)))
            )
            callbacks.append(lr_scheduler)
            
            # Set up training weights to focus more on recent data
            # This helps the model prioritize recent market behavior
            sample_weights = None
            if len(y) > 100:  # Only use weights for sufficient data
                sample_weights = np.linspace(0.5, 1.0, len(y))  # Increasing weights for more recent samples
            
            # Train the model based on available data
            if self.model_type == 'hybrid' and x_features is not None and x_features.size > 0:
                history = self.model.fit(
                    [x_price, x_features], y,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    callbacks=callbacks,
                    sample_weight=sample_weights,
                    verbose=1
                )
            else:
                history = self.model.fit(
                    x_price, y,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    callbacks=callbacks,
                    sample_weight=sample_weights,
                    verbose=1
                )
            
            # Save the model and scalers
            try:
                # Save the trained model
                self.model.save(self.model_path)
                
                # Save the scalers
                joblib.dump({
                    'price': self.scaler_price,
                    'features': self.scaler_features
                }, self.scaler_path)
                
                logger.info("Model and scalers saved successfully")
            except Exception as e:
                logger.error(f"Error saving model or scalers: {e}")
            
            return history
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            return None

    def plot_training_history(self, history):
        """Plot the training and validation loss curves"""
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{self.company} Model Training History')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"models/saved/{self.company}_training_history.png")
        plt.close()

    def make_predictions(self, price_data, feature_data=None):
        """
        Central prediction method used by both predict_next_day and predict_future
        
        Args:
            price_data: Historical price data array
            feature_data: Technical indicators data array (optional)
            
        Returns:
            array: Predicted prices in their original scale
        """
        try:
            # Ensure price data is 1-dimensional
            if len(price_data.shape) > 1:
                price_data = price_data.flatten()
            
            # Check if there's enough data
            if len(price_data) < self.prediction_days:
                logger.warning(f"Not enough price data. Need at least {self.prediction_days} points.")
                return np.array([])
            
            # Get the last prediction_days values for price
            last_price_sequence = price_data[-self.prediction_days:].reshape(-1, 1)
            
            try:
                # Scale the price data
                scaled_price = self.scaler_price.transform(last_price_sequence)
                x_price = scaled_price.reshape(1, self.prediction_days, 1)
            except Exception as e:
                logger.error(f"Error scaling price data: {e}")
                return np.array([])
            
            # Process feature data if available and using hybrid model
            use_features = False
            x_features = None
            
            if feature_data is not None and self.model_type == 'hybrid':
                try:
                    feature_window = min(self.feature_days, self.prediction_days)
                    
                    # Check if we have enough feature data
                    if isinstance(feature_data, np.ndarray) and feature_data.size > 0:
                        # Handle different feature data shapes
                        if len(feature_data.shape) == 1:  # Single feature
                            if len(feature_data) < feature_window:
                                logger.warning(f"Not enough feature data points. Got {len(feature_data)}, need {feature_window}")
                            else:
                                last_feature_sequence = feature_data[-feature_window:].reshape(-1, 1)
                                use_features = True
                        else:  # Multiple features
                            if len(feature_data) < feature_window:
                                logger.warning(f"Not enough feature data points. Got {len(feature_data)}, need {feature_window}")
                            else:
                                last_feature_sequence = feature_data[-feature_window:]
                                use_features = True
                        
                        # If we have feature data to use
                        if use_features:
                            # Check for NaN or infinite values
                            if np.isnan(last_feature_sequence).any() or np.isinf(last_feature_sequence).any():
                                logger.warning("Found NaN or infinite values in feature data. Replacing with zeros.")
                                last_feature_sequence = np.nan_to_num(last_feature_sequence, nan=0.0, posinf=1.0, neginf=-1.0)
                            
                            # Scale the feature data
                            scaled_features = self.scaler_features.transform(last_feature_sequence)
                            x_features = scaled_features.reshape(1, feature_window, scaled_features.shape[1])
                except Exception as e:
                    logger.error(f"Error processing feature data: {e}")
                    use_features = False
            
            # Make prediction based on available data
            try:
                if use_features and x_features is not None:
                    prediction_scaled = self.model.predict([x_price, x_features], verbose=0)
                else:
                    prediction_scaled = self.model.predict(x_price, verbose=0)
                
                # Handle different prediction output shapes
                if len(prediction_scaled.shape) > 2:  # Multiple timestep output
                    prediction_scaled = prediction_scaled.reshape(prediction_scaled.shape[1], 1)
                
                # Convert back to original scale
                predictions = self.scaler_price.inverse_transform(prediction_scaled).flatten()
                
                # Ensure predictions are positive
                predictions = np.maximum(predictions, 0.01)
                
                return predictions
                
            except Exception as e:
                logger.error(f"Error making prediction: {e}")
                import traceback
                traceback.print_exc()
                return np.array([])
            
        except Exception as e:
            logger.error(f"Error in make_predictions: {e}")
            return np.array([])

    def predict_next_day(self, price_data, feature_data=None):
        """Predicts the stock price for the next trading day.
        
        Args:
            price_data: Historical price data array
            feature_data: Technical indicators data array (optional)
            
        Returns:
            float: Predicted price for the next trading day
        """
        try:
            # Ensure price_data is one-dimensional
            if len(price_data.shape) > 1:
                price_data = price_data.flatten()
            
            # Ensure we have enough data
            if len(price_data) < self.prediction_days:
                logger.warning(f"Not enough price data. Got {len(price_data)}, need {self.prediction_days}")
                return None
            
            # Get current market trend and recent price
            current_price = price_data[-1]
            
            # Calculate market trend using recent prices
            recent_window = min(20, len(price_data))  # Use at most 20 recent days for trend
            recent_prices = price_data[-recent_window:]
            trend_slope = np.polyfit(np.arange(len(recent_prices)), recent_prices, 1)[0]
            trend_direction = "UP" if trend_slope > 0 else "DOWN"
            
            logger.info(f"Current price: {current_price:.2f}, Recent trend: {trend_direction} ({trend_slope:.4f})")
            
            # Make predictions using base method
            try:
                predictions = self.make_predictions(price_data, feature_data)
                
                if len(predictions) == 0:
                    logger.warning("No predictions generated")
                    return None
                
                # Get the next day prediction (first value in predictions)
                next_day_pred = predictions[0]
                
                # Apply market sentiment and trend awareness
                # If trend is strong, adjust prediction to better follow the trend
                if abs(trend_slope) > 0.01 * current_price:  # If trend is significant (>1% of price)
                    trend_factor = min(0.2, abs(trend_slope) / current_price)  # Max 20% influence
                    
                    # Ensure prediction follows strong trends more closely
                    if (trend_slope > 0 and next_day_pred < current_price) or (trend_slope < 0 and next_day_pred > current_price):
                        # If prediction contradicts a strong trend, adjust it
                        logger.info(f"Adjusting prediction to better follow strong {trend_direction} trend")
                        adjustment = current_price * trend_factor * (1 if trend_slope > 0 else -1)
                        next_day_pred = current_price + adjustment
                
                # Ensure prediction is not negative
                next_day_pred = max(0, next_day_pred)
                
                # Ensure prediction is not unrealistically far from current price
                max_daily_move = current_price * 0.1  # Max 10% daily move
                if abs(next_day_pred - current_price) > max_daily_move:
                    direction = 1 if next_day_pred > current_price else -1
                    next_day_pred = current_price + (direction * max_daily_move)
                    logger.info(f"Limited prediction to realistic daily move: {current_price:.2f} → {next_day_pred:.2f}")
                
                logger.info(f"Final prediction for next day: ${next_day_pred:.2f}")
                return next_day_pred
                
            except Exception as e:
                logger.error(f"Error in prediction process: {e}")
                return None
        
        except Exception as e:
            logger.error(f"Error predicting next day price: {e}")
            return None

    def predict_future(self, price_data, feature_data=None, future_days=60):
        """Predicts stock prices for multiple future days.
        
        This method uses an iterative approach for multi-step forecasting,
        feeding each prediction back into the model for the next prediction.
        
        Args:
            price_data: Historical price data array
            feature_data: Technical indicators data array (optional)
            future_days: Number of days to predict into the future
            
        Returns:
            array: Predicted prices for future_days
        """
        try:
            # Ensure price_data is one-dimensional
            if len(price_data.shape) > 1:
                price_data = price_data.flatten()
            
            # Ensure we have enough data
            if len(price_data) < self.prediction_days:
                logger.warning(f"Not enough price data. Got {len(price_data)}, need {self.prediction_days}")
                return np.array([])
            
            # Get current market trend for realistic projections
            current_price = price_data[-1]
            
            # Calculate market trend using recent prices
            recent_window = min(30, len(price_data))  # Use at most 30 recent days for trend
            recent_prices = price_data[-recent_window:]
            trend_coeffs = np.polyfit(np.arange(len(recent_prices)), recent_prices, 2)  # Quadratic fit for better trend modeling
            
            # Calculate long-term market volatility
            if len(price_data) > 20:
                daily_returns = np.diff(price_data) / price_data[:-1]
                volatility = np.std(daily_returns)
            else:
                volatility = 0.015  # Default volatility of 1.5% if not enough data
            
            logger.info(f"Market trend analysis - Current price: ${current_price:.2f}, Volatility: {volatility:.4f}")
            
            # Check if we're using a hybrid model but don't have enough feature data
            feature_data_is_valid = feature_data is not None and (
                isinstance(feature_data, np.ndarray) and feature_data.size > 0 and 
                (len(feature_data.shape) == 1 or len(feature_data) >= self.feature_days)
            )
            
            # If hybrid model but no valid feature data, switch to fallback approach
            using_fallback = self.model_type == 'hybrid' and not feature_data_is_valid
            if using_fallback:
                logger.warning("Hybrid model detected but feature data is insufficient. Using trend-based fallback predictions.")
            
            # Copy the price data to avoid modifying the original
            model_input = price_data.copy()
            
            # Initialize the array for future predictions
            future_predictions = []
            
            # If we need to use fallback for hybrid model or future_days is very short, use trend-based approach
            if using_fallback or future_days <= 5:
                logger.info("Using simplified trend-based prediction approach")
                
                # Simple trend-based predictions with volatility
                linear_trend = np.polyfit(np.arange(len(recent_prices)), recent_prices, 1)[0]
                trend_direction = 1 if linear_trend > 0 else -1
                
                # Start with current price
                last_price = current_price
                
                for i in range(future_days):
                    # Calculate trend component
                    trend_component = linear_trend * (1 + (i * 0.01))  # Slightly increasing trend effect
                    
                    # Add volatility component
                    vol_component = np.random.normal(0, volatility * last_price)
                    
                    # Calculate next price
                    next_price = last_price + trend_component + vol_component
                    
                    # Ensure prediction is not negative and not unrealistically high
                    next_price = max(0.1, next_price)
                    max_daily_change = last_price * 0.1  # Max 10% change
                    if abs(next_price - last_price) > max_daily_change:
                        next_price = last_price + (trend_direction * max_daily_change)
                    
                    future_predictions.append(next_price)
                    last_price = next_price
            else:
                # For standard prediction with valid data, use the iterative approach with model
                try:
                    # Make feature data always the most recent available if valid
                    latest_feature_data = None
                    if feature_data_is_valid:
                        if len(feature_data.shape) == 1:
                            latest_feature_data = feature_data[-self.feature_days:] if len(feature_data) >= self.feature_days else None
                        else:
                            latest_feature_data = feature_data[-self.feature_days:] if len(feature_data) >= self.feature_days else None
                    
                    # Make predictions for each future day
                    for i in range(future_days):
                        # Prepare input for the model (sliding window)
                        input_data = model_input[-self.prediction_days:]
                        
                        # Try to use the main model
                        predictions = self.make_predictions(input_data, latest_feature_data)
                        
                        if len(predictions) == 0:
                            logger.warning(f"Failed to predict day {i+1}")
                            # Use trend-based estimation if prediction fails
                            if len(future_predictions) > 0:
                                last_pred = future_predictions[-1]
                            else:
                                last_pred = current_price
                            
                            # Simple trend continuation with noise
                            trend_pred = last_pred * (1 + (np.random.normal(0, 1) * volatility) + (trend_coeffs[0] / current_price))
                            future_predictions.append(max(0.1, trend_pred))  # Ensure positive price
                            continue
                        
                        next_pred = predictions[0]
                        
                        # Apply realistic constraints based on market conditions
                        # 1. Limit daily changes based on historical volatility
                        prev_price = model_input[-1]
                        max_daily_change = prev_price * (volatility * 3)  # Allow up to 3x daily volatility
                        
                        if abs(next_pred - prev_price) > max_daily_change:
                            direction = 1 if next_pred > prev_price else -1
                            next_pred = prev_price + (direction * max_daily_change)
                            logger.debug(f"Limited prediction to realistic daily change: {prev_price:.2f} → {next_pred:.2f}")
                        
                        # 2. Apply trend guidance for long-term predictions
                        # The further we predict, the more we rely on trend
                        if i > 10:  # After 10 days, gradually increase trend influence
                            trend_factor = min(0.7, (i - 10) / future_days)  # Up to 70% trend influence
                            
                            # Calculate expected price based on trend
                            trend_value = np.poly1d(trend_coeffs)(recent_window + i)
                            
                            # Blend model prediction with trend-based prediction
                            blended_pred = (next_pred * (1 - trend_factor)) + (trend_value * trend_factor)
                            next_pred = blended_pred
                        
                        # Add some random noise proportional to volatility and prediction distance
                        noise_factor = volatility * (1 + (i / future_days))  # Increasing noise with prediction distance
                        noise = np.random.normal(0, noise_factor * current_price * 0.01)  # 1% of price * noise factor
                        next_pred += noise
                        
                        # Ensure prediction is not negative
                        next_pred = max(0.1, next_pred)
                        
                        # Add to our predictions
                        future_predictions.append(next_pred)
                        
                        # Update model input for next iteration
                        model_input = np.append(model_input, next_pred)
                        
                except Exception as e:
                    logger.error(f"Error in iterative prediction: {e}")
                    import traceback
                    traceback.print_exc()
                    return np.array([])
            
            return np.array(future_predictions)
        
        except Exception as e:
            logger.error(f"Error in predict_future: {e}")
            import traceback
            traceback.print_exc()
            return np.array([])

    def evaluate_model(self, y_true, y_pred):
        """Evaluate model performance using multiple metrics"""
        try:
            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Calculate directional accuracy (if prediction got the direction right)
            y_true_direction = np.diff(y_true) > 0
            y_pred_direction = np.diff(y_pred) > 0
            directional_accuracy = np.mean(y_true_direction == y_pred_direction)
            
            # Display results
            logger.info(f"Model Evaluation Metrics:")
            logger.info(f"MSE: {mse:.4f}")
            logger.info(f"RMSE: {rmse:.4f}")
            logger.info(f"MAE: {mae:.4f}")
            logger.info(f"R²: {r2:.4f}")
            logger.info(f"Directional Accuracy: {directional_accuracy:.4f}")
            
            # Return metrics as dictionary
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'directional_accuracy': directional_accuracy
            }
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
            
    def save_model(self):
        """Save the trained model and scalers"""
        try:
            # Save model
            self.model.save(self.model_path)
            
            # Save scalers
            scalers = {
                'price': self.scaler_price,
                'features': self.scaler_features
            }
            joblib.dump(scalers, self.scaler_path)
            
            logger.info(f"Model and scalers saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
            
    def load_model(self):
        """Load a previously trained model and scalers"""
        try:
            import keras
            # Load model
            self.model = keras.models.load_model(self.model_path)
            
            # Load scalers
            scalers = joblib.load(self.scaler_path)
            self.scaler_price = scalers['price']
            self.scaler_features = scalers['features']
            
            logger.info(f"Model and scalers loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

def plot_predictions(actual_prices, predicted_prices, company, metrics=None):
    """Plot actual vs predicted prices from test data with metrics"""
    plt.figure(figsize=(14, 7))
    plt.plot(actual_prices, color="black", label=f"Actual {company} Price")
    plt.plot(predicted_prices, color="green", label=f"Predicted {company} Price")
    
    # Add metrics to the plot if provided
    if metrics:
        info_text = (
            f"RMSE: {metrics['rmse']:.2f}\n"
            f"MAE: {metrics['mae']:.2f}\n"
            f"R²: {metrics['r2']:.2f}\n"
            f"Dir Acc: {metrics['directional_accuracy']:.2f}"
        )
        plt.annotate(info_text, xy=(0.02, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
    
    plt.title(f"{company} Share Price Prediction")
    plt.xlabel("Time")
    plt.ylabel(f"{company} Share Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"models/saved/{company}_predictions.png")
    plt.show()

def plot_actual_and_future(test_data, predicted_prices, future_predictions, company, prediction_days=60, metrics=None):
    """
    Enhanced plotting function for actual data, predictions, and future forecast
    """
    import matplotlib.dates as mdates
    
    plt.figure(figsize=(14, 7))
    
    # 1) Plot actual test data
    plt.plot(test_data.index, test_data['Close'], label="Actual Price", color="black")
    
    # 2) Plot predicted test data
    pred_start = prediction_days
    pred_end = len(test_data)
    predicted_dates = test_data.index[pred_start:pred_end]
    
    # If there's a mismatch in lengths, truncate the longer one
    if len(predicted_prices) != len(predicted_dates):
        min_len = min(len(predicted_prices), len(predicted_dates))
        predicted_prices = predicted_prices[:min_len]
        predicted_dates = predicted_dates[:min_len]
    
    plt.plot(predicted_dates, predicted_prices, label="Predicted Price", color="green")
    
    # 3) Plot the future forecast
    if len(future_predictions) > 0:
        last_date = test_data.index[-1]
        # Generate future dates (business days only)
        future_dates = pd.date_range(
            start=last_date,
            periods=len(future_predictions) + 1,
            freq='B'  # Business days
        )[1:]  # Drop the first date to start on "the next day"
        
        plt.plot(future_dates, future_predictions, label="Future Forecast", color="blue")
        
        # Add confidence intervals (simple estimation based on historical error)
        if metrics:
            error_margin = metrics['rmse'] * 1.96  # ~95% confidence interval
            plt.fill_between(
                future_dates,
                future_predictions - error_margin,
                future_predictions + error_margin,
                color='blue',
                alpha=0.2,
                label='95% Confidence Interval'
            )
    
    # Add metrics annotation if provided
    if metrics:
        info_text = (
            f"RMSE: {metrics['rmse']:.2f}\n"
            f"MAE: {metrics['mae']:.2f}\n"
            f"R²: {metrics['r2']:.2f}\n"
            f"Dir Acc: {metrics['directional_accuracy']:.2f}"
        )
        plt.annotate(info_text, xy=(0.02, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
    
    # Format the x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    
    plt.title(f"{company} Share Price & Forecast")
    plt.xlabel("Date")
    plt.ylabel(f"{company} Share Price")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"models/saved/{company}_forecast.png")
    plt.show()

def main(stock_symbol, model_type='hybrid', prediction_days=60, feature_days=30):
    """Main function to run the model training and prediction pipeline"""
    # Initialize parameters
    company = stock_symbol
    
    # Fix: Use March 23, 2025 as the current date for consistent backtesting
    current_date = datetime(2025, 3, 23).replace(tzinfo=None)
    
    # Fix: Adjust date ranges to use proper historical data
    train_start = datetime(2015, 1, 1)  # Start date for training data
    train_end = current_date - timedelta(days=60)  # End date for training, 60 days before now
    test_start = train_end  # Start date for test data (right after training ends)
    test_end = current_date  # End date for test data (now)
    
    logger.info(f"Date ranges: Training {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}, "
               f"Testing {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
    
    try:
        # Initialize predictor with model type
        predictor = StockPredictor(prediction_days=prediction_days, model_type=model_type, company=company)
        
        # Fetch historical data
        price_data, feature_data = predictor.fetch_data()
        
        if price_data.empty or len(price_data) < prediction_days * 2:
            raise ValueError(f"Not enough training data for {company}. Got {len(price_data)} rows, need at least {prediction_days * 2}.")
            
        logger.info(f"Training data shape: {price_data.shape}")
        logger.info(f"Training data range: {price_data.index.min()} to {price_data.index.max()}")
        logger.info(f"Training price range: {price_data.min():.2f} to {price_data.max():.2f}")
        
        # Prepare data for training
        x_price_train, x_features_train, y_train = predictor.prepare_data(price_data, feature_data)
        logger.info(f"Training sequences prepared. X price shape: {x_price_train.shape}, X features shape: {x_features_train.shape}")
        
        # Check if we have meaningful training data
        if x_price_train.size == 0 or y_train.size == 0:
            raise ValueError("Empty training data generated. Cannot train model.")
        
        # Build and train model
        logger.info(f"Building and training {model_type} model...")
        try:
            if model_type == 'hybrid' and x_features_train.size > 0:
                predictor.model = predictor.build_model(
                    (x_price_train.shape[1], 1), 
                    (x_features_train.shape[1], x_features_train.shape[2])
                )
                history = predictor.train_model(x_price_train, x_features_train, y_train)
            else:
                # Fallback to simpler model if no features or not hybrid model
                logger.info(f"Using price-only model architecture.")
                predictor.model = predictor.build_model((x_price_train.shape[1], 1), None)
                history = predictor.train_model(x_price_train, None, y_train)
            
            # Save the trained model
            predictor.save_model()
        except Exception as e:
            logger.error(f"Error training model: {e}")
            logger.warning("Loading saved model if available, or creating a fallback model.")
            
            try:
                # Try to load a previously saved model
                predictor.load_model()
            except:
                # Create a simple fallback model if loading fails
                logger.warning("No saved model found. Creating a simple fallback model.")
                predictor.model = predictor.build_model((prediction_days, 1), None)
                fallback_x = np.zeros((10, prediction_days, 1))
                fallback_y = np.zeros(10)
                predictor.model.fit(fallback_x, fallback_y, epochs=1, verbose=0)
        
        # Fetch and prepare test data
        logger.info(f"Fetching test data for {company}...")
        test_data = predictor.fetch_data(symbol=company)
        
        if test_data.empty:
            raise ValueError(f"No test data available for {company}")
            
        logger.info(f"Test data shape: {test_data.shape}")
        logger.info(f"Test data range: {test_data.index.min()} to {test_data.index.max()}")
        logger.info(f"Test price range: {test_data['Close'].min():.2f} to {test_data['Close'].max():.2f}")
        
        # Check if test data is sufficient
        if len(test_data) < prediction_days:
            logger.warning(f"Test data length ({len(test_data)}) is less than prediction_days ({prediction_days}). "
                         f"This may affect prediction accuracy.")
            
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
            total_price_data = pd.concat((price_data, test_data['Close']), axis=0)
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
        try:
            if model_type == 'hybrid' and test_feature_data is not None:
                predicted_prices = predictor.make_predictions(price_inputs, test_feature_data)
            else:
                predicted_prices = predictor.make_predictions(price_inputs, None)
                
            if len(predicted_prices) == 0:
                logger.error("Failed to generate predictions.")
                return None
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None
        
        # Evaluate model performance
        try:
            # Get actual prices, adjusting for the prediction window
            actual_prices = test_data['Close'].values[prediction_days:]
            
            # If there's a mismatch in lengths, truncate the longer one
            if len(predicted_prices) > len(actual_prices):
                predicted_prices = predicted_prices[:len(actual_prices)]
            elif len(predicted_prices) < len(actual_prices):
                actual_prices = actual_prices[:len(predicted_prices)]
                
            if len(predicted_prices) == 0 or len(actual_prices) == 0:
                logger.warning("No data available for evaluation.")
                metrics = None
            else:
                metrics = predictor.evaluate_model(actual_prices, predicted_prices)
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            metrics = None
        
        # Predict the next day's price
        try:
            if model_type == 'hybrid' and test_feature_data is not None:
                next_day_prediction = predictor.predict_next_day(price_inputs, test_feature_data)
            else:
                next_day_prediction = predictor.predict_next_day(price_inputs)
            
            logger.info(f"Prediction for next trading day: ${next_day_prediction:.2f}")
        except Exception as e:
            logger.error(f"Error predicting next day: {e}")
            next_day_prediction = None
        
        # Predict future 60-day prices
        try:
            # For hybrid models, make sure we have valid feature data
            valid_feature_data = (test_feature_data is not None and 
                                isinstance(test_feature_data, np.ndarray) and 
                                test_feature_data.size > 0)
            
            # Attempt future predictions
            future_predictions = []
            try:
                if model_type == 'hybrid' and valid_feature_data:
                    future_predictions = predictor.predict_future(price_inputs, test_feature_data, future_days=60)
                else:
                    # For non-hybrid models or when feature data is invalid, just use price data
                    if model_type == 'hybrid':
                        logger.warning("Hybrid model requested but no valid feature data found. Using price-only prediction approach.")
                    future_predictions = predictor.predict_future(price_inputs, None, future_days=60)
            except Exception as e:
                logger.error(f"Error in primary future prediction approach: {e}")
                # Fall back to simpler approach
                try:
                    logger.info("Trying fallback approach for future predictions...")
                    future_predictions = predictor.predict_future(price_inputs, None, future_days=60)
                except Exception as e2:
                    logger.error(f"Fallback prediction also failed: {e2}")
                
            if len(future_predictions) > 0:
                # Convert future predictions to float if they're numpy arrays
                future_predictions = [float(fp) if hasattr(fp, 'item') else float(fp) for fp in future_predictions]
                
                # Calculate trend from predictions
                trend = np.polyfit(np.arange(len(future_predictions)), future_predictions, 1)[0]
                trend_percentage = (trend / future_predictions[0]) * 100
                
                # Calculate max and min predictions
                max_price = np.max(future_predictions)
                min_price = np.min(future_predictions)
                max_change = (max_price - current_price) / current_price * 100
                min_change = (min_price - current_price) / current_price * 100
                
                print("\nForecast Summary (Next 60 Days):")
                print(f"Trend: {'UPWARD' if trend > 0 else 'DOWNWARD'} ({trend_percentage:.2f}% per day)")
                print(f"Maximum Price: ${max_price:.2f} ({max_change:.2f}%)")
                print(f"Minimum Price: ${min_price:.2f} ({min_change:.2f}%)")
                print(f"60-Day Price Target: ${future_predictions[-1]:.2f}")
            else:
                # If predictions still failed, create a simple trend-based forecast
                logger.warning("All prediction methods failed. Creating simple trend-based forecast.")
                
                # Calculate recent trend from actual data
                recent_window = min(30, len(price_inputs))
                recent_prices = price_inputs[-recent_window:]
                trend_coeff = np.polyfit(np.arange(len(recent_prices)), recent_prices, 1)[0]
                
                # Calculate volatility
                if len(price_inputs) > 20:
                    daily_returns = np.diff(price_inputs) / price_inputs[:-1]
                    volatility = np.std(daily_returns)
                else:
                    volatility = 0.015  # Default volatility
                
                # Generate trend-based predictions with some randomness
                future_predictions = []
                last_price = price_inputs[-1]
                
                for i in range(60):
                    # Add trend and some random noise proportional to volatility
                    noise = np.random.normal(0, volatility * last_price)
                    next_price = last_price + trend_coeff + noise
                    future_predictions.append(max(0.1, next_price))  # Ensure positive price
                    last_price = next_price
                
                # Summary statistics
                trend_percentage = (trend_coeff / price_inputs[-1]) * 100
                max_price = np.max(future_predictions)
                min_price = np.min(future_predictions)
                max_change = (max_price - current_price) / current_price * 100
                min_change = (min_price - current_price) / current_price * 100
                
                print("\nForecast Summary (Next 60 Days) - TREND-BASED ESTIMATE:")
                print(f"Trend: {'UPWARD' if trend_coeff > 0 else 'DOWNWARD'} ({trend_percentage:.2f}% per day)")
                print(f"Maximum Price: ${max_price:.2f} ({max_change:.2f}%)")
                print(f"Minimum Price: ${min_price:.2f} ({min_change:.2f}%)")
                print(f"60-Day Price Target: ${future_predictions[-1]:.2f}")
        
        except Exception as e:
            logger.error(f"Error predicting future: {e}")
            future_predictions = np.array([])
        
        # Plot the results with metrics
        try:
            if len(predicted_prices) > 0:
                plot_actual_and_future(
                    test_data=test_data, 
                    predicted_prices=predicted_prices,
                    future_predictions=future_predictions,
                    company=company,
                    prediction_days=predictor.prediction_days,
                    metrics=metrics
                )
            else:
                logger.error("Cannot create plot: No predictions available")
        except Exception as e:
            logger.error(f"Error creating plot: {e}")
        
        # Return results for potential further analysis
        return {
            'predictor': predictor,
            'metrics': metrics,
            'next_day': next_day_prediction,
            'future': future_predictions
        }
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    try:
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        
        # Check for debug flag
        debug = "--debug" in sys.argv
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            sys.argv.remove("--debug")
        
        # Parse command-line arguments
        company = "NVDA" if len(sys.argv) < 2 else sys.argv[1]
        model_type = "hybrid" if len(sys.argv) < 3 else sys.argv[2]
        
        logger.info(f"Initializing {model_type} model for {company}")
        logger.info(f"Predicting stock prices for {60} days")
        
        # Create a StockPredictor instance
        prediction_days = 60
        predictor = StockPredictor(prediction_days=prediction_days, model_type=model_type, company=company)
        
        # Fetch historical data
        price_data, feature_data = predictor.fetch_data()
        
        if len(price_data) < 200:  # Require at least 200 days of data
            logger.error(f"Not enough data found for {company}. Need at least 200 days of data.")
            sys.exit(1)
            
        # Prepare training data
        X_price, X_features, y = predictor.prepare_data(price_data, feature_data)
        
        # Setup the model
        feature_input_shape = None
        if X_features is not None and len(X_features) > 0:
            feature_input_shape = (X_features.shape[1], X_features.shape[2])
            
        predictor.model = predictor.build_model(feature_input_shape)
        
        # Train the model
        history = predictor.train_model(X_price, X_features, y)
        
        if history is None:
            logger.error("Model training failed")
            sys.exit(1)
        
        # Make predictions for next day
        next_day = predictor.predict_next_day(price_data, feature_data)
        
        if next_day is not None:
            # Print the prediction with current price for comparison
            current_price = price_data[-1]
            # Convert numpy values to float if needed
            if hasattr(current_price, 'item'):
                current_price = current_price.item()
            if hasattr(next_day, 'item'):
                next_day = next_day.item()
                
            change = (next_day - current_price) / current_price * 100
            direction = "UP" if next_day > current_price else "DOWN"
            
            print("\n" + "="*60)
            print(f"PREDICTION SUMMARY FOR {company}")
            print("="*60)
            print(f"Current Price: ${current_price:.2f}")
            print(f"Next Day Prediction: ${next_day:.2f} ({direction} {abs(change):.2f}%)")
            
            # Calculate some metrics using recent performance
            test_window = min(30, len(price_data) // 4)  # Use 30 days or 1/4 of data, whichever is smaller
            if test_window > prediction_days:
                # Use the last portion of data for backtesting
                test_start = len(price_data) - test_window
                
                actual_prices = price_data[test_start:]
                predictions = []
                
                for i in range(test_window - prediction_days):
                    pred = predictor.predict_next_day(price_data[test_start:test_start+i+prediction_days])
                    if pred is not None:
                        # Convert numpy values to float if needed
                        if hasattr(pred, 'item'):
                            pred = pred.item()
                        predictions.append(pred)
                
                if len(predictions) > 0:
                    # Compare predictions with actual prices
                    actuals = actual_prices[prediction_days+1:prediction_days+1+len(predictions)]
                    if len(actuals) == len(predictions):
                        # Convert actuals to floats if they're numpy arrays
                        actuals_float = [a.item() if hasattr(a, 'item') else float(a) for a in actuals]
                        mae = np.mean(np.abs(np.array(predictions) - np.array(actuals_float)))
                        mape = np.mean(np.abs((np.array(actuals_float) - np.array(predictions)) / np.array(actuals_float))) * 100
                        
                        print("\nModel Performance Metrics:")
                        print(f"Mean Absolute Error: ${mae:.2f}")
                        print(f"Mean Absolute Percentage Error: {mape:.2f}%")
                        print(f"Recent Prediction Accuracy: {100 - mape:.2f}%")
            
            # Make predictions for future days
            future_days = 60
            future_predictions = predictor.predict_future(price_data, feature_data, future_days)
            
            if len(future_predictions) > 0:
                # Convert future predictions to float if they're numpy arrays
                future_predictions = [float(fp) if hasattr(fp, 'item') else float(fp) for fp in future_predictions]
                
                # Calculate trend from predictions
                trend = np.polyfit(np.arange(len(future_predictions)), future_predictions, 1)[0]
                trend_percentage = (trend / future_predictions[0]) * 100
                
                # Calculate max and min predictions
                max_price = np.max(future_predictions)
                min_price = np.min(future_predictions)
                max_change = (max_price - current_price) / current_price * 100
                min_change = (min_price - current_price) / current_price * 100
                
                print("\nForecast Summary (Next 60 Days):")
                print(f"Trend: {'UPWARD' if trend > 0 else 'DOWNWARD'} ({trend_percentage:.2f}% per day)")
                print(f"Maximum Price: ${max_price:.2f} ({max_change:.2f}%)")
                print(f"Minimum Price: ${min_price:.2f} ({min_change:.2f}%)")
                print(f"60-Day Price Target: ${future_predictions[-1]:.2f}")
        
        else:
            logger.error("Failed to generate next day prediction")
        
        # Print usage instructions
        print("\nUsage:")
        print("python beta2.py [SYMBOL] [MODEL_TYPE] [--debug]")
        print("Example: python beta2.py AAPL hybrid")
        print("\nAvailable model types: lstm, gru, cnn_lstm, bilstm, hybrid")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
