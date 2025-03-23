import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential, Model # type: ignore
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D # type: ignore
from tensorflow.keras.layers import Input, Concatenate, Add, BatchNormalization, Attention # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint# type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import os
import joblib
import ta
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self, company, prediction_days=60, feature_days=30, model_type='hybrid'):
        self.company = company
        self.prediction_days = prediction_days
        self.feature_days = feature_days
        self.model_type = model_type
        self.scaler_price = MinMaxScaler(feature_range=(0, 1))
        self.scaler_features = StandardScaler()
        self.model = None
        self.model_path = f"models/saved/{company}_{model_type}_model.keras"
        self.scaler_path = f"models/saved/{company}_scalers.joblib"
        
        # Create directory if it doesn't exist
        os.makedirs("models/saved", exist_ok=True)
        
    def fetch_data(self, start_date, end_date):
        """Fetch stock data from Yahoo Finance with more data points"""
        try:
            # Get stock data
            ticker = yf.Ticker(self.company)
            data = ticker.history(start=start_date, end=end_date, interval="1d")
            
            if data.empty:
                raise ValueError(f"No data found for {self.company}")
            
            # Make a copy to avoid modifying the original dataframe
            data_with_features = data.copy()
                
            # Try to get additional market data: S&P 500 as benchmark
            try:
                sp500 = yf.download("^GSPC", start=start_date, end=end_date, interval="1d", progress=False)
                
                if not sp500.empty:
                    # Match dates and add S&P 500 as a feature
                    sp500 = sp500[['Close', 'Volume']].rename(
                        columns={'Close': 'SP500_Close', 'Volume': 'SP500_Volume'}
                    )
                    
                    # Fix timezone inconsistency before merging
                    # Convert both dataframes to timezone-naive for consistent merging
                    if data_with_features.index.tz is not None:
                        data_with_features.index = data_with_features.index.tz_localize(None)
                        
                    if sp500.index.tz is not None:
                        sp500.index = sp500.index.tz_localize(None)
                    
                    # Ensure index types match before merging
                    data_with_features.index = pd.to_datetime(data_with_features.index)
                    sp500.index = pd.to_datetime(sp500.index)
                    
                    # Now merge the dataframes with consistent timezone handling
                    data_with_features = pd.merge(data_with_features, sp500, left_index=True, right_index=True, how='left')
                else:
                    # If S&P 500 data couldn't be fetched, add dummy columns with stock's own values
                    logger.warning("S&P 500 data not available. Using stock's own values as fallback.")
                    data_with_features['SP500_Close'] = data_with_features['Close']
                    data_with_features['SP500_Volume'] = data_with_features['Volume']
            except Exception as e:
                # If S&P 500 data couldn't be fetched, add dummy columns with stock's own values
                logger.warning(f"Failed to fetch S&P 500 data: {e}. Using stock's own values as fallback.")
                data_with_features['SP500_Close'] = data_with_features['Close']
                data_with_features['SP500_Volume'] = data_with_features['Volume']
            
            # Forward fill any missing values
            data_with_features = data_with_features.fillna(method='ffill').fillna(method='bfill')
            
            # Verify that the required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SP500_Close', 'SP500_Volume']
            for col in required_cols:
                if col not in data_with_features.columns:
                    logger.warning(f"Column {col} missing from data. Adding dummy values.")
                    # Add a dummy column if missing
                    if col in ['SP500_Close']:
                        data_with_features[col] = data_with_features['Close']
                    elif col in ['SP500_Volume']:
                        data_with_features[col] = data_with_features['Volume']
                    else:
                        # Fallback for other columns
                        data_with_features[col] = 0
            
            return data_with_features
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise
            
    def add_technical_indicators(self, data):
        """Add technical indicators to the dataset with safe fallbacks"""
        try:
            # Create a copy to avoid modifying the original data
            df = data.copy()
            
            # Ensure all required base columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SP500_Close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns for technical indicators: {missing_cols}")
                # Add missing columns with fallbacks
                for col in missing_cols:
                    if col == 'SP500_Close':
                        df[col] = df['Close'] if 'Close' in df.columns else 1.0
                    elif col == 'Volume':
                        df[col] = 1000000  # Default volume
                    elif col in ['Open', 'High', 'Low']:
                        df[col] = df['Close'] if 'Close' in df.columns else 1.0
                    else:
                        df[col] = 1.0  # Default fallback
            
            # Make sure we don't have NaN values in critical columns
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # ---------- SAFELY ADD INDICATORS ----------
            # Price and volume-based indicators
            try:
                df['Returns'] = df['Close'].pct_change().fillna(0)
                df['LogReturns'] = np.log(df['Close']/df['Close'].shift(1)).fillna(0)
                df['VolChange'] = df['Volume'].pct_change().fillna(0)
            except Exception as e:
                logger.warning(f"Error calculating returns: {e}")
                df['Returns'] = 0
                df['LogReturns'] = 0
                df['VolChange'] = 0
            
            # Moving averages
            try:
                df['MA5'] = df['Close'].rolling(window=5).mean().fillna(df['Close'])
                df['MA10'] = df['Close'].rolling(window=10).mean().fillna(df['Close'])
                df['MA20'] = df['Close'].rolling(window=20).mean().fillna(df['Close'])
                df['MA50'] = df['Close'].rolling(window=50).mean().fillna(df['Close'])
            except Exception as e:
                logger.warning(f"Error calculating moving averages: {e}")
                df['MA5'] = df['Close']
                df['MA10'] = df['Close']
                df['MA20'] = df['Close']
                df['MA50'] = df['Close']
            
            # Exponential moving averages
            try:
                df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean().fillna(df['Close'])
                df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean().fillna(df['Close'])
            except Exception as e:
                logger.warning(f"Error calculating EMAs: {e}")
                df['EMA12'] = df['Close']
                df['EMA26'] = df['Close']
            
            # MACD
            try:
                df['MACD'] = df['EMA12'] - df['EMA26']
                df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean().fillna(0)
                df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            except Exception as e:
                logger.warning(f"Error calculating MACD: {e}")
                df['MACD'] = 0
                df['MACD_Signal'] = 0
                df['MACD_Hist'] = 0
            
            # Relative Strength Index (RSI)
            try:
                delta = df['Close'].diff().fillna(0)
                gain = delta.where(delta > 0, 0).fillna(0)
                loss = -delta.where(delta < 0, 0).fillna(0)
                avg_gain = gain.rolling(window=14).mean().fillna(0)
                avg_loss = loss.rolling(window=14).mean().fillna(0)
                rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
                df['RSI'] = 100 - (100 / (1 + rs))
                df['RSI'] = df['RSI'].fillna(50)  # Neutral RSI when not enough data
            except Exception as e:
                logger.warning(f"Error calculating RSI: {e}")
                df['RSI'] = 50  # Neutral RSI
            
            # Bollinger Bands
            try:
                df['UpperBand'] = df['MA20'] + (df['Close'].rolling(window=20).std().fillna(0) * 2)
                df['LowerBand'] = df['MA20'] - (df['Close'].rolling(window=20).std().fillna(0) * 2)
                df['BBWidth'] = np.where(df['MA20'] != 0, (df['UpperBand'] - df['LowerBand']) / df['MA20'], 0)
            except Exception as e:
                logger.warning(f"Error calculating Bollinger Bands: {e}")
                df['UpperBand'] = df['Close'] * 1.05  # 5% above close
                df['LowerBand'] = df['Close'] * 0.95  # 5% below close
                df['BBWidth'] = 0.1  # Default width
            
            # Stochastic Oscillator
            try:
                low_min = df['Low'].rolling(window=14).min().fillna(df['Low'])
                high_max = df['High'].rolling(window=14).max().fillna(df['High'])
                denominator = high_max - low_min
                df['%K'] = np.where(denominator != 0, 
                                    100 * ((df['Close'] - low_min) / denominator), 
                                    50)
                df['%D'] = df['%K'].rolling(window=3).mean().fillna(50)
            except Exception as e:
                logger.warning(f"Error calculating Stochastic Oscillator: {e}")
                df['%K'] = 50
                df['%D'] = 50
            
            # Average True Range (ATR)
            try:
                high_low = df['High'] - df['Low']
                high_close = (df['High'] - df['Close'].shift()).abs().fillna(0)
                low_close = (df['Low'] - df['Close'].shift()).abs().fillna(0)
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df['ATR'] = true_range.rolling(window=14).mean().fillna(true_range)
            except Exception as e:
                logger.warning(f"Error calculating ATR: {e}")
                df['ATR'] = df['Close'] * 0.02  # Default 2% of price
            
            # On-Balance Volume (OBV)
            try:
                df['OBV'] = (np.sign(df['Close'].diff().fillna(0)) * df['Volume']).fillna(0).cumsum()
            except Exception as e:
                logger.warning(f"Error calculating OBV: {e}")
                df['OBV'] = 0
            
            # Market relative performance
            try:
                if 'SP500_Close' in df.columns:
                    df['RelPerf'] = np.where(df['SP500_Close'] != 0, 
                                           df['Close'] / df['SP500_Close'], 
                                           1.0)
                    df['RelPerfChange'] = df['RelPerf'].pct_change().fillna(0)
                else:
                    logger.warning("SP500_Close not available for relative performance calculations")
                    df['RelPerf'] = 1.0
                    df['RelPerfChange'] = 0
            except Exception as e:
                logger.warning(f"Error calculating market relative performance: {e}")
                df['RelPerf'] = 1.0
                df['RelPerfChange'] = 0
            
            # Final check for NaN values
            df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
            
            # Verify all columns have sensible values
            for col in df.columns:
                # Replace inf or -inf with reasonable values
                df[col] = np.where(np.isinf(df[col]), 0, df[col])
                
                # If a column somehow still has NaN, set to 0
                if df[col].isna().any():
                    logger.warning(f"Column {col} still has NaN values after processing. Filling with 0.")
                    df[col] = df[col].fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            # Return the original data with minimal additions if we encountered an error
            try:
                data['Returns'] = 0
                data['MA5'] = data['Close']
                data['MA20'] = data['Close']
                data['RSI'] = 50
                data['MACD'] = 0
                data['BBWidth'] = 0.1
                data['%K'] = 50
                data['%D'] = 50
                data['ATR'] = data['Close'] * 0.02
                data['OBV'] = 0
                data['RelPerf'] = 1.0
                return data
            except:
                logger.error("Could not even add minimal indicators. Returning original data.")
                return data

    def prepare_data(self, data):
        """Prepare and scale the data for training with multiple features"""
        try:
            # Add technical indicators
            processed_data = self.add_technical_indicators(data)
            logger.info(f"Technical indicators added. Shape: {processed_data.shape}")
            
            # Create price target (what we want to predict)
            price_data = processed_data['Close'].values.reshape(-1, 1)
            scaled_price = self.scaler_price.fit_transform(price_data)
            
            # Extract features (exclude the target 'Close' and date column)
            features_list = [
                'Open', 'High', 'Low', 'Volume', 
                'Returns', 'MA5', 'MA20', 'RSI', 'MACD', 
                'BBWidth', '%K', '%D', 'ATR', 'OBV',
                'SP500_Close', 'RelPerf'
            ]
            
            # Ensure all selected features exist in the dataframe
            available_features = [col for col in features_list if col in processed_data.columns]
            
            if not available_features:
                logger.warning("No technical features available. Creating basic features.")
                # Create basic features if none are available
                processed_data['Returns'] = 0
                processed_data['MA5'] = processed_data['Close']
                processed_data['RSI'] = 50
                available_features = ['Returns', 'MA5', 'RSI']
            
            logger.info(f"Using {len(available_features)} features: {available_features}")
            
            # Get feature data
            try:
                features = processed_data[available_features].values
                # Check for and handle any remaining NaN values
                if np.isnan(features).any():
                    logger.warning("NaN values found in features. Filling with zeros.")
                    features = np.nan_to_num(features, nan=0.0)
                
                # Check for infinite values
                if np.isinf(features).any():
                    logger.warning("Infinite values found in features. Replacing with zeros.")
                    features = np.nan_to_num(features, posinf=0.0, neginf=0.0)
                
                scaled_features = self.scaler_features.fit_transform(features)
            except Exception as e:
                logger.error(f"Error scaling features: {e}")
                # Fallback to simple features
                logger.warning("Using simple fallback features due to scaling error.")
                features = np.ones((len(processed_data), 3))  # Simple dummy features
                scaled_features = features
            
            # Prepare sequences for time series data
            x_price, x_features, y = [], [], []
            
            # Ensure we have enough data points
            if len(scaled_price) <= self.prediction_days:
                logger.warning(f"Not enough data points ({len(scaled_price)}) for prediction window ({self.prediction_days}).")
                # Fallback: create a few dummy sequences
                dummy_seq_count = max(10, len(scaled_price) // 2)
                x_price = np.zeros((dummy_seq_count, self.prediction_days, 1))
                x_features = np.zeros((dummy_seq_count, min(self.feature_days, self.prediction_days), scaled_features.shape[1]))
                y = np.zeros(dummy_seq_count)
                
                logger.info(f"Created {dummy_seq_count} dummy training sequences as fallback.")
                
                # Save scalers for later use
                scalers = {
                    'price': self.scaler_price,
                    'features': self.scaler_features
                }
                joblib.dump(scalers, self.scaler_path)
                
                return x_price, x_features, y
            
            # Normal sequence creation when we have enough data
            try:
                for i in range(self.prediction_days, len(scaled_price)):
                    # Price sequence data (last N days of price)
                    x_price.append(scaled_price[i-self.prediction_days:i, 0])
                    
                    # Feature sequence data (last feature_days of all features)
                    feature_window = min(self.feature_days, self.prediction_days)
                    x_features.append(scaled_features[i-feature_window:i])
                    
                    # Target (next day's close price)
                    y.append(scaled_price[i, 0])
                
                # Convert to numpy arrays
                x_price = np.array(x_price)
                x_features = np.array(x_features)
                y = np.array(y)
                
                # Handle edge case of empty sequences
                if len(x_price) == 0 or len(x_features) == 0 or len(y) == 0:
                    raise ValueError("Empty training sequences generated")
                
                # Reshape for LSTM/CNN input
                x_price = np.reshape(x_price, (x_price.shape[0], x_price.shape[1], 1))
                
                logger.info(f"Successfully created {len(x_price)} training sequences.")
                logger.info(f"Shapes - X price: {x_price.shape}, X features: {x_features.shape}, Y: {y.shape}")
                
                # Save scalers for later use
                scalers = {
                    'price': self.scaler_price,
                    'features': self.scaler_features
                }
                joblib.dump(scalers, self.scaler_path)
                
                return x_price, x_features, y
                
            except Exception as e:
                logger.error(f"Error creating sequences: {e}")
                # Fallback: create a few dummy sequences
                dummy_seq_count = 10
                x_price = np.zeros((dummy_seq_count, self.prediction_days, 1))
                x_features = np.zeros((dummy_seq_count, min(self.feature_days, self.prediction_days), scaled_features.shape[1]))
                y = np.zeros(dummy_seq_count)
                
                logger.warning(f"Using {dummy_seq_count} dummy sequences due to sequence creation error.")
                
                # Save scalers for later use
                scalers = {
                    'price': self.scaler_price,
                    'features': self.scaler_features
                }
                joblib.dump(scalers, self.scaler_path)
                
                return x_price, x_features, y
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            
            # Return dummy data as fallback
            logger.warning("Returning dummy data as fallback due to data preparation error.")
            dummy_seq_count = 10
            dummy_feature_count = 3  # Minimal feature set
            
            x_price = np.zeros((dummy_seq_count, self.prediction_days, 1))
            x_features = np.zeros((dummy_seq_count, min(self.feature_days, self.prediction_days), dummy_feature_count))
            y = np.zeros(dummy_seq_count)
            
            return x_price, x_features, y

    def build_model(self, price_input_shape, feature_input_shape=None):
        """Build an advanced hybrid model architecture"""
        try:
            # If we're trying to build a hybrid model but don't have feature shape info,
            # fall back to a simpler model
            if self.model_type == 'hybrid' and feature_input_shape is None:
                logger.warning("Hybrid model requested but no feature shape provided. Falling back to LSTM model.")
                self.model_type = 'lstm'
            
            if self.model_type == 'lstm':
                # Basic LSTM model with improvements
                model = Sequential([
                    LSTM(units=100, return_sequences=True, input_shape=price_input_shape),
                    BatchNormalization(),
                    Dropout(0.3),
                    LSTM(units=100, return_sequences=True),
                    BatchNormalization(),
                    Dropout(0.3),
                    LSTM(units=100),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(units=50, activation='relu'),
                    Dense(units=1)
                ])
                
            elif self.model_type == 'bilstm':
                # Bidirectional LSTM
                model = Sequential([
                    Bidirectional(LSTM(units=100, return_sequences=True), input_shape=price_input_shape),
                    BatchNormalization(),
                    Dropout(0.3),
                    Bidirectional(LSTM(units=100)),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(units=50, activation='relu'),
                    Dense(units=1)
                ])
                
            elif self.model_type == 'cnn_lstm':
                # CNN-LSTM model
                model = Sequential([
                    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=price_input_shape),
                    MaxPooling1D(pool_size=2),
                    Conv1D(filters=128, kernel_size=3, activation='relu'),
                    BatchNormalization(),
                    LSTM(units=100, return_sequences=True),
                    Dropout(0.3),
                    LSTM(units=100),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(units=50, activation='relu'),
                    Dense(units=1)
                ])
                
            elif self.model_type == 'hybrid' and feature_input_shape is not None:
                # Verify both input shapes are valid
                if price_input_shape is None or len(price_input_shape) != 2:
                    raise ValueError(f"Invalid price input shape: {price_input_shape}")
                if feature_input_shape is None or len(feature_input_shape) != 2:
                    raise ValueError(f"Invalid feature input shape: {feature_input_shape}")
                    
                # Hybrid model with separate inputs for price and technical indicators
                try:
                    # Price sequence input branch
                    price_input = Input(shape=price_input_shape)
                    price_conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(price_input)
                    price_pool = MaxPooling1D(pool_size=2)(price_conv1)
                    price_conv2 = Conv1D(filters=128, kernel_size=3, activation='relu')(price_pool)
                    price_bn = BatchNormalization()(price_conv2)
                    price_lstm = LSTM(units=100, return_sequences=True)(price_bn)
                    price_drop = Dropout(0.3)(price_lstm)
                    price_lstm2 = LSTM(units=100)(price_drop)
                    price_branch = BatchNormalization()(price_lstm2)
                    
                    # Technical indicators input branch
                    features_input = Input(shape=feature_input_shape)
                    features_lstm = Bidirectional(LSTM(units=100, return_sequences=True))(features_input)
                    features_drop = Dropout(0.3)(features_lstm)
                    features_lstm2 = Bidirectional(LSTM(units=100))(features_drop)
                    features_branch = BatchNormalization()(features_lstm2)
                    
                    # Merge branches
                    merged = Concatenate()([price_branch, features_branch])
                    dense1 = Dense(units=100, activation='relu')(merged)
                    bn = BatchNormalization()(dense1)
                    drop = Dropout(0.3)(bn)
                    dense2 = Dense(units=50, activation='relu')(drop)
                    output = Dense(units=1)(dense2)
                    
                    # Create model
                    model = Model(inputs=[price_input, features_input], outputs=output)
                except Exception as e:
                    logger.error(f"Error building hybrid model: {e}. Falling back to GRU model.")
                    # If hybrid model fails, fall back to GRU
                    self.model_type = 'gru'
                    model = Sequential([
                        GRU(units=100, return_sequences=True, input_shape=price_input_shape),
                        BatchNormalization(),
                        Dropout(0.3),
                        GRU(units=100),
                        BatchNormalization(),
                        Dropout(0.3),
                        Dense(units=50, activation='relu'),
                        Dense(units=1)
                    ])
            
            else:
                # Default model (GRU-based)
                self.model_type = 'gru'  # Set the model type explicitly
                model = Sequential([
                    GRU(units=100, return_sequences=True, input_shape=price_input_shape),
                    BatchNormalization(),
                    Dropout(0.3),
                    GRU(units=100),
                    BatchNormalization(),
                    Dropout(0.3),
                    Dense(units=50, activation='relu'),
                    Dense(units=1)
                ])
            
            # Compile model with better optimizer settings
            try:
                optimizer = Adam(learning_rate=0.001)
                model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=['mae', 'mse'])
            except Exception as e:
                logger.error(f"Error compiling model with Adam optimizer: {e}. Using default optimizer.")
                # Fall back to default optimizer if Adam fails
                model.compile(optimizer='adam', loss='mean_squared_error')
            
            return model
        except Exception as e:
            logger.error(f"Error building model: {e}")
            # Return a very basic model as fallback
            logger.warning("Creating a simple fallback model due to build error.")
            basic_model = Sequential([
                Dense(10, input_shape=(price_input_shape[0],)),
                Dense(1)
            ])
            basic_model.compile(optimizer='adam', loss='mean_squared_error')
            return basic_model

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
            
            model_checkpoint = ModelCheckpoint(
                filepath=self.model_path,
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            )
            
            callbacks = [early_stopping, reduce_lr, model_checkpoint]
            
            # Train model based on architecture type
            if self.model_type == 'hybrid':
                history = self.model.fit(
                    [x_price, x_features], y,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    callbacks=callbacks,
                    verbose=1
                )
            else:
                history = self.model.fit(
                    x_price, y,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    callbacks=callbacks,
                    verbose=1
                )
                
            # Plot training history
            self.plot_training_history(history)
            
            return history
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
            
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
        """Make predictions using the trained model with both price and feature data"""
        try:
            logger.info(f"Price data shape: {price_data.shape}")
            
            # Load scalers if not already in memory
            if not hasattr(self, 'scaler_price') or not hasattr(self, 'scaler_features'):
                try:
                    scalers = joblib.load(self.scaler_path)
                    self.scaler_price = scalers['price']
                    self.scaler_features = scalers['features']
                except Exception as e:
                    logger.warning(f"Scalers not found or could not be loaded: {e}. Using the current ones.")
            
            # Check if input data is sufficient 
            if len(price_data) < self.prediction_days:
                logger.warning(f"Not enough price data for prediction. Need at least {self.prediction_days} points.")
                return np.array([])
            
            # Prepare price data
            price_inputs = price_data.reshape(-1, 1)
            scaled_price = self.scaler_price.transform(price_inputs)
            
            # Prepare feature data if available
            if feature_data is not None and self.model_type == 'hybrid':
                try:
                    scaled_features = self.scaler_features.transform(feature_data)
                except Exception as e:
                    logger.warning(f"Error scaling features: {e}. Falling back to price-only model.")
                    feature_data = None
                    self.model_type = 'lstm'  # Fallback to simple model
            else:
                feature_data = None
                if self.model_type == 'hybrid':
                    logger.warning("No feature data provided for hybrid model. Using price-only prediction.")
            
            # Create sequences
            x_price, x_features = [], []
            
            # Handle edge case with short data
            max_sequences = max(0, len(scaled_price) - self.prediction_days)
            
            for i in range(self.prediction_days, len(scaled_price)):
                # Price sequence
                x_price.append(scaled_price[i-self.prediction_days:i, 0])
                
                # Feature sequence (if available)
                if feature_data is not None:
                    feature_window = min(self.feature_days, self.prediction_days)
                    # Handle edge case with short feature data
                    if i >= feature_window and i < len(feature_data) + self.prediction_days:
                        feature_idx = min(i, len(feature_data)) - feature_window
                        feature_end_idx = min(i, len(feature_data))
                        x_features.append(scaled_features[feature_idx:feature_end_idx])
            
            if not x_price:
                logger.warning("No valid prediction sequences could be created.")
                return np.array([])
                
            x_price = np.array(x_price)
            x_price = np.reshape(x_price, (x_price.shape[0], x_price.shape[1], 1))
            
            if feature_data is not None and x_features:
                try:
                    x_features = np.array(x_features)
                    # Check if all feature sequences have the expected shape
                    if len(x_features) != len(x_price):
                        logger.warning(f"Feature sequence count ({len(x_features)}) doesn't match price sequences ({len(x_price)}). Using price-only model.")
                        feature_data = None
                except Exception as e:
                    logger.warning(f"Error preparing feature sequences: {e}. Using price-only model.")
                    feature_data = None
            else:
                feature_data = None
            
            # Make predictions based on model type
            try:
                if self.model_type == 'hybrid' and feature_data is not None and len(x_features) == len(x_price):
                    predicted_scaled = self.model.predict([x_price, x_features], verbose=0)
                else:
                    predicted_scaled = self.model.predict(x_price, verbose=0)
                
                # Convert predictions back to original scale
                predicted_prices = self.scaler_price.inverse_transform(predicted_scaled)
                return predicted_prices
            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                return np.array([])
                
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return np.array([])

    def predict_next_day(self, price_data, feature_data=None):
        """Predict the next day's price using the trained model"""
        try:
            # Ensure price data is 1-dimensional
            if len(price_data.shape) > 1:
                price_data = price_data.flatten()
            
            # Check if there's enough data
            if len(price_data) < self.prediction_days:
                logger.warning(f"Not enough price data for next-day prediction. Need at least {self.prediction_days} points.")
                return None
            
            # Get the last prediction_days values for price
            last_price_sequence = price_data[-self.prediction_days:].reshape(-1, 1)
            
            try:
                scaled_price = self.scaler_price.transform(last_price_sequence)
                x_price = scaled_price.reshape(1, self.prediction_days, 1)
            except Exception as e:
                logger.error(f"Error scaling price data for next-day prediction: {e}")
                return None
            
            # Get feature data if available
            if feature_data is not None and self.model_type == 'hybrid':
                try:
                    feature_window = min(self.feature_days, self.prediction_days)
                    if len(feature_data) < feature_window:
                        logger.warning(f"Not enough feature data for prediction. Need at least {feature_window} points.")
                        # Fall back to price-only prediction
                        prediction_scaled = self.model.predict(x_price, verbose=0)
                    else:
                        last_feature_sequence = feature_data[-feature_window:]
                        # Check for NaN or infinite values
                        if np.isnan(last_feature_sequence).any() or np.isinf(last_feature_sequence).any():
                            last_feature_sequence = np.nan_to_num(last_feature_sequence, nan=0.0, posinf=0.0, neginf=0.0)
                            
                        scaled_features = self.scaler_features.transform(last_feature_sequence)
                        x_features = scaled_features.reshape(1, feature_window, scaled_features.shape[1])
                        
                        # Make prediction with both inputs
                        prediction_scaled = self.model.predict([x_price, x_features], verbose=0)
                except Exception as e:
                    logger.error(f"Error processing feature data for next-day prediction: {e}")
                    # Fall back to price-only prediction
                    prediction_scaled = self.model.predict(x_price, verbose=0)
            else:
                # Make prediction with just price data
                prediction_scaled = self.model.predict(x_price, verbose=0)
            
            # Convert back to original scale
            try:
                prediction = self.scaler_price.inverse_transform(prediction_scaled)[0][0]
                logger.info(f"Final prediction for next day: ${prediction:.2f}")
                return prediction
            except Exception as e:
                logger.error(f"Error converting prediction back to original scale: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error predicting next day: {e}")
            return None

    def predict_future(self, price_data, feature_data=None, future_days=60):
        """
        Predict future prices using an iterative approach with advanced features
        """
        try:
            # Ensure price data is 1-dimensional
            if len(price_data.shape) > 1:
                price_data = price_data.flatten()
            
            # Check if there's enough data
            if len(price_data) < self.prediction_days:
                logger.warning(f"Not enough price data for future prediction. Need at least {self.prediction_days} points.")
                return np.array([])
                
            # Get the last prediction_days values for price data
            last_price_sequence = price_data[-self.prediction_days:].reshape(-1, 1)
            
            try:
                scaled_price = self.scaler_price.transform(last_price_sequence)
                current_price_sequence = list(scaled_price.flatten())
            except Exception as e:
                logger.error(f"Error scaling price data for future prediction: {e}")
                return np.array([])
            
            # Get feature data if hybrid model
            use_features = False
            feature_window = min(self.feature_days, self.prediction_days)
            current_feature_sequence = None
            
            if feature_data is not None and self.model_type == 'hybrid':
                try:
                    if len(feature_data) < feature_window:
                        logger.warning(f"Not enough feature data for future prediction. Falling back to price-only.")
                    else:
                        last_feature_sequence = feature_data[-feature_window:]
                        # Check for NaN or infinite values
                        if np.isnan(last_feature_sequence).any() or np.isinf(last_feature_sequence).any():
                            last_feature_sequence = np.nan_to_num(last_feature_sequence, nan=0.0, posinf=0.0, neginf=0.0)
                            
                        scaled_features = self.scaler_features.transform(last_feature_sequence)
                        current_feature_sequence = list(scaled_features)
                        use_features = True
                except Exception as e:
                    logger.error(f"Error processing feature data for future prediction: {e}")
                    use_features = False
            
            future_predictions = []
            
            for step in range(future_days):
                try:
                    # Prepare price sequence (last prediction_days values)
                    price_sequence = np.array(current_price_sequence[-self.prediction_days:])
                    price_sequence = price_sequence.reshape(1, self.prediction_days, 1)
                    
                    # Make predictions based on model type
                    if use_features and current_feature_sequence is not None:
                        # Prepare feature sequence
                        feature_sequence = np.array(current_feature_sequence[-feature_window:])
                        feature_sequence = feature_sequence.reshape(1, feature_window, feature_sequence.shape[1])
                        
                        # Make prediction with both inputs
                        next_pred_scaled = self.model.predict([price_sequence, feature_sequence], verbose=0)
                    else:
                        # Make prediction with just price
                        next_pred_scaled = self.model.predict(price_sequence, verbose=0)
                    
                    # Inverse transform to get the real price
                    next_pred = self.scaler_price.inverse_transform(next_pred_scaled)[0][0]
                    future_predictions.append(next_pred)
                    
                    # Update sequences for next iteration
                    current_price_sequence.append(next_pred_scaled[0][0])
                    
                    # For hybrid model, we'd need to generate feature values for future days
                    # This is a simple approach that keeps the last feature values
                    if use_features and current_feature_sequence is not None:
                        current_feature_sequence.append(current_feature_sequence[-1])
                        
                except Exception as e:
                    logger.error(f"Error in future prediction step {step}: {e}")
                    # If we failed in the middle, return what we have so far
                    if future_predictions:
                        return np.array(future_predictions)
                    else:
                        return np.array([])
                
            return np.array(future_predictions)
        except Exception as e:
            logger.error(f"Error predicting future: {e}")
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
    
    # Ensure we use datetime objects without timezone info to avoid conflicts
    current_date = dt.datetime.now().replace(tzinfo=None)
    
    # Fix: Adjust date ranges to use proper historical data
    train_start = dt.datetime(2015, 1, 1)  # Start date for training data
    train_end = current_date - dt.timedelta(days=60)  # End date for training, 60 days before now
    test_start = train_end  # Start date for test data (right after training ends)
    test_end = current_date  # End date for test data (now)
    
    logger.info(f"Date ranges: Training {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}, "
               f"Testing {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
    
    try:
        # Initialize predictor with model type
        predictor = StockPredictor(company, prediction_days, feature_days, model_type)
        
        # Fetch and prepare training data
        logger.info(f"Fetching training data for {company}...")
        train_data = predictor.fetch_data(train_start, train_end)
        
        if train_data.empty or len(train_data) < prediction_days * 2:
            raise ValueError(f"Not enough training data for {company}. Got {len(train_data)} rows, need at least {prediction_days * 2}.")
            
        logger.info(f"Training data shape: {train_data.shape}")
        logger.info(f"Training data range: {train_data.index.min()} to {train_data.index.max()}")
        logger.info(f"Training price range: {train_data['Close'].min():.2f} to {train_data['Close'].max():.2f}")
        
        # Prepare data for training
        x_price_train, x_features_train, y_train = predictor.prepare_data(train_data)
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
        test_data = predictor.fetch_data(test_start, test_end)
        
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
            if model_type == 'hybrid' and test_feature_data is not None:
                future_predictions = predictor.predict_future(price_inputs, test_feature_data, future_days=60)
            else:
                future_predictions = predictor.predict_future(price_inputs, future_days=60)
                
            if len(future_predictions) == 0:
                logger.warning("Failed to generate future predictions.")
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
    # Configure logging to show more detailed information during debug
    import sys
    
    # Check for debug flag
    debug_mode = False
    args = sys.argv[1:]
    
    if "--debug" in args:
        debug_mode = True
        args.remove("--debug")
        logging.basicConfig(level=logging.DEBUG, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    try:
        # Parse command line arguments for stock symbol
        stock_symbol = "NVDA"  # Default
        model_type = "hybrid"  # Default
        
        if len(args) > 0:
            stock_symbol = args[0]
            
        if len(args) > 1:
            model_type = args[1]
        
        logger.info(f"Running prediction for {stock_symbol} using {model_type} model")
        
        # Run with hybrid model for better performance
        results = main(stock_symbol, model_type=model_type, prediction_days=60, feature_days=30)
        
        if results:
            # Show the next day prediction
            if results['next_day'] is not None:
                print(f"\n=== {stock_symbol} Prediction Summary ===")
                print(f"Next trading day prediction: ${results['next_day']:.2f}")
                
                # Show basic metrics if available
                if results['metrics']:
                    print(f"\nModel metrics:")
                    print(f"  RMSE: {results['metrics']['rmse']:.4f}")
                    print(f"  Directional Accuracy: {results['metrics']['directional_accuracy']:.2%}")
                    
                # Show a simple forecast summary
                if len(results['future']) > 0:
                    print(f"\nForecast summary (next 60 days):")
                    print(f"  Min: ${min(results['future']):.2f}")
                    print(f"  Max: ${max(results['future']):.2f}")
                    print(f"  Avg: ${np.mean(results['future']):.2f}")
                    
                    # Calculate trend
                    if results['future'][-1] > results['future'][0]:
                        trend_pct = (results['future'][-1] - results['future'][0]) / results['future'][0] * 100
                        print(f"  Trend: UP {trend_pct:.1f}%")
                    else:
                        trend_pct = (results['future'][0] - results['future'][-1]) / results['future'][0] * 100
                        print(f"  Trend: DOWN {trend_pct:.1f}%")
                
                print("\nPrediction images saved to models/saved/ directory")
        else:
            logger.error("Prediction failed. Check logs for details.")
    
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        
    # Print instructions for use
    print("\nUsage:")
    print("  python models/beta2.py [SYMBOL] [MODEL_TYPE] [--debug]")
    print("  Example: python models/beta2.py AAPL lstm")
    print("  Available model types: lstm, bilstm, cnn_lstm, hybrid, gru")
