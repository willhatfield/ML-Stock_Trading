import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self, company, prediction_days=60):
        self.company = company
        self.prediction_days = prediction_days
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
    def fetch_data(self, start_date, end_date):
        """Fetch stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(self.company)
            data = ticker.history(start=start_date, end=end_date)
            if data.empty:
                raise ValueError(f"No data found for {self.company}")
            return data
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise

    def prepare_data(self, data):
        """Prepare and scale the data for training"""
        try:
            scaled_data = self.scaler.fit_transform(data['Close'].values.reshape(-1, 1))
            
            x_train, y_train = [], []
            for x in range(self.prediction_days, len(scaled_data)):
                x_train.append(scaled_data[x-self.prediction_days:x, 0])
                y_train.append(scaled_data[x, 0])
            
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            
            return x_train, y_train
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise

    def build_model(self, input_shape):
        """Build and compile the LSTM model"""
        try:
            model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(units=50, return_sequences=True),
                Dropout(0.2),
                LSTM(units=50),
                Dropout(0.2),
                Dense(units=1)
            ])
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            return model
        except Exception as e:
            logger.error(f"Error building model: {e}")
            raise

    def train_model(self, x_train, y_train, epochs=25, batch_size=40):
        """Train the model with early stopping"""
        try:
            early_stopping = EarlyStopping(
                monitor='loss',
                patience=5,
                restore_best_weights=True
            )
            
            self.model.fit(
                x_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                validation_split=0.1,
                verbose=1
            )
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

    def make_predictions(self, data):
        """Make predictions on the data"""
        try:
            logger.info(f"Input data shape: {data.shape}")
            logger.info(f"Input data range: {data.min():.2f} to {data.max():.2f}")
            
            model_inputs = data.reshape(-1, 1)
            model_inputs = self.scaler.transform(model_inputs)
            
            logger.info(f"Scaled data range: {model_inputs.min():.2f} to {model_inputs.max():.2f}")
            
            x_test = []
            for x in range(self.prediction_days, len(model_inputs)):
                x_test.append(model_inputs[x-self.prediction_days:x, 0])
            
            x_test = np.array(x_test)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
            
            predicted_prices = self.model.predict(x_test)
            logger.info(f"Raw predictions range: {predicted_prices.min():.2f} to {predicted_prices.max():.2f}")
            
            predicted_prices = self.scaler.inverse_transform(predicted_prices)
            logger.info(f"Final predictions range: {predicted_prices.min():.2f} to {predicted_prices.max():.2f}")
            
            return predicted_prices
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise

    def predict_next_day(self, data):
        """Predict the next day's price"""
        try:
            # Ensure data is 1-dimensional
            if len(data.shape) > 1:
                data = data.flatten()
            
            logger.info(f"Input data for next day prediction: {data[-5:]}")  # Show last 5 values
            
            # Get the last prediction_days values
            real_data = data[-self.prediction_days:]
            real_data = real_data.reshape(1, self.prediction_days, 1)
            
            prediction = self.model.predict(real_data)
            logger.info(f"Raw prediction: {prediction}")
            
            prediction = self.scaler.inverse_transform(prediction)
            logger.info(f"Final prediction: {prediction[0][0]:.2f}")
            
            return prediction[0][0]
        except Exception as e:
            logger.error(f"Error predicting next day: {e}")
            raise

    def predict_future(self, data, future_days=60):
        """
        Predict the stock price for the next `future_days` using an iterative approach.
        It starts with the last `prediction_days` of actual data, then feeds each prediction
        back into the input sequence for the next day's prediction.
        """
        try:
            # Ensure data is 1-dimensional
            if len(data.shape) > 1:
                data = data.flatten()
                
            # Get the last `prediction_days` values and scale them
            last_sequence = data[-self.prediction_days:]
            last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1))
            current_sequence = list(last_sequence_scaled.flatten())
            
            future_predictions = []
            for _ in range(future_days):
                # Prepare the input shape (1, prediction_days, 1)
                sequence_array = np.array(current_sequence[-self.prediction_days:])
                sequence_array = sequence_array.reshape(1, self.prediction_days, 1)
                
                # Predict the next value
                next_pred_scaled = self.model.predict(sequence_array)
                # Inverse transform to get the real price
                next_pred = self.scaler.inverse_transform(next_pred_scaled)[0][0]
                future_predictions.append(next_pred)
                # Append the scaled predicted value to the sequence for next iteration
                current_sequence.append(next_pred_scaled[0][0])
                
            return np.array(future_predictions)
        except Exception as e:
            logger.error(f"Error predicting future: {e}")
            raise

def plot_predictions(actual_prices, predicted_prices, company):
    """Plot actual vs predicted prices from test data"""
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, color="black", label=f"Actual {company} Price")
    plt.plot(predicted_prices, color="green", label=f"Predicted {company} Price")
    plt.title(f"{company} Share Price Prediction")
    plt.xlabel("Time")
    plt.ylabel(f"{company} Share Price")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_actual_and_future(test_data, predicted_prices, future_predictions, company, prediction_days=60):
    """
    Plots:
      1) Actual test data (with real dates)
      2) Model's predicted prices for the test period
      3) A 60-day future forecast after the last actual date
    """
    import matplotlib.dates as mdates
    
    plt.figure(figsize=(12, 6))
    
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
    
    # 3) Plot the future forecast (60-day)
    if len(future_predictions) > 0:
        last_date = test_data.index[-1]
        # Generate one extra date and then slice the first so we start exactly on "the next day"
        future_dates = pd.date_range(
            start=last_date,  # Start exactly at the last date in test data
            periods=len(future_predictions) + 1,
            freq='B'  # Business days
        )[1:]  # Drop the first date to move seamlessly to "the next day"
        
        plt.plot(future_dates, future_predictions, label="Future Forecast", color="blue")
    
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
    plt.show()



''' 
def plot_future_predictions(current_price, future_predictions, company):
    """Plot current price and future 60-day predictions"""
    plt.figure(figsize=(12, 6))
    # Show current price at day 0
    plt.scatter(0, current_price, color="red", label="Current Price")
    # Plot future predictions from day 1 to future_days
    future_days = np.arange(1, len(future_predictions) + 1)
    plt.plot(future_days, future_predictions, color="blue", label="Predicted Future Price")
    plt.title(f"{company} 60-Day Future Price Forecast")
    plt.xlabel("Days from Today")
    plt.ylabel(f"{company} Price")
    plt.legend()
    plt.grid(True)
    plt.show() 
'''

def main(stock_symbol):
    # Initialize parameters
    company = stock_symbol
    train_start = dt.datetime(2010, 1, 1)
    train_end = dt.datetime(2024, 12, 31)
    test_start = dt.datetime(2025, 1, 1)
    test_end = dt.datetime.now()
    
    try:
        # Initialize predictor
        predictor = StockPredictor(company)
        
        # Fetch and prepare training data
        logger.info("Fetching training data...")
        train_data = predictor.fetch_data(train_start, train_end)
        logger.info(f"Training data range: {train_data['Close'].min():.2f} to {train_data['Close'].max():.2f}")
        
        x_train, y_train = predictor.prepare_data(train_data)
        
        # Build and train model
        logger.info("Building and training model...")
        predictor.model = predictor.build_model((x_train.shape[1], 1))
        predictor.train_model(x_train, y_train)
        
        # Fetch and prepare test data
        logger.info("Fetching test data...")
        test_data = predictor.fetch_data(test_start, test_end)
        logger.info(f"Test data range: {test_data['Close'].min():.2f} to {test_data['Close'].max():.2f}")
        actual_prices = test_data['Close'].values
        
        # Combine data for predictions (we need extra days for the prediction window)
        total_dataset = pd.concat((train_data['Close'], test_data['Close']), axis=0)
        model_inputs = total_dataset[len(total_dataset) - len(test_data) - predictor.prediction_days:].values
        
        # Make predictions on test data
        logger.info("Making predictions on test data...")
        predicted_prices = predictor.make_predictions(model_inputs)
        
        # Predict the next day's price (optional single day)
        next_day_prediction = predictor.predict_next_day(model_inputs)
        logger.info(f"Prediction for next day: ${next_day_prediction:.2f}")
        
        # Predict future 60-day prices
        future_predictions = predictor.predict_future(total_dataset.values, future_days=60)
        
        # >>> NEW: Single plot for actual + predicted + future forecast <<<
        plot_actual_and_future(
            test_data=test_data, 
            predicted_prices=predicted_prices,
            future_predictions=future_predictions,
            company=company,
            prediction_days=predictor.prediction_days
        )
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main("NVDA")
