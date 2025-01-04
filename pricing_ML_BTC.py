import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

def load_data(ticker, start_date, interval):
    """
    Load cryptocurrency data using yfinance and preprocess it
    """
    data = yf.download(ticker, start=start_date, interval=interval)
    data = data[['Close']]
    return data

def preprocess_data(data, lookback=120):
    """
    Normalize the data and create sequences for LSTM
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

def build_model(input_shape):
    """
    Build and compile an LSTM model
    """
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs=20, batch_size=32):
    """
    Train the LSTM model
    """
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return history

def evaluate_model(model, X_test, y_test, scaler):
    """
    Evaluate the model and plot predictions vs actual values
    """
    test_loss = model.evaluate(X_test, y_test, verbose=1)
    print(f"Mean Squared Error on Test Data: {test_loss}")

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    plt.figure(figsize=(10, 6))
    plt.plot(y_test_actual, color='blue', label='Actual Prices')
    plt.plot(predictions, color='red', label='Predicted Prices')
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def save_model(model, filename="lstm_price_prediction_model_BTC.h5"):
    """
    Save the trained model to a file
    """
    model.save(filename)
    print(f"Model saved as {filename}")

if __name__ == "__main__":
    # Parameters
    ticker = "BTC-USD"
    start_date = "2024-11-06"
    interval = "15m"
    lookback = 120

    # Load and preprocess data
    data = load_data(ticker, start_date, interval)
    X, y, scaler = preprocess_data(data, lookback=lookback)

    # Split data into training and testing sets
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build and train the model
    model = build_model(input_shape=(X_train.shape[1], 1))
    train_model(model, X_train, y_train, epochs=20, batch_size=32)

    # Evaluate the model
    evaluate_model(model, X_test, y_test, scaler)

    # Save the model
    save_model(model)
