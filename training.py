import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam  # Import the legacy optimizer

# Define the stock symbol and date range for historical data
stock_symbol = 'USDT-BTC'
start_date = '2010-01-01'
end_date = '2024-02-18'

# Download historical stock price data
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Extract the 'Close' prices
closing_prices = stock_data['Close'].values.reshape(-1, 1)

# Normalize the closing prices to the range [0, 1]
scaler = MinMaxScaler()
closing_prices_scaled = scaler.fit_transform(closing_prices)

# Define a function to create input sequences and labels for training the LSTM model
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

# Define hyperparameters
sequence_length = 30  # Number of days to look back in time for each prediction
epochs = 50
batch_size = 64

# Create input sequences and labels
X, y = create_sequences(closing_prices_scaled, sequence_length)

# Split the data into training and testing sets
split_ratio = 0.8
split_index = int(split_ratio * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build and compile the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

if __name__ == "__main__":

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)

    # Save the model
    model.save('base_AI.keras')

    # Visualize training loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()