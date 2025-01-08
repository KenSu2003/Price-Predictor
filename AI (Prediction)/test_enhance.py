# accuracy.py

# test_ai_model.py
# Code for testing and evaluating the AI model's performance
from training import scaler, sequence_length, create_sequences
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load the pretrained model
model = load_model('updated_AI.keras')
# model = load_model('prediction_ai_model_updated.h5')

# Define the stock symbol and date range for testing data
test_stock_symbol = 'AMD'
test_start_date = '2023-01-01'
test_end_date = '2023-12-31'

# Download historical stock price data for testing
test_stock_data = yf.download(test_stock_symbol, start=test_start_date, end=test_end_date)

# Extract the 'Close' prices
test_closing_prices = test_stock_data['Close'].values.reshape(-1, 1)

# Normalize the test closing prices using the same scaler used for training
test_closing_prices_scaled = scaler.transform(test_closing_prices)

# Create input sequences and labels for testing
X_test, y_test = create_sequences(test_closing_prices_scaled, sequence_length)

# Make predictions on the test data
test_predictions = model.predict(X_test)

# Inverse transform the test predictions and actual prices to the original scale
test_predictions = scaler.inverse_transform(test_predictions)
y_test = scaler.inverse_transform(y_test)

# Calculate Mean Squared Error (MSE) as a measure of model accuracy
mse = mean_squared_error(y_test, test_predictions)
print(f"Mean Squared Error (MSE): {mse}")

# Visualize the test predictions and actual prices
plt.figure(figsize=(12, 6))
plt.plot(test_stock_data.index, test_stock_data['Close'], label='Actual Close Prices', color='blue')
plt.plot(test_stock_data.index[sequence_length:], test_predictions, label='Predicted Close Prices', color='red', linestyle='--')
plt.title(f'Stock Price Prediction for {test_stock_symbol} (Test Data)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()