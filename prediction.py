# predict_ai_model.py
# Code for loading the trained model and making predictions
from training import stock_data, scaler, closing_prices_scaled, sequence_length, stock_symbol
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Load the pretrained model
model = load_model('base_AI.keras')

# Predict future closing prices
future_days = 180  # Number of days into the future to predict
future_predictions = []
last_sequence = closing_prices_scaled[-sequence_length:].reshape(1, sequence_length, 1)

for _ in range(future_days):
    predicted_value = model.predict(last_sequence)[0][0]
    future_predictions.append(predicted_value)
    last_sequence = np.roll(last_sequence, -1, axis=1)
    last_sequence[0][-1] = predicted_value

# Inverse transform the predictions to the original scale
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Generate future dates for plotting
last_date = stock_data.index[-1]
future_dates = pd.date_range(start=last_date, periods=future_days + 1)

# Plot historical and predicted closing prices
plt.figure(figsize=(12, 6))
plt.plot(stock_data.index, stock_data['Close'], label='Historical Close Prices', color='blue')
plt.plot(future_dates[1:], future_predictions, label='Predicted Close Prices', color='red', linestyle='--')
plt.title(f'Stock Price Prediction for {stock_symbol} (Future Predictions)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()