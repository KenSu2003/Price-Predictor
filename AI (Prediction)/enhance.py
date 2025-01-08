from training import *
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


# Define the stock symbol and date range for historical data
stock_symbol = 'MSFT'
start_date = '2010-01-01'
end_date = '2022-12-31'

# Load AAPL data
aapl_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Assuming the same preprocessing as in training.py
# Adjust as per your original training script
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(aapl_data['Close'].values.reshape(-1,1))

# Creating the dataset in the same time series format
x_train = []
y_train = []

time_frame = 30  # Assuming the same time frame used in training.py

for i in range(time_frame, len(scaled_data)):
    x_train.append(scaled_data[i-time_frame:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Load your existing model
# model = load_model('prediction_AI.keras')
model = load_model('enhanced_AI_2.keras')

# Train the model with AAPL data
model.fit(x_train, y_train, epochs=25, batch_size=32)  # Adjust epochs and batch size as needed

# Save the updated model
model.save('enhanced_AI_2_2.keras')