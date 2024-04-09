from enhance import *
import joblib

# Define the stock symbols and date ranges for historical data
amd_symbol = 'AMD'
aapl_symbol = 'AAPL'
start_date = '2010-01-01'
end_date = '2022-12-31'

# Download historical stock price data for both AMD and AAPL
amd_data = yf.download(amd_symbol, start=start_date, end=end_date)
aapl_data = yf.download(aapl_symbol, start=start_date, end=end_date)

# Extract the 'Close' prices for both stocks
amd_closing_prices = amd_data['Close'].values.reshape(-1, 1)
aapl_closing_prices = aapl_data['Close'].values.reshape(-1, 1)

# Normalize the closing prices to the range [0, 1] for both stocks
scaler = MinMaxScaler()
amd_closing_prices_scaled = scaler.fit_transform(amd_closing_prices)
aapl_closing_prices_scaled = scaler.transform(aapl_closing_prices)

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

# Create input sequences and labels for both stocks
X_amd, y_amd = create_sequences(amd_closing_prices_scaled, sequence_length)
X_aapl, y_aapl = create_sequences(aapl_closing_prices_scaled, sequence_length)

# Split the data into training and testing sets for both stocks
split_ratio = 0.8
split_index = int(split_ratio * len(X_amd))
X_amd_train, X_amd_test = X_amd[:split_index], X_amd[split_index:]
y_amd_train, y_amd_test = y_amd[:split_index], y_amd[split_index:]
X_aapl_train, X_aapl_test = X_aapl[:split_index], X_aapl[split_index:]
y_aapl_train, y_aapl_test = y_aapl[:split_index], y_aapl[split_index:]

# Load the pretrained model
model = load_model('prediction_ai_model.h5')

# Fine-tune the model
model.add(LSTM(64, activation='relu'))  # Adjust architecture as needed
model.add(Dense(1))
model.compile(optimizer=LegacyAdam(learning_rate=0.001), loss='mean_squared_error')

# Train the model on the combined dataset (including both AMD and AAPL data)
X_combined_train = np.concatenate((X_amd_train, X_aapl_train), axis=0)
y_combined_train = np.concatenate((y_amd_train, y_aapl_train), axis=0)

model.fit(X_combined_train, y_combined_train, epochs=epochs, batch_size=batch_size, verbose=1)

# Evaluate the model on a test dataset containing AAPL data
mse = model.evaluate(X_aapl_test, y_aapl_test)
print(f'Mean Squared Error on AAPL Test Data: {mse}')

# Save the updated model
model.save('prediction_ai_model_updated.h5')