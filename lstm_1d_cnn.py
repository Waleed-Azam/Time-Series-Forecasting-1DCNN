"""
# Acoustic Doppler Time Series Forecasting
This script implements two deep learning models (LSTM and 1D CNN) to forecast beam values from acoustic Doppler data. The models are designed to identify deviations in beam values, which may indicate the presence of fish interference. The script includes data preprocessing, sequence creation, model training, and result visualization.

## Key Features:
- **Data Preprocessing**: Handles timestamped beam data with scaling and sequence generation.
- **LSTM Model**: Captures temporal dependencies using Long Short-Term Memory networks.
- **1D CNN Model**: Utilizes convolutional layers to extract features from sequences.
- **Mini-Batch Training**: Improves training efficiency with configurable batch sizes.
- **Forecasting and Visualization**: Provides comparative plots of actual vs. predicted beam values.

## Instructions:
1. Replace `acoustic_doppler_data.csv` with your dataset file.
2. Adjust `sequence_length` and `epochs` parameters as needed.
3. Run the script and inspect the generated plots for predictions.

"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv1D, Flatten, MaxPooling1D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load and preprocess data
data = pd.read_csv('acoustic_doppler_data.csv')  # Replace with your file

# Assuming data has a 'timestamp' and 'beam_values' column
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)
beam_values = data[['beam1', 'beam2', 'beam3', 'beam4']]

# Scale data
scaler = MinMaxScaler()
beam_values_scaled = scaler.fit_transform(beam_values)

# Create sequences for time series forecasting
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

sequence_length = 30  # 30 timesteps
X, y = create_sequences(beam_values_scaled, sequence_length)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LSTM model
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(50),
    Dense(4)  # Four beams output
])

lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Define 1D CNN model
cnn_model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(4)  # Four beams output
])

cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
cnn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Forecasting
lstm_forecast = lstm_model.predict(X_test)
cnn_forecast = cnn_model.predict(X_test)

# Inverse transform predictions and actual values
lstm_forecast_original = scaler.inverse_transform(lstm_forecast)
cnn_forecast_original = scaler.inverse_transform(cnn_forecast)
y_test_original = scaler.inverse_transform(y_test)

# Plot results
def plot_results(y_true, lstm_pred, cnn_pred):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true[:, 0], label='Actual Beam1')
    plt.plot(lstm_pred[:, 0], label='LSTM Beam1 Prediction')
    plt.plot(cnn_pred[:, 0], label='CNN Beam1 Prediction')
    plt.legend()
    plt.title('Beam1 Predictions vs Actual')
    plt.xlabel('Time Steps')
    plt.ylabel('Beam Values')
    plt.show()

plot_results(y_test_original, lstm_forecast_original, cnn_forecast_original)
