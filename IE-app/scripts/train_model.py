import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
import joblib

# Function to create sequences for RNN
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data)-1:
            break
        seq_x, seq_y = data[i:end_ix, :-1], data[end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Function to train and save an RNN model for each station
def train_and_save_models(processed_data_dir='data/processed', n_steps=3):
    stations = os.listdir(processed_data_dir)
    scaler = MinMaxScaler()
    
    for station_file in stations:
        station_name = station_file.split('_')[0]
        df = pd.read_csv(os.path.join(processed_data_dir, station_file))
        
        # Assuming 'available_bike_stands' is the target and all other columns are features
        # Normalize features
        scaled_data = scaler.fit_transform(df)
        joblib.dump(scaler, f'models/{station_name}_scaler.pkl') # Save scaler for later use
        
        X, y = create_sequences(scaled_data, n_steps)
        
        # Define RNN model
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(n_steps, X.shape[2])),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Fit model
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X, y, epochs=100, validation_split=0.2, batch_size=64, callbacks=[es], verbose=0)
        
        # Save model
        model.save(f'models/{station_name}_model')
        
        print(f'Model trained and saved for station: {station_name}')

if __name__ == "__main__":
    train_and_save_models()
