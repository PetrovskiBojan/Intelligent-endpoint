import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'])  # Ensure 'time' column is datetime format
    df.sort_values(by='time', inplace=True)
    
    # Assuming 'temperature_2m' and 'precipitation_probability' are your features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(df[['temperature_2m', 'precipitation_probability']])
    
    X = scaled_features
    y = df['available_bike_stands'].values.reshape(-1, 1)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_and_train_model(X_train, y_train, X_test, y_test, model_name, model_dir):
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    
    model = Sequential([
        LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=2)
    
    model.save(os.path.join(model_dir, f'{model_name}.h5'))

# Determine the correct path to the 'processed' directory
current_dir = os.path.dirname(__file__)  # Gets the directory where the script is located
project_dir = os.path.join(current_dir, '..', 'data', 'processed')  # Adjusts the path to reach 'data/processed' from the script's location
processed_dir = os.path.abspath(project_dir)

# Ensure the 'models' directory exists
models_dir = os.path.join(current_dir, '..', 'models')
os.makedirs(models_dir, exist_ok=True)

# Process each CSV file in the 'processed' directory
for filename in os.listdir(processed_dir):
    if filename.endswith(".csv"):
        print(f"Processing {filename}...")
        csv_path = os.path.join(processed_dir, filename)
        X_train, X_test, y_train, y_test = load_and_preprocess_data(csv_path)
        model_name = filename.replace('.csv', '')
        build_and_train_model(X_train, y_train, X_test, y_test, model_name, models_dir)
        print(f"Model for {model_name} saved.")
    else:
        continue
