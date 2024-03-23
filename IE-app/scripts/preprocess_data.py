import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(csv_path, processed_dir):
    # Load the data
    df = pd.read_csv(csv_path, parse_dates=['time'])
    
    # Simplify the 'time' to remove seconds and milliseconds
    df['time'] = df['time'].dt.floor('Min')
    
    # Normalize 'temperature_2m' and 'precipitation_probability' to range [0, 1]
    scaler = MinMaxScaler()
    df[['temperature_2m', 'precipitation_probability']] = scaler.fit_transform(df[['temperature_2m', 'precipitation_probability']])
    
    # Save the preprocessed data to the 'processed' directory
    processed_path = os.path.join(processed_dir, os.path.basename(csv_path))
    df.to_csv(processed_path, index=False)
    print(f"Preprocessed data saved to {processed_path}")

if __name__ == "__main__":
    # Specify the directories
    current_dir = os.path.dirname(__file__)
    combined_dir = os.path.abspath(os.path.join(current_dir, '..', 'data', 'combined'))
    processed_dir = os.path.abspath(os.path.join(current_dir, '..', 'data', 'processed'))
    
    # Ensure the 'processed' directory exists
    os.makedirs(processed_dir, exist_ok=True)
    
    # Process each CSV file in the 'combined' directory
    for filename in os.listdir(combined_dir):
        if filename.endswith(".csv"):
            print(f"Processing {filename}...")
            csv_path = os.path.join(combined_dir, filename)
            preprocess_data(csv_path, processed_dir)
