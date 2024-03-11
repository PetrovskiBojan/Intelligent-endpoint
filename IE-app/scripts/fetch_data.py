import requests
import pandas as pd
import os
from datetime import datetime

def fetch_bike_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch bike data: Status code {response.status_code}")

def fetch_weather_data(lat, lng):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lng,
        "hourly": ["temperature_2m", "precipitation_probability"],
        "timezone": "auto"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        weather_data = response.json()
        return (weather_data['hourly']['time'][0], weather_data['hourly']['temperature_2m'][0], 
                weather_data['hourly']['precipitation_probability'][0])
    else:
        raise Exception(f"Failed to fetch weather data: Status code {response.status_code}")

if __name__ == "__main__":
    bike_url = "https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b"
    bike_data = fetch_bike_data(bike_url)

    if bike_data:
        combined_dir = os.path.join('IE-app', 'data', 'combined')  # Ensure cross-platform compatibility
        os.makedirs(combined_dir, exist_ok=True)
        
        for station in bike_data:
            time_now, temperature, precipitation_probability = fetch_weather_data(station['position']['lat'], station['position']['lng'])
            combined_data = pd.DataFrame({
                'time': [datetime.now()],
                'temperature_2m': [temperature],
                'precipitation_probability': [precipitation_probability],
                'available_bike_stands': [station['available_bike_stands']]
            })
            
            filename = f"{station['number']}_{station['name'].replace(' ', '_').replace(',', '').replace('-', '_').replace('/', '_')}_combined.csv"
            filepath = os.path.join(combined_dir, filename)
            
            print(f"Processing {filepath}.")
            if os.path.exists(filepath):
                print(f"Appending to {filepath}.")
                combined_data.to_csv(filepath, mode='a', header=False, index=False)
            else:
                print(f"Creating {filepath}.")
                combined_data.to_csv(filepath, index=False)

        print("Data fetching, processing, and combining completed.")
        print("Current directory:", os.getcwd())
        try:
            print("Contents of 'data/combined':", os.listdir(combined_dir))
        except Exception as e:
            print(f"Could not list contents of 'data/combined': {e}")
