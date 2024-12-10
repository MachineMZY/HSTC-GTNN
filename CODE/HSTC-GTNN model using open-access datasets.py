import os
import requests
import zipfile
import pandas as pd
import numpy as np

# 1. Download OpenStreetMap (OSM) Data (Example using Geofabrik for OSM extracts)
def download_osm_data(region="beijing"):
    url = f"https://download.geofabrik.de/asia/{region}-latest.osm.bz2"
    filename = f"{region}-latest.osm.bz2"
    
    # Check if the file already exists
    if not os.path.exists(filename):
        print(f"Downloading {region} OSM data...")
        response = requests.get(url, stream=True)
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete!")
    else:
        print("File already downloaded.")

# 2. Download Uber Movement Data (Example for NYC)
def download_uber_data(city="new-york-city"):
    url = f"https://movement.uber.com/cities/{city}/data"
    filename = f"{city}-movement-data.zip"
    
    # Check if the file already exists
    if not os.path.exists(filename):
        print(f"Downloading Uber movement data for {city}...")
        response = requests.get(url, stream=True)
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete!")
    else:
        print("File already downloaded.")
    
    # Unzipping the file if needed
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall(f"{city}_movement_data")
    print(f"{city} movement data extracted.")

# 3. Generate Synthetic Traffic and Task Data
def generate_synthetic_data(num_samples=1000):
    print("Generating synthetic data...")
    data = {
        "vehicle_id": np.random.randint(1, 100, num_samples),
        "task_id": np.random.randint(1, 500, num_samples),
        "start_lat": np.random.uniform(40.6, 40.9, num_samples),
        "start_lon": np.random.uniform(-74.0, -73.7, num_samples),
        "end_lat": np.random.uniform(40.6, 40.9, num_samples),
        "end_lon": np.random.uniform(-74.0, -73.7, num_samples),
        "energy_consumption": np.random.uniform(1, 10, num_samples),
        "completion_time": np.random.normal(30, 5, num_samples),
        "task_priority": np.random.choice(["high", "medium", "low"], num_samples)
    }
    
    df = pd.DataFrame(data)
    df.to_csv("synthetic_task_data.csv", index=False)
    print("Synthetic data generated and saved as synthetic_task_data.csv.")

# Main function to run downloads
if __name__ == "__main__":
    # Download OSM data for Beijing
    download_osm_data("beijing")
    
    # Download Uber data for New York City
    download_uber_data("new-york-city")
    
    # Generate synthetic data for testing
    generate_synthetic_data()
