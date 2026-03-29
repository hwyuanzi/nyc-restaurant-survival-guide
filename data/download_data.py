import os
import pandas as pd

def download_dohmh_data(limit=50000, output_path="data/raw_dohmh.csv"):
    """
    Downloads the NYC DOHMH Restaurant Inspection Results dataset.
    Uses a limit of 50,000 requests for development purposes to keep things fast,
    but can be expanded for the final training run.
    """
    # NYC Open Data API endpoint for DOHMH dataset
    url = f"https://data.cityofnewyork.us/resource/43nn-pn8j.csv?$limit={limit}"
    
    print(f"Downloading dataset from {url}...")
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Load directly into pandas and save
        df = pd.read_csv(url)
        print(f"Successfully downloaded {len(df)} rows.")
        df.to_csv(output_path, index=False)
        print(f"Saved raw data to {output_path}")
    except Exception as e:
        print(f"Error downloading data: {e}")

if __name__ == "__main__":
    download_dohmh_data()
