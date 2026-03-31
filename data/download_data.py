"""
download_data.py — Fetch NYC DOHMH Restaurant Inspection Results from OpenData API

Downloads the dataset via the Socrata Open Data API (SODA) and saves as CSV.
Default limit is 50,000 rows for development; set a higher limit for final training.

Author: Ryan Han (Data & DevOps)
Course: CSCI-UA 473 · Fundamentals of Machine Learning · Spring 2026
"""

import os
import sys
import pandas as pd


def download_dohmh_data(limit=50000, output_path="data/raw_dohmh.csv"):
    """
    Downloads the NYC DOHMH Restaurant Inspection Results dataset.

    Parameters
    ----------
    limit : int
        Number of rows to fetch. The full dataset has ~400k rows.
        Use 50000 for development, 200000+ for final training.
    output_path : str
        Where to save the resulting CSV file.
    """
    # NYC Open Data SODA API endpoint (dataset ID: 43nn-pn8j)
    url = f"https://data.cityofnewyork.us/resource/43nn-pn8j.csv?$limit={limit}"

    print(f"Downloading DOHMH dataset ({limit:,} rows)...")
    print(f"  Source: {url}")

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df = pd.read_csv(url, low_memory=False)
        print(f"  Downloaded {len(df):,} rows × {len(df.columns)} columns.")

        # Quick sanity check
        expected_cols = ["camis", "dba", "boro", "cuisine_description",
                         "score", "grade", "critical_flag"]
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            print(f"  WARNING: expected columns missing: {missing}")
            print(f"  Available columns: {list(df.columns)}")

        df.to_csv(output_path, index=False)
        print(f"  Saved to {output_path}")
        print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

    except Exception as e:
        print(f"ERROR: Failed to download data — {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Allow overriding limit from command line: python download_data.py 100000
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    download_dohmh_data(limit=limit)
