import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def preprocess_dohmh(input_path="data/raw_dohmh.csv", output_train="data/train.csv", output_test="data/test.csv"):
    """
    Cleans the raw DOHMH data, encodes categorical features, and splits into train/test sets.
    """
    if not os.path.exists(input_path):
        print(f"{input_path} not found. Please run download_data.py first.")
        return

    print("Loading raw data...")
    df = pd.read_csv(input_path)
    
    # We are trying to predict the Health Grade. Let's drop rows without a grade.
    df = df.dropna(subset=['grade'])
    
    # We only care about A, B, C grades for simplicity in the classification task
    df = df[df['grade'].isin(['A', 'B', 'C'])]
    
    # Map grades to target indices: A -> 0, B -> 1, C -> 2
    grade_map = {'A': 0, 'B': 1, 'C': 2}
    df['target'] = df['grade'].map(grade_map)
    
    # Select our features. We will use categorical features which we'll one-hot encode later
    # or use embeddings for. Let's use: boro, cuisine_description, critical_flag.
    features = ['boro', 'cuisine_description', 'critical_flag']
    
    for col in features:
        df[col] = df[col].fillna("Unknown")
        
    # We will do a generic 80/20 split to simulate testing generalization. 
    # For a true OOD (Out-of-Distribution) test (Week 5 concept), we might split by 'boro'.
    # Here, we do standard shuffling.
    X = df[features]
    y = df['target']
    
    # Combine back to save as CSVs
    final_df = df[features + ['target']]
    
    train_df, test_df = train_test_split(final_df, test_size=0.2, random_state=42)
    
    os.makedirs(os.path.dirname(output_train), exist_ok=True)
    os.makedirs(os.path.dirname(output_test), exist_ok=True)
    
    train_df.to_csv(output_train, index=False)
    test_df.to_csv(output_test, index=False)
    print(f"Preprocessing complete. Total records: {len(final_df)}.")
    print(f"Train set saved to {output_train} ({len(train_df)} rows)")
    print(f"Test set saved to {output_test} ({len(test_df)} rows)")

if __name__ == "__main__":
    preprocess_dohmh()
