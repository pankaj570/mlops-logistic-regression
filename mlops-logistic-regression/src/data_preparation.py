# src/data_preparation.py
import pandas as pd
from sklearn.datasets import load_iris
import os

def prepare_data():
    """Loads Iris dataset and saves it as CSV."""
    print("Preparing data...")
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    os.makedirs('data/raw', exist_ok=True)
    csv_path = 'data/raw/iris.csv'
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")
    return csv_path

if __name__ == "__main__":
    prepare_data()