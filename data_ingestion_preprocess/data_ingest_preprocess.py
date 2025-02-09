# data_ingest_preprocess.py
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def main():
    # Load the Iris dataset as a DataFrame
    iris = load_iris(as_frame=True)
    data = iris.frame
    # Optional preprocessing: rename the target column for clarity
    data.rename(columns={'target': 'label'}, inplace=True)

    # Split data into training and testing sets (80/20 split)
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

    # Save the splits to CSV files; in production these could be pushed to cloud storage.
    train_df.to_csv('data_ingestion_preprocess/data/train.csv', index=False)
    test_df.to_csv('data_ingestion_preprocess/data/test.csv', index=False)

    print("Data ingestion and preprocessing complete. Files saved to /data/train.csv and /data/test.csv")

if __name__ == '__main__':
    main()
