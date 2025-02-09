# data_drift_monitor.py
import numpy as np
import pandas as pd
from prometheus_client import start_http_server, Gauge
import time

# Define a Prometheus Gauge for PSI
psi_metric = Gauge('psi_metric', 'Population Stability Index (PSI) for data drift detection')

def calculate_psi(reference, current, bins=10):
    # Create bins based on the reference distribution
    ref_percents, bin_edges = np.histogram(reference, bins=bins, density=True)
    curr_percents, _ = np.histogram(current, bins=bin_edges, density=True)

    # Avoid division by zero by replacing zeros
    ref_percents = np.where(ref_percents == 0, 0.0001, ref_percents)
    curr_percents = np.where(curr_percents == 0, 0.0001, curr_percents)

    # Compute PSI
    psi = np.sum((ref_percents - curr_percents) * np.log(ref_percents / curr_percents))
    return psi

def main():
    # Load reference distribution from training data (example: using one feature)
    ref_df = pd.read_csv('data_ingestion_preprocess/data/train.csv')
    reference = ref_df['sepal length (cm)']  # Adjust feature name as needed

    # Start Prometheus metrics server on port 8000
    start_http_server(9090)
    print("Prometheus metrics available at http://localhost:9090/metrics")

    while True:
        # In production, replace this with actual production data ingestion
        curr_df = pd.read_csv('data_ingestion_preprocess/data/iris_dataset.csv')
        current = curr_df['sepal length (cm)']
        
        psi_value = calculate_psi(reference, current)
        psi_metric.set(psi_value)
        
        print(f"Calculated PSI: {psi_value}")
        time.sleep(60)  # Update every minute

if __name__ == '__main__':
    main()
