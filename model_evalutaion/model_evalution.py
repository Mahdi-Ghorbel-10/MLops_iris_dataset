# model_evaluation.py
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

def main():
    # Load test data and the best model
    test_df = pd.read_csv('data_ingestion_preprocess/data/test.csv')
    X_test = test_df.drop(columns=['label'])
    y_test = test_df['label']
    model = joblib.load('model_taining/model/best_model.pkl')

    # Generate predictions and compute accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Save the evaluation metric (accuracy) to a file
    with open('model_evalutaion/accuracy.txt', 'w') as f:
        f.write(str(accuracy))

    print(f"Model evaluation complete. Accuracy: {accuracy}")

if __name__ == '__main__':
    main()
