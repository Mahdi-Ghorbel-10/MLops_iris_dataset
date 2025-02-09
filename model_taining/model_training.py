# model_training.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def main():
    # Load training data saved from the ingestion step
    train_df = pd.read_csv('data_ingestion_preprocess/data/train.csv')
    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label']

    # Define hyperparameter grid for RandomForestClassifier
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 5, 10]
    }

    # Initialize the classifier and perform grid search with 5-fold cross-validation
    clf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Save the best model using joblib
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'model_taining/model/best_model.pkl')

    print("Model training complete. Best model saved to /model/best_model.pkl")

if __name__ == '__main__':
    main()
