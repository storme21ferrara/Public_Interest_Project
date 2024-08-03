import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelTrainingError(Exception):
    """Custom exception for errors during model training."""
    pass

def test_model_training(data, target_column):
    """
    Function to train and evaluate a RandomForestClassifier on the provided data.

    Parameters:
    data (pd.DataFrame): DataFrame containing the dataset.
    target_column (str): The name of the target column in the dataset.

    Returns:
    best_model (RandomForestClassifier): Best trained model.
    best_report (str): Classification report of the best model.
    """
    try:
        # Check if the target column exists in the DataFrame
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data.")

        # Splitting the dataset into features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Splitting the dataset into training and testing sets with stratification
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, stratify=y_resampled, random_state=42)

        # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Initializing RandomForestClassifier with GridSearchCV for hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=StratifiedKFold(n_splits=5), n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # Making predictions on the test set
        y_pred = best_model.predict(X_test)

        # Calculating accuracy and generating classification report
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        logging.info(f"Best Model Parameters: {grid_search.best_params_}")
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"Precision: {precision}")
        logging.info(f"Recall: {recall}")
        logging.info(f"F1 Score: {f1}")
        logging.info("Classification Report:")
        logging.info(report)

        return best_model, report

    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
        raise ModelTrainingError(f"ValueError encountered: {ve}")
    except Exception as e:
        logging.error(f"Error in model training: {e}")
        raise ModelTrainingError(f"An unexpected error occurred during model training: {e}")

if __name__ == "__main__":
    # Example usage
    try:
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'target': [0, 1, 0, 1, 0]
        })
        model, report = test_model_training(data, 'target')
    except ModelTrainingError as e:
        logging.error(f"Model training failed: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred in the main function: {e}")
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
