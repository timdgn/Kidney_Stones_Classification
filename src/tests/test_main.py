import os
from main import *
import pytest
import numpy as np
from xgboost import XGBClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Define the models folder
models_folder = "models"

# Define a fixture to generate some synthetic data
@pytest.fixture(scope="module")
def data():
    # Generate a random binary classification problem with 100 samples and 10 features
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)

    # Split the data into train and test sets (80/20)
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    return X_train, y_train, X_test, y_test


# Define a test function to check the output and behavior of the xgb_predict function
def test_xgb_predict(data):
    # Unpack the data from the fixture
    X_train, y_train, X_test, y_test = data

    # Call the xgb_predict function and get the predictions
    y_pred = xgb_predict(X_train, y_train, X_test)

    # Check that the predictions are an array of integers (0 or 1)
    assert isinstance(y_pred, np.ndarray)
    assert all(y_pred == 0) or all(y_pred == 1)

    # Check that a model file is saved in the models folder with the correct format
    assert os.path.exists(models_folder)

    # Get the list of files in the models folder
    files = os.listdir(models_folder)

    # Check that there is at least one file in the models folder
    assert len(files) > 0

    # Check that the last file in the models folder has the expected format (xgb_model_YYYY-MM-DD_HH-MM-SS.pkl)

    last_file = files[-1]

    # Split the file name by underscore and dot
    parts = last_file.split("_")

    # Check that the before-the-last part is a valid date (YYYY-MM-DD)
    try:
        date = datetime.datetime.strptime(parts[-2], "%Y-%m-%d")
    except ValueError:
        pytest.fail("Invalid date format")

    # Check that the last part is a valid time (HH-MM-SS.pkl)
    try:
        time = datetime.datetime.strptime(parts[-1], "%H-%M-%S.pkl")
    except ValueError:
        pytest.fail("Invalid time format")
