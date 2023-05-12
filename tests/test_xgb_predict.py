import os
import pytest
import numpy as np
from pathlib import Path
from ..FastAPI_Backend.src.main import xgb_predict


# Define a fixture to generate some synthetic data
@pytest.fixture()
def data():
    # Create some dummy data for testing
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(50, 10)
    return X_train, y_train, X_test


def test_xgb_predict(data):
    # Call the function and get the predictions
    X_train, y_train, X_test = data
    models_folder = str(Path(__file__).parent.parent / "FastAPI_Backend/models")
    y_pred, model_name = xgb_predict(X_train, y_train, X_test, models_folder)

    # Check that the predictions are binary
    assert np.all(np.isin(y_pred, [0, 1]))

    # Check that the model is saved in the output folder
    model_file = f"{models_folder}/{model_name}"
    assert os.path.exists(model_file)

    # Remove the model created by the test
    os.remove(model_file)
