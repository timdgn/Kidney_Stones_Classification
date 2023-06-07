import pickle
import numpy as np
import os
import glob


def load_model():
    """
    Loads the latest model file from the models folder

    Parameters
    ----------
    None

    Returns
    -------
    model : object
        The model object loaded from the pickle file.
    """

    # Get the list of files in the folder that match the pattern
    files = glob.glob("models/*.pkl")

    # Sort the files by their modification time in descending order
    files.sort(key=os.path.getmtime, reverse=True)

    # Get the last file created
    last_file = files[0]

    # Load the model file
    with open(last_file, "rb") as f:
        model = pickle.load(f)

    return model


def inference(gravity, ph, osmo, cond, urea, calc):
    """
    Makes predictions on the input features using the loaded model.

    Parameters
    ----------
    gravity : float
        The specific gravity of the urine sample.
    ph : float
        The pH of the urine sample.
    osmo : float
        The osmolality of the urine sample.
    cond : float
        The conductivity of the urine sample.
    urea : float
        The urea concentration of the urine sample.
    calc : float
        The calcium concentration of the urine sample.

    Returns
    -------
    predictions[0] : int
        The predicted class label for the input features (0 for normal, 1 for abnormal)
    """

    # Load the model
    model = load_model()

    # Define the input features for inference
    input_data = np.array([gravity, ph, osmo, cond, urea, calc])

    # Make predictions
    predictions = model.predict(input_data.reshape(1, -1))

    return predictions[0]


if __name__ == "__main__":

    # Test the inference() fct to see if it gives a prediction
    print(inference(1.013, 6.19, 443, 14.8, 124, 1.45))
