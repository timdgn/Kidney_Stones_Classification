import pickle
import numpy as np
import os
import glob


def load_model():
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
