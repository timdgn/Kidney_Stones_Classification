import pickle
import numpy as np
from pathlib import Path
import os
import glob


def inference(gravity, ph, osmo, cond, urea, calc):
    # Path creation
    cwd = Path.cwd()
    root = cwd.parent
    models_folder = f"{root}/models"

    # Get the list of files in the folder that match the pattern
    files = glob.glob(f"{models_folder}/*.pkl")

    # Sort the files by their modification time in descending order
    files.sort(key=os.path.getmtime, reverse=True)

    # Get the last file created
    last_file = files[0]

    # Load the model file
    with open(last_file, "rb") as f:
        model = pickle.load(f)

    # Define the input features for inference as a numpy array
    # You should modify this according to your data format and feature names
    input_data = np.array([gravity, ph, osmo, cond, urea, calc])

    # Make predictions
    predictions = model.predict(input_data.reshape(1, -1))

    return predictions[0]


if __name__ == "__main__":
    print(inference(1.013, 6.19, 443, 14.8, 124, 1.45))
