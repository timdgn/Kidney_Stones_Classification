from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, auc, roc_curve, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from eda_reports import *
import pickle
import datetime


def plot_roc_auc(fpr, tpr, auc_score):
    """
    Plots the ROC curve and the AUC score.

    Parameters
    ----------
    fpr : array-like
        The false positive rate values for different thresholds.
    tpr : array-like
        The true positive rate values for different thresholds.
    auc_score : float
        The area under the ROC curve.

    Returns
    -------
    None

    Outputs
    -------
    An image file named "ROC AUC Curve.png" in the docs folder.
    A matplotlib figure showing the ROC curve and the AUC score.
    """

    # Set the style of the plot
    sns.set_style("darkgrid")

    # Create a dataframe with the fpr and tpr values
    df = pd.DataFrame({"fpr": fpr, "tpr": tpr})

    # Plot the line with confidence intervals
    sns.lineplot(x="fpr", y="tpr", data=df, label=f"AUC = {auc_score}")

    # Plot the diagonal line
    plt.plot([0, 1], [0, 1], "k--")

    # Add title and labels
    plt.title("ROC curve")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")

    # Save the plot as an image
    plt.savefig(f"{docs_folder}/ROC AUC Curve.png", dpi=300)

    # Show the legend and the plot
    plt.legend()
    plt.show()


def plot_confusion_matrix(y_test, y_pred):
    """
    Plots the confusion matrix.

    Parameters
    ----------
    y_test : array-like
        The true labels of the test set.
    y_pred : array-like
        The predicted labels of the test set.

    Returns
    -------
    None

    Outputs
    -------
    An image file named "Confusion Matrix.png" in the docs folder.
    A matplotlib figure showing the confusion matrix.
    """

    # Plotting the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Using seaborn heatmap function
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    # Adding title and labels
    plt.title("Confusion matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    # Save the plot as an image
    plt.savefig(f"{docs_folder}/Confusion Matrix.png", dpi=300)

    # Showing the plot
    plt.show()


def xgb_predict(X_train, y_train, X_test):
    """
    Trains and predicts a binary classifier using XGBoost.

    Parameters
    ----------
    X_train : array-like
        The features of the training set.
    y_train : array-like
        The labels of the training set.
    X_test : array-like
        The features of the test set.

    Returns
    -------
    y_pred : array-like
        The predicted labels of the test set.
    """

    # Create the model and fit it
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # Predict with the xgb model
    y_pred = model.predict(X_test)

    # Save the model in the "models" directory
    now = datetime.datetime.now()
    filename = f"xgb_model_{now.strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
    pickle.dump(model, open(f"{models_folder}/{filename}", "wb"))

    return y_pred


def evaluate_model(y_test, y_pred):
    """
    Evaluates the performance of the binary classifier using Accuracy, F1 and AUC scores.

    Parameters
    ----------
    y_test : array-like
        The true labels of the test set.
    y_pred : array-like
        The predicted labels of the test set.

    Returns
    -------
    None

    Outputs
    -------
    Prints the accuracy, F1 score and AUC score of the classifier.
    Calls the plot_confusion_matrix and plot_roc_auc functions to generate plots.
    """

    # Calculate and print the accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Calculate and print the F1 score
    f1 = f1_score(y_test, y_pred)
    print(f"F1 score: {f1}")

    # Plotting the ROC curve and computing the AUC score
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc_score = auc(fpr, tpr)
    print(f"AUC score: {auc_score}")

    plot_confusion_matrix(y_test, y_pred)

    plot_roc_auc(fpr, tpr, auc_score)


if __name__ == "__main__":

    # Path creation
    cwd = Path.cwd()
    root = cwd.parent
    data_folder = f"{root}/data"
    docs_folder = f"{root}/docs"
    models_folder = f"{root}/models"

    # Train/test split
    train_df = pd.read_csv(f"{data_folder}/train.csv")
    X = train_df.drop("target", axis=1)
    y = train_df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

    # Predict with the xgb model
    y_pred = xgb_predict(X_train, y_train, X_test)

    # Evaluate the results
    evaluate_model(y_test, y_pred)

    print("Finished ✅")
