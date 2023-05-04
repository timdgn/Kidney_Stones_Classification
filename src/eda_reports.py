import pandas as pd
from pathlib import Path
import datetime
from dataprep.eda import create_report
import sweetviz


def dataprep_report(df, output_path, name):
    """
    Generates a data report using dataprep and saves it as an HTML file.

    Parameters
    ----------
    df : pandas.DataFrame
        The input data frame to be analyzed.
    output_path : str
        The path where the report will be saved.
    name : str
        The title of the report.

    Returns
    -------
    None
    """

    # Create the report
    report = create_report(df, title=name)

    # See the report in the notebook
    # report

    # Get the current date as a string
    date = datetime.date.today().strftime("%Y-%m-%d")

    # Save the report
    report.save(f"{output_path}/{name} {date}.html")


def sweetviz_report(df, output_path, name):
    """
    Generates a data report using sweetviz and saves it as an HTML file.

    Parameters
    ----------
    df : pandas.DataFrame
        The input data frame to be analyzed.
    output_path : str
        The path where the report will be saved.
    name : str
        The title of the report.

    Returns
    -------
    None
    """

    # Create the report
    report = sweetviz.analyze([df, name])

    # See the report in the notebook
    # report.show_notebook()

    # Get the current date as a string
    date = datetime.date.today().strftime("%Y-%m-%d")

    # Save the report
    report.show_html(f"{output_path}/{name} {date}.html", open_browser=False)


def reports(df, output_path):
    """
    Generates two reports for a given dataframe: a dataprep report and a sweetviz report.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to be analyzed.
    output_path : str
        The path where the reports will be saved.

    Returns
    -------
    None
    """

    # Generating reports
    report_name = "Train Report"
    dataprep_report(df, output_path, "Dataprep "+report_name)
    sweetviz_report(df, output_path, "Sweetviz "+report_name)


if __name__ == "__main__":

    # Path creation
    cwd = Path.cwd()
    root = cwd.parent
    data_folder = f"{root}/data"
    docs_folder = f"{root}/docs"
    train_csv = f"{data_folder}/train.csv"

    # Train & test dataframes
    train_df = pd.read_csv(train_csv)

    # Generate a dataprep and a sweetviz report
    reports(train_df, docs_folder)

    print("Finished ✅")
