from pathlib import Path
import datetime

import pandas as pd
import sweetviz
from dataprep.eda import create_report


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
    report = sweetviz.analyze((df, name))

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
    dataprep_report(df, output_path, "Dataprep " + report_name)
    sweetviz_report(df, output_path, "Sweetviz " + report_name)


def main():
    """
    This function creates the paths for the reports and the train data, reads the train data as a pandas dataframe,
    and generates two reports using the dataprep and sweetviz modules.

    Parameters
    ----------
    # None

    Returns
    -------
    None
    """

    # Path creation
    cwd = Path.cwd()
    parent = cwd.parent
    reports_folder = parent.joinpath("docs", "reports")
    train_csv = parent.joinpath("data", "train.csv")

    # Train & test dataframes
    train_df = pd.read_csv(train_csv)

    # Generate a dataprep and a sweetviz report
    reports(train_df, str(reports_folder))

    print("Finished ✅")


if __name__ == "__main__":
    main()
