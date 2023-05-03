import pandas as pd
from pathlib import Path
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

    # Save the report
    report.save(f'{output_path}/{name}.html')


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

    # Save the report
    report.show_html(f'{output_path}/{name}.html', open_browser=False)


# Path creation
cwd = Path.cwd()
root = cwd.parent
data_folder = root / 'data'

train = pd.read_csv(data_folder / 'train.csv')
test = pd.read_csv(data_folder / 'test.csv')

report_name = 'Train Report'
dataprep_report(train, cwd, 'Dataprep '+report_name)
sweetviz_report(train, cwd, 'Sweetviz '+report_name)

print('hello')





