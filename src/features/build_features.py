"""This script builds features from the raw data."""


import os
import sys
import logging
import yaml
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


if len(sys.argv) != 3 and len(sys.argv) != 5:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write(
        "\tpython build_features.py data-dir-path ouput-dir-path\n"
    )
    sys.exit(1)


input_path, output_path = sys.argv[1], sys.argv[2]
input_path = os.path.join(
    input_path, "biomedical-text-publication-classification.zip"
)
processed_output_path = os.path.join(output_path, "processed_data.csv")


with open("params.yaml", "r", encoding="utf-8") as file:
    params = yaml.load(file, Loader=yaml.SafeLoader)
    encoding = params["build_features"]["encoding"]
    column_names = params["build_features"]["column_names"]
    use_cols = params["build_features"]["use_cols"]
    header = params["build_features"]["header"]
    text_column = params["build_features"]["text_column"]


def clean_text(input_data, output_path):
    """Clean text column in dataframe.

    Parameters:

    input_data (pandas.DataFrame): Dataframe to clean.
    output_path (str): Path to save the cleaned dataframe.
    """
    logger.info("Loading data...")
    dataframe = pd.read_csv(
        input_data,
        encoding=encoding,
        names=column_names,
        usecols=use_cols,
        header=header,
    )
    logger.info("Building features...")

    # apply map_target function to target in train and test at the same time
    dataframe[text_column] = dataframe[text_column].str.lower()
    dataframe[text_column] = dataframe[text_column].str.replace(
        r"[^\w\s]+", ""
    )
    dataframe[text_column] = dataframe[text_column].str.replace(r"\d+", "")
    dataframe[text_column] = dataframe[text_column].astype(str)

    logger.info("Saving data...")
    os.makedirs(sys.argv[2], exist_ok=True)
    dataframe.to_csv(output_path, index=False)


if __name__ == "__main__":
    clean_text(input_path, processed_output_path)
