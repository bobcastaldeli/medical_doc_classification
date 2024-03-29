import os
import sys
import logging
import pickle
import yaml
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from embetter.grab import ColumnGrabber
from embetter.text import SentenceEncoder


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


if len(sys.argv) != 3 and len(sys.argv) != 5:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write(
        "\tpython train_model.py input-data-path ouput-model-path\n"
    )
    sys.exit(1)


input_path, output_path = sys.argv[1], sys.argv[2]
train_input_path = os.path.join(input_path, "train.csv")
model_output_path = os.path.join(output_path, "model.pkl")
encoder_output_path = os.path.join(output_path, "encoder.pkl")
# model_output_path = os.path.join(output_path, "model.pkl")


with open("params.yaml", "r", encoding="utf-8") as file:
    params = yaml.load(file, Loader=yaml.SafeLoader)
    encoding = params["train_model"]["encoding"]
    text_column = params["train_model"]["text_column"]
    target = params["train_model"]["target"]
    encoding = params["train_model"]["encoding"]
    pretrained_model = params["train_model"]["pretrained_model"]


def train_model(train_path, model_path, encoder_path):
    """Train the model.

    params:
        train_path: path to the train data
        model_path: path to the model
        encoder_path: path to the encoder
    """
    logger.info("Loading data...")
    train = pd.read_csv(train_path, encoding=encoding)
    X_train = pd.DataFrame(train.drop(target, axis=1))
    y_train = pd.DataFrame(train[target])

    # Encode target
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)

    # Create pipeline
    logger.info("Training model...")
    pipeline = make_pipeline(
        ColumnGrabber(text_column),
        SentenceEncoder(pretrained_model),
        LogisticRegression(),
    )

    # Fit model
    pipeline.fit(X_train, y_train)
    logger.info("Saving model...")
    os.makedirs(sys.argv[2], exist_ok=True)
    with open(model_path, "wb") as file:
        pickle.dump(pipeline, file)
    logger.info("Saving encoder...")
    with open(encoder_path, "wb") as file:
        pickle.dump(label_encoder, file)


if __name__ == "__main__":
    train_model(train_input_path, model_output_path, encoder_output_path)
