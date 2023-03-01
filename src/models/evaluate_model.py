"""This script is used to predict the output of the model."""


import os
import sys
import logging
import pickle
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from score_model import score_model, save_metrics
from plot_metric import (
    plot_confusion_matrix,
    plot_precision_recall_curve,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


if len(sys.argv) != 4 and len(sys.argv) != 5:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write(
        "\tpython evaluate_model.py input_data-path train-model-path"
        " predictions-output-path\n"
    )
    sys.exit(1)


input_path, model_path, prediction_path = sys.argv[1], sys.argv[2], sys.argv[3]
test_input_path = os.path.join(input_path, "test.csv")
model_input_path = os.path.join(model_path, "model.pkl")
encoder_input_path = os.path.join(model_path, "encoder.pkl")
metrics_path = os.path.join(prediction_path, "metrics.txt")
predict_path = os.path.join(prediction_path, "predictions.csv")
confusion_matrix_path = os.path.join(prediction_path, "confusion_matrix.png")
# precision_recall_path = os.path.join(prediction_path, "precision_recall.png")
# roc_auc_path = os.path.join(prediction_path, "roc_auc.png")


with open("params.yaml", "r", encoding="utf-8") as file:
    params = yaml.load(file, Loader=yaml.SafeLoader)
    encoding = params["evaluate_model"]["encoding"]
    target = params["evaluate_model"]["target"]
    text_column = params["evaluate_model"]["text_column"]


def predict_model(test_path, model, pred_path):
    """Predict the output of the model.

    params:
        test_path: path to the test data
        model: path to the model
        pred_path: path to the predictions output
    """
    logger.info("Loading data...")
    test = pd.read_csv(test_path, encoding=encoding)
    X_test = test.drop(target, axis=1)
    y_test = test[target]
    logger.info("Loading model...")
    with open(model, "rb") as file:
        model = pickle.load(file)
    logger.info("Loading encoder...")
    with open(encoder_input_path, "rb") as file:
        encoder = pickle.load(file)
    y_test = encoder.transform(y_test)
    logger.info("Predicting output...")
    y_pred = model.predict(X_test)
    os.makedirs(sys.argv[3], exist_ok=True)
    logger.info("Computing metrics...")
    accuracy, balanced_accuracy, precision, f1 = score_model(y_test, y_pred)
    save_metrics(metrics_path, accuracy, balanced_accuracy, precision, f1)
    logger.info("Generating plots...")
    plot_confusion_matrix(model, X_test, y_test, confusion_matrix_path)
    # plot_precision_recall_curve(model, X_test, y_test, precision_recall_path)
    # plot_roc_auc(model, X_test, y_test, roc_auc_path)
    logger.info("Saving output...")
    output = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})
    output.to_csv(pred_path, index=False)


if __name__ == "__main__":
    predict_model(test_input_path, model_input_path, predict_path)
