# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import logging
import sys
import warnings

import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # # Read the wine-quality csv file from the URL to get training and test datasets
    train = pd.read_csv(sys.argv[1], sep=',')
    test = pd.read_csv(sys.argv[2], sep=',')

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[3]) if len(sys.argv) > 2 else 0.5
    l1_ratio = float(sys.argv[4]) if len(sys.argv) > 3 else 0.5

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        # create and log artifacts
        predicted_qualities = lr.predict(test_x)
        plt.scatter(test_y, predicted_qualities,color='g')
        plt.savefig('results.png')
        mlflow.log_artifact("results.png")

        dictionary = {"a":"b"}
        # Log a dictionary as a JSON file under the run's root artifact directory
        mlflow.log_dict(dictionary, "additional-artifacts/data.json")
        # Log a dictionary as a YAML file in a subdirectory of the run's root artifact directory
        mlflow.log_dict(dictionary, "additional-artifacts/data.yml")
        # If the file extension doesn't exist or match any of [".json", ".yaml", ".yml"],
        # JSON format is used.
        mlflow.log_dict(dictionary, "data")
        mlflow.log_dict(dictionary, "data.txt")

        mlflow.log_text("text1", "additional-artifacts/file1.txt")
        # Log text in a subdirectory of the run's root artifact directory
        mlflow.log_text("text2", "file2.txt")
        # Log HTML text
        mlflow.log_text("<h1>header</h1>", "additional-artifacts/index.html")


        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)



        mlflow.sklearn.log_model(lr, "model")
