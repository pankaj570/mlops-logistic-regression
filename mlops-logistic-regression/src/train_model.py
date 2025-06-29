# src/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import logging
import os

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def train_model(data_path, C_param, max_iter_param):
    """
    Trains a Logistic Regression model, logs parameters, metrics, and model to MLflow.
    """
    #mlflow.set_experiment("Logistic_Regression_Iris_Experiment")
    mlflow.set_tracking_uri("http://localhost:5000")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_param("C", C_param)
        mlflow.log_param("max_iter", max_iter_param)
        mlflow.log_param("data_path", data_path)

        try:
            df = pd.read_csv(data_path)
            X = df.drop('target', axis=1)
            y = df['target']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            model = LogisticRegression(C=C_param, max_iter=max_iter_param, solver='liblinear', random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            print(f"Metrics - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")

            mlflow.sklearn.log_model(model, "logistic_regression_model")
            print("Model logged to MLflow.")
            print(f"MLflow Run ID: {run_id}")
            return run_id

        except Exception as e:
            logger.exception(
                f"Unable to train model. Error: {e}"
            )
            return None

if __name__ == "__main__":
    # Ensure MLflow tracking URI is set (e.g., to a local server)
    # If you run a separate MLflow server, uncomment and adjust the line below:
    # mlflow.set_tracking_uri("http://localhost:5000")

    # DVC pull to ensure data is available before training
    print("Ensuring data is available via DVC pull...")
    os.system("dvc pull src/data/raw/iris.csv") # This assumes DVC is configured

    data_path = 'src/data/raw/iris.csv'

    print("\n--- Training Run 1 ---")
    C_param_1 = 0.1
    max_iter_param_1 = 100
    run_id_1 = train_model(data_path, C_param_1, max_iter_param_1)
    if run_id_1:
        print(f"Model 1 trained with Run ID: {run_id_1}")

    print("\n--- Training Run 2 ---")
    C_param_2 = 1.0
    max_iter_param_2 = 200
    run_id_2 = train_model(data_path, C_param_2, max_iter_param_2)
    if run_id_2:
        print(f"Model 2 trained with Run ID: {run_id_2}")