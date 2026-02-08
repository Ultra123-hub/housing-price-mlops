import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import pickle
import json
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load preprocessed data
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test = pd.read_csv("data/y_test.csv").squeeze()

# Load preprocessor
with open("data/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# Load parameters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("housing-price-prediction")

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(
        n_estimators=params["train"]["random_forest"]["n_estimators"],
        max_depth=params["train"]["random_forest"]["max_depth"],
        random_state=params["train"]["random_state"]
    )
}

all_metrics = {}

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # Log to MLflow
        mlflow.log_param("model_type", model_name)
        if model_name == "RandomForest":
            mlflow.log_param("n_estimators", params["train"]["random_forest"]["n_estimators"])
            mlflow.log_param("max_depth", params["train"]["random_forest"]["max_depth"])

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Log model
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=f"{model_name}_housing_model"
        )
        
        print(f"{model_name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
        
        # Store for DVC metrics
        all_metrics[model_name] = {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2)
        }

# Save metrics for DVC
with open("metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=4)

print("Training complete. Metrics saved to metrics.json")