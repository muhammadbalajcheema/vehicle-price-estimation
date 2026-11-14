import json
import yaml
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PROCESSED_PATH = "data/processed/vehicles_processed.csv"

def train_and_register():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    test_size = params["data"]["test_size"]
    random_state = params["data"]["random_state"]
    n_estimators = params["model"]["n_estimators"]
    max_depth = params["model"]["max_depth"]
    min_samples_split = params["model"]["min_samples_split"]
    model_name = params["model"]["registry_model_name"]

    df = pd.read_csv(PROCESSED_PATH)
    X = df.drop(columns=["price_lakhs"])
    y = df["price_lakhs"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Vehicle_Price_Estimation_Phase1")

    with mlflow.start_run(run_name="RandomForest_Price_Estimator"):
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("num_features", X.shape[1])

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        mlflow.log_metric("rmse_lakhs", rmse)
        mlflow.log_metric("mae_lakhs", mae)
        mlflow.log_metric("r2_score", r2)

        metrics_data = {"rmse_lakhs": rmse, "mae_lakhs": mae, "r2_score": r2}
        with open("metrics.json", "w") as mf:
            json.dump(metrics_data, mf, indent=4)
        mlflow.log_artifact("metrics.json")

        feat_imp = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        feat_imp.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="vehicle_price_model",
            registered_model_name=model_name
        )

        client = MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
        client.set_model_version_tag(
            name=model_name,
            version=latest_version,
            key="stage",
            value="phase1_baseline"
        )
        client.set_model_version_tag(
            name=model_name,
            version=latest_version,
            key="data_version",
            value="dvc_v1"
        )

        print(f"Training Complete | RMSE: {rmse:.3f} Lakhs | MAE: {mae:.3f} Lakhs | R2: {r2:.4f}")
        print(f"Registered '{model_name}' Version {latest_version} in MLflow Model Registry.")

if __name__ == "__main__":
    train_and_register()