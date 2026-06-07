import json
import math
import pickle
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_FILE = Path("eBayNetwork.csv")
OUTPUT_DIR = Path("model_artifacts")
MODEL_FILE = OUTPUT_DIR / "weight_prediction_model.pkl"
METRICS_FILE = OUTPUT_DIR / "model_metrics.json"
PREDICTIONS_FILE = OUTPUT_DIR / "test_predictions.csv"
FEATURES_FILE = OUTPUT_DIR / "top_feature_importance.csv"


def build_preprocessor() -> ColumnTransformer:
    categorical_features = ["Seller", "Bidder"]
    numeric_features = ["Bidder.Volume", "Seller.Volume"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    encoder_kwargs = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        encoder_kwargs["sparse_output"] = False
    else:
        encoder_kwargs["sparse"] = False

    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(**encoder_kwargs), categorical_features),
            ("numeric", numeric_pipeline, numeric_features),
        ]
    )


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", GradientBoostingRegressor(random_state=42)),
        ]
    )


def rmse(y_true, y_pred) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    expected_columns = {"Seller", "Bidder", "Weight", "Bidder.Volume", "Seller.Volume"}
    missing_columns = expected_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    X = df[["Seller", "Bidder", "Bidder.Volume", "Seller.Volume"]]
    y = df["Weight"]

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    baseline = DummyRegressor(strategy="median")
    model = build_model()

    baseline_mae = -cross_val_score(
        baseline, X[["Bidder.Volume", "Seller.Volume"]], y, cv=cv, scoring="neg_mean_absolute_error"
    ).mean()
    model_mae = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error").mean()
    model_r2 = cross_val_score(model, X, y, cv=cv, scoring="r2").mean()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rounded_predictions = np.clip(np.rint(predictions), 1, None).astype(int)

    test_mae = mean_absolute_error(y_test, predictions)
    test_rmse = rmse(y_test, predictions)
    test_r2 = r2_score(y_test, predictions)

    OUTPUT_DIR.mkdir(exist_ok=True)

    with MODEL_FILE.open("wb") as model_handle:
        pickle.dump(model, model_handle)

    prediction_frame = X_test.copy()
    prediction_frame["Actual.Weight"] = y_test.values
    prediction_frame["Predicted.Weight"] = predictions.round(3)
    prediction_frame["Predicted.Weight.Rounded"] = rounded_predictions
    prediction_frame.to_csv(PREDICTIONS_FILE, index=False)

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["model"].feature_importances_
    importance_frame = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )
    importance_frame.to_csv(FEATURES_FILE, index=False)

    metrics = {
        "dataset_rows": int(df.shape[0]),
        "dataset_columns": int(df.shape[1]),
        "unique_sellers": int(df["Seller"].nunique()),
        "unique_bidders": int(df["Bidder"].nunique()),
        "weight_mean": float(y.mean()),
        "weight_median": float(y.median()),
        "weight_max": int(y.max()),
        "baseline_cv_mae": round(float(baseline_mae), 4),
        "model_cv_mae": round(float(model_mae), 4),
        "model_cv_r2": round(float(model_r2), 4),
        "test_mae": round(float(test_mae), 4),
        "test_rmse": round(float(test_rmse), 4),
        "test_r2": round(float(test_r2), 4),
    }

    with METRICS_FILE.open("w", encoding="utf-8") as metrics_handle:
        json.dump(metrics, metrics_handle, indent=2)

    print("Model training complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to: {MODEL_FILE}")
    print(f"Saved test predictions to: {PREDICTIONS_FILE}")
    print(f"Saved top feature importances to: {FEATURES_FILE}")


if __name__ == "__main__":
    main()
