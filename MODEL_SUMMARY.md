# Swarovski eBay Weight Prediction

## Project goal

This project uses the `eBayNetwork.csv` dataset to predict `Weight`, which looks like the strength of the connection or transaction count between a seller and bidder in the network data.

## Dataset used

- Rows: 200
- Columns: 5
- Predictors: `Seller`, `Bidder`, `Bidder.Volume`, `Seller.Volume`
- Target: `Weight`

## Why this target was chosen

The CSV does not contain price, date, product title, or item condition, so the only sensible prediction target already present in the dataset is `Weight`.

## Model approach

The script `train_prediction_model.py`:

1. loads the CSV
2. one-hot encodes `Seller` and `Bidder`
3. scales the volume fields
4. trains a `GradientBoostingRegressor`
5. compares it with a simple median baseline
6. saves metrics, predictions, feature importance, and the trained model

## Important limitation

This dataset is very small and heavily skewed toward low `Weight` values. Because it only includes IDs and volume counts, the model can make a rough estimate, but it is not strong enough to claim accurate business forecasting.

## Results from the trained model

- Baseline cross-validation MAE: `1.9650`
- Model cross-validation MAE: `1.7939`
- Model cross-validation R2: `0.0456`
- Test MAE: `1.6655`
- Test RMSE: `2.5625`
- Test R2: `0.0847`

## Interpretation

The model performs slightly better than a simple baseline, so it does learn some useful pattern from the dataset. However, the `R2` score is still low, which means most of the variation in `Weight` cannot be explained by the available columns.

## Main finding

The strongest signals came mostly from specific bidder IDs plus the two volume columns. That suggests the current CSV behaves more like network interaction data than a rich product-pricing dataset.

## How to run

```powershell
py train_prediction_model.py
```

## Outputs

After running the script, the project will contain:

- `model_artifacts/model_metrics.json`
- `model_artifacts/test_predictions.csv`
- `model_artifacts/top_feature_importance.csv`
- `model_artifacts/weight_prediction_model.pkl`
