
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import joblib
import os

def rmsle(y_true, y_pred):
    """
    Compute RMSLE (Root Mean Squared Logarithmic Error)
    as per the competition's requirements.
    """
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

def main():
    # 1. Load train.csv
    train_data_path = os.path.join("data", "house_pricing", "train.csv")
    df = pd.read_csv(train_data_path)
    
    # 2. Data Preprocessing:
    # ------------------------------------------------
    # For simplicity, let's do a minimal approach:
    #   - Fill numerical NaNs with mean
    #   - Fill categorical NaNs with mode
    #   - Drop some obviously problematic columns if needed
    #   - Transform target using log1p for training

    # Separate features and target
    y = df["SalePrice"].values
    X = df.drop(["Id", "SalePrice"], axis=1)

    # Identify numeric and categorical columns quickly
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns

    # Fill numeric columns with mean
    for c in numeric_cols:
        X[c].fillna(X[c].mean(), inplace=True)

    # Fill categorical columns with mode
    for c in cat_cols:
        X[c].fillna(X[c].mode()[0], inplace=True)

    # Convert categorical columns to one-hot
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # 3. Train/test split for local validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42, test_size=0.2)

    # 4. Build RandomForestRegressor model
    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        n_jobs=-1
    )

    # 5. Train model
    rf.fit(X_train, y_train)

    # 6. Evaluate using validation data
    y_pred_val = rf.predict(X_val)
    score_val = rmsle(y_val, y_pred_val)
    print(f"Validation RMSLE: {score_val:.4f}")

    # 7. Cross-validation (optional, but let's do a quick check):
    scores = cross_val_score(rf, X, y, cv=5, scoring=None)  # We'll do a standard scoring, then handle RMSLE
    # Cross_val_score uses R^2 by default. Let's calculate RMSLE manually:
    cross_rmsle = []
    for train_idx, test_idx in [* ( (t, v) for (t,v) in  list( (tr,ts) for tr,ts in [] ) )]:
        pass
    # Actually let's do a direct approach:
    # We'll do a custom cross_val so we can produce RMSLE:

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmsle_scores = []
    for train_indices, test_indices in kf.split(X):
        X_tr, X_te = X.iloc[train_indices], X.iloc[test_indices]
        y_tr, y_te = y[train_indices], y[test_indices]
        model_temp = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            n_jobs=-1
        )
        model_temp.fit(X_tr, y_tr)
        preds_te = model_temp.predict(X_te)
        score = rmsle(y_te, preds_te)
        rmsle_scores.append(score)
    cv_rmsle_mean = np.mean(rmsle_scores)
    print(f"Cross-Validation RMSLE: {cv_rmsle_mean:.4f} across {len(rmsle_scores)} folds")

    # 8. Save model and columns
    joblib.dump(rf, os.path.join("solution", "model.joblib"))
    # also store the list of columns for inference
    joblib.dump(list(X.columns), os.path.join("solution", "columns.joblib"))

if __name__ == "__main__":
    main()
