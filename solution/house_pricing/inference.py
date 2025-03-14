
import pandas as pd
import numpy as np
import joblib
import os

def main():
    # 1. Load artifacts
    model_path = os.path.join("solution", "model.joblib")
    columns_path = os.path.join("solution", "columns.joblib")
    rf = joblib.load(model_path)
    train_columns = joblib.load(columns_path)
    
    # 2. Load test.csv
    test_data_path = os.path.join("data", "house_pricing", "test.csv")
    df_test = pd.read_csv(test_data_path)
    
    # Keep the Id for submission
    test_ids = df_test["Id"].values
    
    # 3. Perform same data preprocessing
    # We'll fill numeric columns with mean, categoric with mode,
    # then one-hot, aligning with train_columns by reindexing if needed.
    
    X_test = df_test.drop(["Id"], axis=1)
    
    # Identify numeric and categorical columns quickly
    numeric_cols = X_test.select_dtypes(include=[np.number]).columns
    cat_cols = X_test.select_dtypes(exclude=[np.number]).columns
    
    for c in numeric_cols:
        X_test[c].fillna(X_test[c].mean(), inplace=True)
    for c in cat_cols:
        X_test[c].fillna(X_test[c].mode()[0], inplace=True)
    
    # Get dummies
    X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
    
    # Now align to the same columns as training
    for col in train_columns:
        if col not in X_test.columns:
            X_test[col] = 0
    # If there are any extra columns in X_test that were not in training, drop them
    X_test = X_test[train_columns]
    
    # 4. Predictions
    preds = rf.predict(X_test)
    
    # 5. Create submission file in the solution directory
    sub_path = os.path.join("solution", "submission.csv")
    submission = pd.DataFrame({"Id": test_ids, "SalePrice": preds})
    submission.to_csv(sub_path, index=False)
    print("Submission saved at:", sub_path)
    
    # 6. Print a dummy local evaluation score message (real test data doesn't have SalePrice).
    # Just to confirm it runs:
    print("Inference completed successfully.")

if __name__ == "__main__":
    main()
