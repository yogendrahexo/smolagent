import pandas as pd                                                                                                                        
import numpy as np                                                                                                                         
from joblib import load                                                                                                                    
from sklearn.metrics import mean_squared_error                                                                                             
                                                                                                                                            
def rmsle(y_true, y_pred):                                                                                                                 
    """Compute RMSLE."""                                                                                                                   
    import numpy as np                                                                                                                     
    from sklearn.metrics import mean_squared_error                                                                                         
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))                                                                 
                                                                                                                                            
def inference():                                                                                                                           
    # Load pipeline                                                                                                                        
    pipeline = load("solution/restricted_house_pricing/model_gbr.joblib")                                                                  
                                                                                                                                            
    # For local evaluation on full train (to see performance):                                                                             
    train_df = pd.read_csv("data/house_pricing/train.csv")                                                                                 
    if "Id" in train_df.columns:                                                                                                           
        train_df.drop("Id", axis=1, inplace=True)                                                                                          
    y_train = train_df["SalePrice"]                                                                                                        
    X_train = train_df.drop(["SalePrice"], axis=1)                                                                                         
                                                                                                                                            
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns                                                                      
    X_train = X_train[numeric_cols].copy()                                                                                                 
    X_train.fillna(X_train.mean(), inplace=True)                                                                                           
                                                                                                                                            
    train_preds = pipeline.predict(X_train)                                                                                                
    train_score = rmsle(y_train, train_preds)                                                                                              
    print(f"[inference.py] Full train RMSLE: {train_score:.5f}")                                                                           
                                                                                                                                            
    # Predict on test data                                                                                                                 
    test_df = pd.read_csv("data/house_pricing/test.csv")                                                                                   
    test_ids = test_df["Id"]                                                                                                               
    test_numeric = test_df.select_dtypes(include=[np.number]).copy()                                                                       
    test_numeric.fillna(test_numeric.mean(), inplace=True)                                                                                 
                                                                                                                                            
    # Ensure alignment with training features                                                                                              
    for col in numeric_cols:                                                                                                               
        if col not in test_numeric.columns:                                                                                                
            test_numeric[col] = 0                                                                                                          
    test_numeric = test_numeric[numeric_cols]                                                                                              
                                                                                                                                            
    predictions_test = pipeline.predict(test_numeric)                                                                                      
                                                                                                                                            
    # Save submission                                                                                                                      
    submission = pd.DataFrame({"Id": test_ids, "SalePrice": predictions_test})                                                             
    submission.to_csv("solution/restricted_house_pricing/submission_gbr.csv", index=False)                                                 
    print("[inference.py] Created submission_gbr.csv")                                                                                     
                                                                                                                                            
# Call inference() directly                                                                                                                
inference()     