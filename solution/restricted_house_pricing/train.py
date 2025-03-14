import pandas as pd                                                                                                                        
import numpy as np                                                                                                                         
from sklearn.model_selection import train_test_split                                                                                       
from sklearn.preprocessing import StandardScaler                                                                                           
from sklearn.pipeline import Pipeline                                                                                                      
from sklearn.metrics import mean_squared_error                                                                                             
from sklearn.ensemble import GradientBoostingRegressor                                                                                     
from joblib import dump                                                                                                                    
                                                                                                                                            
def rmsle(y_true, y_pred):                                                                                                                 
    """Compute the RMSLE (Root Mean Squared Log Error)."""                                                                                 
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))                                                                 
                                                                                                                                            
def train():                                                                                                                               
    # 1. Read training data                                                                                                                
    train_df = pd.read_csv("data/house_pricing/train.csv")                                                                                 
                                                                                                                                            
    # 2. Drop ID (if exists) and handle missing target                                                                                     
    if "Id" in train_df.columns:                                                                                                           
        train_df.drop("Id", axis=1, inplace=True)                                                                                          
    train_df.dropna(axis=0, subset=["SalePrice"], inplace=True)                                                                            
                                                                                                                                            
    # 3. Separate features and target                                                                                                      
    y = train_df["SalePrice"]                                                                                                              
    X = train_df.drop(["SalePrice"], axis=1)                                                                                               
                                                                                                                                            
    # Only keep numeric columns                                                                                                            
    numeric_cols = X.select_dtypes(include=[np.number]).columns                                                                            
    X = X[numeric_cols].copy()                                                                                                             
    X.fillna(X.mean(), inplace=True)                                                                                                       
                                                                                                                                            
    # 4. Train-valid split                                                                                                                 
    X_train, X_val, y_train, y_val = train_test_split(                                                                                     
        X, y, test_size=0.2, random_state=999                                                                                              
    )                                                                                                                                      
                                                                                                                                            
    # 5. Build pipeline with StandardScaler + GradientBoostingRegressor                                                                    
    pipeline = Pipeline([                                                                                                                  
        ("scaler", StandardScaler()),                                                                                                      
        ("gbr", GradientBoostingRegressor(                                                                                                 
            n_estimators=200,                                                                                                              
            learning_rate=0.05,                                                                                                            
            random_state=999                                                                                                               
        ))                                                                                                                                 
    ])                                                                                                                                     
                                                                                                                                            
    # 6. Train and evaluate                                                                                                                
    pipeline.fit(X_train, y_train)                                                                                                         
    val_preds = pipeline.predict(X_val)                                                                                                    
    val_score = rmsle(y_val, val_preds)                                                                                                    
    print(f"[train.py] Validation RMSLE: {val_score:.5f}")                                                                                 
                                                                                                                                            
    # 7. Save pipeline                                                                                                                     
    dump(pipeline, "solution/restricted_house_pricing/model_gbr.joblib")                                                                   
                                                                                                                                            
# Call train() directly                                                                                                                    
train() 