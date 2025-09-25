import pandas as pd
import numpy as np
import joblib

def run_day8_residual_calculation(df):
    # Drop NaNs to clean up data before using the model
    df.dropna(inplace=True) 
    
    try:
        optimized_model = joblib.load('optimized_model.pkl')
    except FileNotFoundError:
        print("Warning: Optimized model not found. Cannot calculate residuals.")
        df['Residuals'] = 0 
        df['Is_Anomaly'] = 1 
        return df

    # 1. Prepare Features (X): Must match the feature set used in Day 6 Optimization
    X = df.select_dtypes(include=['number'])
    X = X.drop(columns=['Energy Consumption (kWh)_normalized', 'Energy Consumption (kWh)', 'Consumption Volatility', 'target'], errors='ignore')
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)

    # 2. Prepare Target (y) and align with cleaned X
    y = df['Energy Consumption (kWh)_normalized']
    y = y.loc[X.index]

    # 3. Prediction and Calculation
    y_pred_full = optimized_model.predict(X)
    
    # Create a DataFrame for residuals aligned to the existing index
    residuals_df = pd.DataFrame(y - y_pred_full, index=y.index, columns=['Residuals'])
    
    # Merge the new Residuals column back into the main DataFrame
    df = df.merge(residuals_df, left_index=True, right_index=True, how='left')
    
    # Fill NaNs with 0 for rows that couldn't be predicted (so they don't break Day 9)
    df['Residuals'].fillna(0, inplace=True) 

    return df