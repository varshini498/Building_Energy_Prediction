import pandas as pd
import numpy as np
import joblib

def run_day8_residual_calculation(df):
    df.dropna(inplace=True)
    
    # Load the optimized model
    try:
        optimized_model = joblib.load('optimized_model.pkl')
    except FileNotFoundError:
        print("Warning: Optimized model not found. Skipping residual calculation.")
        df['Residuals'] = 0
        return df

    # --- FIX: Create the prediction DataFrame (X) with the correct features ---
    # We must remove the same columns that were removed during the training in day6
    X = df.select_dtypes(include=['number'])
    X = X.drop(columns=['Energy Consumption (kWh)_normalized', 'Energy Consumption (kWh)', 'Consumption Volatility', 'target', 'Residuals', 'Is_Anomaly'], errors='ignore')
    
    # Clean up any potential inf or nan values that may have been created
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)

    y = df['Energy Consumption (kWh)_normalized']
    
    # Align the target variable (y) with the cleaned feature set (X)
    y = y.loc[X.index]

    # Make the prediction with the cleaned feature set
    y_pred_full = optimized_model.predict(X)
    
    # Calculate residuals and add them to the original DataFrame
    # Note: We need to align the residuals with the original DataFrame index
    residuals_df = pd.DataFrame(y - y_pred_full, index=y.index, columns=['Residuals'])
    df = df.merge(residuals_df, left_index=True, right_index=True)
    
    return df