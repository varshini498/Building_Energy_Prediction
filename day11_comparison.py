import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import tensorflow as tf
import os

def run_day11_comparison(df):
    results = {}

    # Ensure necessary columns are present
    df.dropna(inplace=True)
    X_full = df.select_dtypes(include=['number'])
    y_full = df['Energy Consumption (kWh)_normalized']
    
    # Remove columns that were dropped during training of RandomForest
    X_rf = X_full.drop(columns=['Energy Consumption (kWh)_normalized', 'Energy Consumption (kWh)', 'Consumption Volatility', 'target', 'Residuals', 'Is_Anomaly'], errors='ignore')
    X_rf.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_rf.dropna(inplace=True)
    y_rf = y_full.loc[X_rf.index] # Align target with X_rf

    # Ensure a consistent feature set for LSTM (if possible)
    # The LSTM part assumes a fixed set of sequence and static features
    sequence_features = ['lag_24h', 'lag_168h', 'rolling_mean_24h']
    
    # Dynamically find static features by excluding sequence features and target columns
    static_features = [col for col in X_full.columns if col not in sequence_features and col not in ['Energy Consumption (kWh)_normalized', 'Energy Consumption (kWh)', 'Consumption Volatility', 'target', 'Residuals', 'Is_Anomaly']]
    
    X_lstm = X_full[sequence_features + static_features] # Prepare X for LSTM
    X_lstm.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_lstm.dropna(inplace=True)
    y_lstm = y_full.loc[X_lstm.index] # Align target with X_lstm

    # Initialize lists to store model performance
    model_names = []
    r2_scores = []
    mae_scores = []

    # 1. Evaluate RandomForest (non-optimized, if available - for full comparison)
    # This assumes a model was saved in day6 before optimization or a default one
    try:
        # A simple non-optimized RF for comparison (re-train for comparison if not saved)
        from sklearn.ensemble import RandomForestRegressor
        rfr_base = RandomForestRegressor(n_estimators=50, random_state=42)
        rfr_base.fit(X_rf, y_rf)
        y_pred_rfr = rfr_base.predict(X_rf)
        model_names.append('RandomForest (Base)')
        r2_scores.append(r2_score(y_rf, y_pred_rfr))
        mae_scores.append(mean_absolute_error(y_rf, y_pred_rfr))
    except Exception as e:
        print(f"Could not evaluate Base RandomForest: {e}")

    # 2. Evaluate Optimized RandomForest
    try:
        optimized_model = joblib.load('optimized_model.pkl')
        y_pred_rfr_optimized = optimized_model.predict(X_rf)
        model_names.append('RandomForest (Optimized)')
        r2_scores.append(r2_score(y_rf, y_pred_rfr_optimized))
        mae_scores.append(mean_absolute_error(y_rf, y_pred_rfr_optimized))
    except FileNotFoundError:
        print("Optimized RandomForest model not found.")
    except Exception as e:
        print(f"Error evaluating Optimized RandomForest: {e}")

    # 3. Evaluate Hybrid LSTM Model
    try:
        # Load the Keras model
        hybrid_model = tf.keras.models.load_model('hybrid_model.h5')
        
        # Scale static features using a new scaler or re-use if possible
        from sklearn.preprocessing import StandardScaler
        scaler_static = StandardScaler()
        X_lstm_static = scaler_static.fit_transform(X_lstm[static_features])
        X_lstm_sequence = np.reshape(X_lstm[sequence_features].values, (X_lstm.shape[0], 1, len(sequence_features)))
        
        y_pred_hybrid = hybrid_model.predict([X_lstm_sequence, X_lstm_static])
        model_names.append('Hybrid LSTM')
        r2_scores.append(r2_score(y_lstm, y_pred_hybrid))
        mae_scores.append(mean_absolute_error(y_lstm, y_pred_hybrid))
    except FileNotFoundError:
        print("Hybrid LSTM model not found.")
    except Exception as e:
        print(f"Error evaluating Hybrid LSTM: {e}")

    # Create the comparison DataFrame
    comparison_table = pd.DataFrame({
        'Model': model_names,
        'R² Score': r2_scores,
        'MAE': mae_scores
    })

    results['comparison_table'] = comparison_table.to_markdown(index=False)
    
    # --- Advanced Output: Best Model Identification ---
    if not comparison_table.empty:
        best_r2_model = comparison_table.loc[comparison_table['R² Score'].idxmax()]
        best_mae_model = comparison_table.loc[comparison_table['MAE'].idxmin()]
        results['best_r2_model_name'] = best_r2_model['Model']
        results['best_r2_score'] = best_r2_model['R² Score']
        results['best_mae_model_name'] = best_mae_model['Model']
        results['best_mae_score'] = best_mae_model['MAE']
    else:
        results['best_r2_model_name'] = "N/A"
        results['best_r2_score'] = "N/A"
        results['best_mae_model_name'] = "N/A"
        results['best_mae_score'] = "N/A"

    return results, df