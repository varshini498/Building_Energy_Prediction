import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.keras.models import load_model

def save_plot_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('latin-1')
    plt.close(fig)
    return image_base64

def run_day10_wastage_analysis(df):
    results = {}
    
    # --- Data Setup ---
    if 'Energy Consumption (kWh)_normalized' not in df.columns:
        df['Energy Consumption (kWh)_normalized'] = df['Energy Consumption (kWh)'] / df['Energy Consumption (kWh)'].max()

    if 'Residuals' not in df.columns:
        df['Residuals'] = 0 
    
    # Check if the Is_Anomaly column exists. If not, create a default column.
    if 'Is_Anomaly' not in df.columns:
        df['Is_Anomaly'] = 1 # Default all to non-anomaly (1)

    # Define thresholds
    high_usage_threshold = df['Energy Consumption (kWh)_normalized'].quantile(0.95)
    low_usage_threshold = df['Energy Consumption (kWh)_normalized'].quantile(0.05)
    
    high_usage_events_df = df[df['Energy Consumption (kWh)_normalized'] > high_usage_threshold]
    low_usage_events_df = df[df['Energy Consumption (kWh)_normalized'] < low_usage_threshold]
    
    # Wastage Events (from high positive residuals)
    wastage_events_df = df[(df['Is_Anomaly'] == -1) & (df['Residuals'] > 0)]
    
    # --- Analysis & Counts ---
    results['high_usage_count'] = len(high_usage_events_df)
    results['low_usage_count'] = len(low_usage_events_df)
    results['wastage_count'] = len(wastage_events_df)
    
    # Total Energy Wastage Value (using actual kWh)
    total_wastage_kWh = wastage_events_df['Energy Consumption (kWh)'].sum()
    results['total_wastage_value'] = total_wastage_kWh

    # Extracting timestamps for display
    results['high_usage_times'] = high_usage_events_df.index.strftime('%Y-%m-%d %H:%M').tolist()[:5]
    results['low_usage_times'] = low_usage_events_df.index.strftime('%Y-%m-%d %H:%M').tolist()[:5]

    # --- PLOT 1: Consumption Distribution with Thresholds (Histogram) ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Energy Consumption (kWh)_normalized'], bins=50, kde=True, color='#66ccff', ax=ax1)
    ax1.axvline(high_usage_threshold, color='#ff6666', linestyle='--', label='High Usage Threshold')
    ax1.axvline(low_usage_threshold, color='#ffcc66', linestyle='--', label='Low Usage Threshold')
    ax1.set_title('Normalized Energy Consumption Distribution')
    ax1.legend()
    results['consumption_distribution_plot'] = save_plot_to_base64(fig1)

    # --- PLOT 3: Wastage Residuals Distribution ---
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Residuals'], bins=50, kde=True, color='#b16286', ax=ax3)
    if not wastage_events_df.empty:
        wastage_threshold = wastage_events_df['Residuals'].min()
        ax3.axvline(wastage_threshold, color='#ff6666', linestyle='--', label=f'Wastage Threshold (>{wastage_threshold:.4f})')
    ax3.set_title('Distribution of Prediction Residuals (Wastage)')
    ax3.set_xlabel('Residuals (Normalized kWh)')
    ax3.legend()
    results['wastage_residual_plot'] = save_plot_to_base64(fig3)

    # 2. Final Report & Comparison (Model Evaluation)
    X_rf = df.select_dtypes(include=['number'])
    y_rf = df['Energy Consumption (kWh)_normalized']
    X_rf = X_rf.drop(columns=['Energy Consumption (kWh)_normalized', 'Energy Consumption (kWh)', 
                              'Consumption Volatility', 'target', 'Residuals', 'Is_Anomaly'], errors='ignore')
    X_rf.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_rf.dropna(inplace=True)
    y_rf = y_rf.loc[X_rf.index]

    model_names, r2_scores, mae_scores = [], [], []

    if not X_rf.empty:
        # Evaluate Base Random Forest
        try:
            rfr_base = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
            rfr_base.fit(X_rf, y_rf)
            y_pred_rfr = rfr_base.predict(X_rf)
            model_names.append('RandomForest (Base)')
            r2_scores.append(r2_score(y_rf, y_pred_rfr))
            mae_scores.append(mean_absolute_error(y_rf, y_pred_rfr))
        except Exception: pass
        
        # Evaluate Optimized RandomForest
        try:
            optimized_model = joblib.load('optimized_model.pkl')
            y_pred_rfr_optimized = optimized_model.predict(X_rf)
            model_names.append('RandomForest (Optimized)')
            r2_scores.append(r2_score(y_rf, y_pred_rfr_optimized))
            mae_scores.append(mean_absolute_error(y_rf, y_pred_rfr_optimized))
        except (FileNotFoundError, Exception): pass

        # Evaluate Hybrid LSTM Model
        try:
            hybrid_model = load_model('hybrid_model.h5')
            sequence_features = ['lag_24h', 'lag_168h', 'rolling_mean_24h']
            static_features = [col for col in X_rf.columns if col not in sequence_features]
            scaler_static = StandardScaler()
            X_static = scaler_static.fit_transform(X_rf[static_features])
            X_sequence = np.reshape(X_rf[sequence_features].values, (X_rf.shape[0], 1, len(sequence_features)))
            y_pred_hybrid = hybrid_model.predict([X_sequence, X_static])
            model_names.append('Hybrid LSTM')
            r2_scores.append(r2_score(y_rf, y_pred_hybrid))
            mae_scores.append(mean_absolute_error(y_rf, y_pred_hybrid))
        except (FileNotFoundError, Exception): pass
    
    comparison_table = pd.DataFrame({'Model': model_names, 'R² Score': r2_scores, 'MAE': mae_scores})

    if not comparison_table.empty:
        best_r2_model = comparison_table.loc[comparison_table['R² Score'].idxmax()]
        best_mae_model = comparison_table.loc[comparison_table['MAE'].idxmin()]
        results['comparison_table'] = comparison_table.to_markdown(index=False)
        results['best_r2_model_name'] = best_r2_model['Model']
        results['best_r2_score'] = best_r2_model['R² Score']
        results['best_mae_model_name'] = best_mae_model['Model']
        results['best_mae_score'] = best_mae_model['MAE']
    else:
        results['comparison_table'] = "Not enough data/models trained for comparison."
        results['best_r2_model_name'] = "N/A"
        results['best_r2_score'] = "N/A"
        results['best_mae_model_name'] = "N/A"
        results['best_mae_score'] = "N/A"

    return results, df