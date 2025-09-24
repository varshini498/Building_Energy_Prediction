import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.ensemble import IsolationForest

def save_plot_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('latin-1')
    plt.close(fig)
    return image_base64

def run_day9_anomaly_detection(df):
    df.dropna(inplace=True)
    results = {}
    
    # --- FIX: Check if the DataFrame is empty after dropping NaNs ---
    if df.empty:
        print("Warning: DataFrame is empty. Skipping anomaly detection.")
        return {'anomaly_plot': None, 'anomalies_found': 0}, df
        
    if 'Residuals' not in df.columns:
        print("Warning: Residuals not found for anomaly detection. Skipping.")
        return {'anomaly_plot': None, 'anomalies_found': 0}, df
        
    isolation_forest = IsolationForest(contamination=0.01, random_state=42)
    df['Is_Anomaly'] = isolation_forest.fit_predict(df[['Residuals']])
    anomalies = df[df['Is_Anomaly'] == -1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index, df['Residuals'], label='Residuals', color='#66ccff', linewidth=0.5)
    ax.scatter(anomalies.index, anomalies['Residuals'], color='#ff6666', s=50, label='Anomalies', zorder=5)
    ax.set_title('Prediction Errors with Anomalies Highlighted')
    ax.set_xlabel('Date')
    ax.set_ylabel('Residual (Normalized)')
    results['anomaly_plot'] = save_plot_to_base64(fig)
    results['anomalies_found'] = len(anomalies)
    
    return results, df