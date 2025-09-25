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
    # Only consider rows where residuals were successfully calculated
    df_clean = df.dropna(subset=['Residuals'])
    results = {}
    
    if df_clean.empty or len(df_clean) < 100:
        print(f"Warning: Insufficient data ({len(df_clean)} samples) for robust anomaly detection. Skipping model fit.")
        return {'anomaly_plot': None, 'anomalies_found': 0}, df
        
    # --- FIX: INCREASED CONTAMINATION TO 0.05 (5%) ---
    # This forces the model to flag the most extreme 5% of the data points as anomalies.
    isolation_forest = IsolationForest(contamination=0.05, random_state=42)
    
    # Fit and predict only on the cleaned residual data
    df_clean['Is_Anomaly'] = isolation_forest.fit_predict(df_clean[['Residuals']])
    anomalies = df_clean[df_clean['Is_Anomaly'] == -1]
    
    # --- Plot Generation ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_clean.index, df_clean['Residuals'], label='Residuals', color='#66ccff', linewidth=0.5)
    ax.scatter(anomalies.index, anomalies['Residuals'], color='#ff6666', s=50, label='Anomalies', zorder=5)
    ax.set_title('Prediction Errors with Anomalies Highlighted')
    ax.set_xlabel('Date')
    ax.set_ylabel('Residual (Normalized)')
    results['anomaly_plot'] = save_plot_to_base64(fig)
    results['anomalies_found'] = len(anomalies)
    
    # Update the original DataFrame with the anomaly labels
    df = df.merge(df_clean[['Is_Anomaly']], left_index=True, right_index=True, how='left')
    df['Is_Anomaly'].fillna(1, inplace=True) 
    
    return results, df