import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

def save_plot_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('latin-1')
    plt.close(fig)
    return image_base64

def run_day9_anomaly_detection(df):
    df_clean = df.dropna(subset=['Residuals']).copy()

    # Optional: normalize residuals for more sensitive detection
    scaler = MinMaxScaler()
    df_clean['Residuals_Norm'] = scaler.fit_transform(df_clean[['Residuals']])

    results = {}

    if df_clean.empty or len(df_clean) < 100:
        print(f"Warning: Insufficient data ({len(df_clean)} samples) for robust anomaly detection. Skipping model fit.")
        return {'anomaly_plot': None, 'anomalies_found': 0}, df

    # Try higher contamination (5-10%)
    isolation_forest = IsolationForest(contamination=0.08, random_state=42)
    df_clean['Is_Anomaly'] = isolation_forest.fit_predict(df_clean[['Residuals_Norm']])
    anomalies = df_clean[df_clean['Is_Anomaly'] == -1]

    # --- Diagnostic Plot (shows all residuals, highlights anomalies) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_clean.index, df_clean['Residuals_Norm'], label='Residuals (Norm)', color='#66ccff', linewidth=0.6)
    ax.scatter(anomalies.index, anomalies['Residuals_Norm'], color='#ff3333', s=45, label='Anomaly', zorder=6)
    ax.set_title('Normalized Residual Errors With Detected Anomalies')
    ax.set_xlabel('Index')
    ax.set_ylabel('Normalized Residual')
    ax.legend()
    results['anomaly_plot'] = save_plot_to_base64(fig)
    results['anomalies_found'] = len(anomalies)

    # Merge anomaly labels back
    df['Is_Anomaly'] = 1  # Default as not-anomaly
    df.loc[df_clean.index, 'Is_Anomaly'] = df_clean['Is_Anomaly']

    return results, df
