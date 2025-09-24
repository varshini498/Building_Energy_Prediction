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
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    return image_base64

def run_day9_anomaly_detection(file_path):
    df = pd.read_csv(file_path, index_col='Timestamp', parse_dates=True)
    df.dropna(inplace=True)
    
    results = {}
    
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
    
    output_file = 'anomaly_data.csv'
    df.to_csv(output_file)
    
    return results, output_file