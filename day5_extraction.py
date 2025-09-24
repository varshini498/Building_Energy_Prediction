import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import io
import base64

def save_plot_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    return image_base64

def run_day5_extraction(file_path):
    df = pd.read_csv(file_path, index_col='Timestamp', parse_dates=True)
    df.dropna(inplace=True)
    
    pca_features = [
        'Temperature (°C)', 'Humidity (%)', 'Occupancy Rate (%)',
        'Lighting Consumption (kWh)', 'HVAC Consumption (kWh)',
        'Energy Price ($/kWh)', 'Solar Irradiance (W/m²)',
        'lag_24h', 'lag_168h', 'rolling_mean_24h',
        'sin_hour', 'cos_hour', 'sin_day', 'cos_day'
    ]
    
    df_pca_subset = df[pca_features]
    X_scaled = StandardScaler().fit_transform(df_pca_subset)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    
    principal_df = pd.DataFrame(data=principal_components, index=df.index, columns=['PC1', 'PC2'])
    principal_df['Building Type'] = df['Building Type']
    
    plot = {}
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='Building Type', data=principal_df, palette='viridis', s=20, ax=ax)
    ax.set_title('2D PCA Visualization')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    plot['pca_plot'] = save_plot_to_base64(fig)
    
    output_file = 'extracted_data.csv'
    df.to_csv(output_file)
    
    return plot, output_file