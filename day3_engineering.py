import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import io
import base64

def save_plot_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    return image_base64

def run_day3_engineering(file_path):
    df = pd.read_csv(file_path, index_col='Timestamp', parse_dates=True)
    df['target'] = df['Energy Consumption (kWh)_normalized']
    
    df['lag_24h'] = df['target'].shift(24)
    df['lag_168h'] = df['target'].shift(168)
    df['rolling_mean_24h'] = df['target'].rolling(window=24).mean()
    
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_day'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['cos_day'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    
    df.dropna(inplace=True)
    
    features_to_rank = [
        'Temperature (°C)', 'Humidity (%)', 'Occupancy Rate (%)', 'Lighting Consumption (kWh)', 
        'HVAC Consumption (kWh)', 'Energy Price ($/kWh)', 'Solar Irradiance (W/m²)',
        'lag_24h', 'lag_168h', 'rolling_mean_24h',
        'sin_hour', 'cos_hour', 'sin_day', 'cos_day'
    ]
    X = df[features_to_rank]
    y = df['target']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': features_to_rank, 'importance': importances}).sort_values(by='importance', ascending=False)
    
    plot = {}
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15), palette='viridis', ax=ax)
    ax.set_title('Feature Importance Ranking')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    plot['feature_importance_plot'] = save_plot_to_base64(fig)
    
    output_file = 'engineered_data.csv'
    df.to_csv(output_file)
    
    return feature_importance_df.head(10).to_markdown(index=False), plot, output_file