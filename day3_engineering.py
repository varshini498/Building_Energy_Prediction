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
    image_base64 = base64.b64encode(buffer.read()).decode('latin-1')
    plt.close(fig)
    return image_base64

def run_day3_engineering(df):
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
    categorical_cols = ['Building Type', 'Building Status']
    for col in categorical_cols:
        if col in df.columns and df[col].dtype == 'object':
            df = pd.get_dummies(df, columns=[col], prefix=col, dummy_na=False)
    df.dropna(inplace=True)
    features_to_rank = [
        'Temperature (°C)', 'Humidity (%)', 'Occupancy Rate (%)', 'Lighting Consumption (kWh)', 
        'HVAC Consumption (kWh)', 'Energy Price ($/kWh)', 'Solar Irradiance (W/m²)',
        'lag_24h', 'lag_168h', 'rolling_mean_24h', 'sin_hour', 'cos_hour', 'sin_day', 'cos_day'
    ]
    one_hot_features = [col for col in df.columns if 'Building Type_' in col or 'Building Status_' in col]
    features_to_rank.extend(one_hot_features)
    available_features = [f for f in features_to_rank if f in df.columns]
    X = df[available_features]
    y = df['target']
    
    # --- OPTIMIZED FOR SPEED ---
    model = RandomForestRegressor(n_estimators=20, max_depth=10, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    
    feature_importance_df = pd.DataFrame({'feature': available_features, 'importance': importances}).sort_values(by='importance', ascending=False)
    
    plot = {}
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15), palette='viridis', ax=ax)
    ax.set_title('Feature Importance Ranking')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    plot['feature_importance_plot'] = save_plot_to_base64(fig)
    
    return feature_importance_df.head(10).to_markdown(index=False), plot, df