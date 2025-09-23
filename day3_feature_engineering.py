import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # <--- This line was missing
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from IPython.display import display, Markdown

# Set plotting style
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#2b2b2b'
plt.rcParams['axes.facecolor'] = '#2b2b2b'
plt.rcParams['grid.color'] = '#444444'
plt.rcParams['text.color'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['lines.color'] = 'cyan'
plt.rcParams['patch.edgecolor'] = 'white'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

def create_and_rank_features(file_path):
    """
    Creates new features and ranks them by importance using a Random Forest model.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        print("Please ensure you have run the Day 2 script to create this file.")
        return None

    print("### 🤖 Day 3: The Machine Learning Pipeline (Feature Engineering)")
    
    # Load the cleaned dataset from Day 2
    try:
        df = pd.read_csv(file_path, index_col='Timestamp', parse_dates=True)
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None
        
    print("\n#### ⚙️ Feature Creation")
    # We will use the normalized energy consumption for features
    df['target'] = df['Energy Consumption (kWh)_normalized']

    # --- Lag Features ---
    # Consumption from 24 hours ago
    df['lag_24h'] = df['target'].shift(24)
    # Consumption from 7 days ago (168 hours)
    df['lag_168h'] = df['target'].shift(168)
    
    # --- Rolling Features ---
    # Average consumption over the last 24 hours
    df['rolling_mean_24h'] = df['target'].rolling(window=24).mean()
    
    # --- Time-based Features (for seasonality) ---
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    # --- Fourier Features (advanced seasonality) ---
    print("Creating Fourier Features...")
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_day'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['cos_day'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    
    # Drop rows with NaN values created by shifting
    df.dropna(inplace=True)
    print("New features created and rows with NaN values dropped.")
    
    # --- Feature Importance Ranking with Random Forest ---
    print("\n#### 📊 Feature Importance Ranking")
    
    # Define features and target
    features = [
        'Temperature (°C)', 'Humidity (%)', 'Occupancy Rate (%)', 
        'Lighting Consumption (kWh)', 'HVAC Consumption (kWh)', 
        'Energy Price ($/kWh)', 'Solar Irradiance (W/m²)',
        'lag_24h', 'lag_168h', 'rolling_mean_24h',
        'sin_hour', 'cos_hour', 'sin_day', 'cos_day'
    ]
    
    X = df[features]
    y = df['target']
    
    # Normalize features for consistent model training
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize and train a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Get feature importances
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': features, 'importance': importances}).sort_values(by='importance', ascending=False)
    
    print("Top 10 most important features:")
    print(feature_importance_df.head(10).to_markdown(index=False, numalign="left", stralign="left"))
    
    # Plot feature importances
    print("\n#### 📈 Feature Importance Bar Chart")
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15), palette='viridis')
    plt.title('Feature Importance Ranking for Energy Consumption Prediction', fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance_ranking.png')
    plt.show()
    
    # Save the dataframe with new features
    output_file_path = 'engineered_building_data.csv'
    df.to_csv(output_file_path)
    print(f"\nUpdated data saved to '{output_file_path}'.")
    
    return df

# To run the script:
if __name__ == "__main__":
    file_for_day3 = 'cleaned_building_data_with_volatility.csv'
    create_and_rank_features(file_for_day3)