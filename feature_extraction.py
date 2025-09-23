import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

def perform_feature_extraction(file_path):
    """
    Performs feature extraction and visualization.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        print("Please ensure you have run the Day 3 script to create this file.")
        return None

    print("### 🤖 Separate Feature Extraction Code")
    
    # Load the engineered dataset from Day 3
    try:
        df = pd.read_csv(file_path, index_col='Timestamp', parse_dates=True)
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None
        
    # Drop rows with NaN values if any
    df.dropna(inplace=True)
    
    # --- PCA for Visualization (Feature Extraction) ---
    print("\n#### 📊 PCA for Dataset Visualization")
    
    # Define a list of features to use for PCA
    pca_features = [
        'Temperature (°C)', 'Humidity (%)', 'Occupancy Rate (%)',
        'Lighting Consumption (kWh)', 'HVAC Consumption (kWh)',
        'Energy Price ($/kWh)', 'Solar Irradiance (W/m²)',
        'lag_24h', 'lag_168h', 'rolling_mean_24h',
        'sin_hour', 'cos_hour', 'sin_day', 'cos_day'
    ]
    
    # Scale the data before applying PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[pca_features])
    
    # Apply PCA to reduce to 2 components
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    
    # Create the PCA DataFrame with the correct index
    principal_df = pd.DataFrame(data=principal_components, index=df.index, columns=['PC1', 'PC2'])
    
    # Get the building types for coloring the plot
    principal_df['Building Type'] = df['Building Type']
    
    # Create the scatter plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Building Type', data=principal_df, palette='viridis', s=20)
    plt.title('2D PCA Visualization of Extracted Features', fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.grid(True, linestyle='--')
    plt.legend(title='Building Type')
    plt.tight_layout()
    plt.savefig('pca_feature_extraction.png')
    plt.show()
    
    # Save the dataframe with new features
    output_file_path = 'extracted_features_building_data.csv'
    df.to_csv(output_file_path)
    print(f"\nFinal dataset with extracted features saved to '{output_file_path}'.")

# To run the script:
if __name__ == "__main__":
    file_for_extraction = 'engineered_building_data.csv'
    perform_feature_extraction(file_for_extraction)