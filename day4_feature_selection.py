import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def select_and_visualize_features(file_path):
    """
    Performs feature selection using correlation analysis and visualizes the dataset with PCA.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        print("Please ensure you have run the Day 3 script to create this file.")
        return None

    print("### 🤖 Day 4: The Machine Learning Pipeline (Feature Selection)")
    
    # Load the engineered dataset from Day 3
    try:
        df = pd.read_csv(file_path, index_col='Timestamp', parse_dates=True)
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None
        
    # Drop the original non-numeric and unneeded columns
    df_numeric = df.select_dtypes(include=np.number)
    
    # Exclude the target variable from correlation analysis
    features_for_corr = df_numeric.drop(columns=['Energy Consumption (kWh)', 'Energy Consumption (kWh)_normalized', 'Consumption Volatility', 'Peak Demand Reduction Indicator', 'Power Outage Indicator', 'Maintenance Status', 'Electric Vehicle Charging Status', 'Demand Response Participation'])
    
    print("\n#### 🔍 Correlation Analysis & Feature Selection")
    # Calculate the correlation matrix
    correlation_matrix = features_for_corr.corr()

    # Identify highly correlated features
    highly_correlated_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > 0.9:
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                highly_correlated_pairs.append((col1, col2, correlation_matrix.iloc[i, j]))

    if highly_correlated_pairs:
        print("Highly correlated feature pairs found (correlation > 0.9):")
        for pair in highly_correlated_pairs:
            print(f"- {pair[0]} and {pair[1]} with a correlation of {pair[2]:.2f}")
    else:
        print("No highly correlated feature pairs found.")

    # Final list of selected features for the model
    selected_features = features_for_corr.columns.tolist()
    
    # --- PCA for Visualization ---
    print("\n#### 📊 PCA for Dataset Visualization")
    
    # Drop NaN values from the relevant columns to ensure consistent indexing
    df.dropna(subset=selected_features, inplace=True)
    
    # Scale the data before applying PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[selected_features])
    
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
    plt.title('2D PCA Visualization of Building Energy Data', fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.grid(True, linestyle='--')
    plt.legend(title='Building Type')
    plt.tight_layout()
    plt.savefig('pca_scatter_plot.png')
    plt.show()

    # Save the dataframe with selected features
    final_features_df = df[['Energy Consumption (kWh)_normalized'] + selected_features]
    output_file_path = 'selected_features_building_data.csv'
    final_features_df.to_csv(output_file_path)
    print(f"\nFinal dataset with selected features saved to '{output_file_path}'.")

# To run the script:
if __name__ == "__main__":
    file_for_day4 = 'engineered_building_data.csv'
    select_and_visualize_features(file_for_day4)