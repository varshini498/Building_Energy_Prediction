import pandas as pd
import numpy as np

def run_day4_selection(df):
    df.dropna(inplace=True)
    df_numeric = df.select_dtypes(include=np.number)
    features_for_corr = df_numeric.drop(columns=[
        'Energy Consumption (kWh)', 'Energy Consumption (kWh)_normalized', 
        'Consumption Volatility', 'Peak Demand Reduction Indicator', 
        'Power Outage Indicator', 'Maintenance Status', 
        'Electric Vehicle Charging Status', 'Demand Response Participation',
        'target'
    ], errors='ignore')
    correlation_matrix = features_for_corr.corr()
    correlated_features = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > 0.9:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
    results = {
        'highly_correlated_count': len(correlated_features),
        'removed_features': list(correlated_features)
    }
    df_final = df.drop(columns=list(correlated_features), errors='ignore')
    return results, df_final