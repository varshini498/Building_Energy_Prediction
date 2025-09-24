import pandas as pd
import numpy as np

def run_day10_wastage_analysis(file_path):
    df = pd.read_csv(file_path, index_col='Timestamp', parse_dates=True)
    df.dropna(inplace=True)
    
    wastage_count = len(df[df['Residuals'] > df['Residuals'].quantile(0.99)])
    high_usage_count = len(df[df['Energy Consumption (kWh)_normalized'] > df['Energy Consumption (kWh)_normalized'].quantile(0.95)])
    low_usage_count = len(df[df['Energy Consumption (kWh)_normalized'] < df['Energy Consumption (kWh)_normalized'].quantile(0.05)])
    
    results = {
        'wastage_count': wastage_count,
        'high_usage_count': high_usage_count,
        'low_usage_count': low_usage_count
    }
    
    return results