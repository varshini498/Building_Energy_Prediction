import pandas as pd
import joblib

def run_day8_residual_calculation(file_path):
    df = pd.read_csv(file_path, index_col='Timestamp', parse_dates=True)
    df.dropna(inplace=True)
    
    features = [col for col in df.columns if col not in ['Energy Consumption (kWh)_normalized', 'Energy Consumption (kWh)', 'Consumption Volatility']]
    X = df[features]
    y = df['Energy Consumption (kWh)_normalized']
    
    optimized_model = joblib.load('optimized_model.pkl')
    y_pred_full = optimized_model.predict(X)
    df['Residuals'] = y - y_pred_full
    
    output_file = 'residuals_data.csv'
    df.to_csv(output_file)
    
    return output_file