import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def run_day1_preprocessing(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")

    try:
        df = pd.read_csv(file_path, encoding='latin1')
    except Exception as e:
        raise ValueError(f"An error occurred while loading the file: {e}")

    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    duplicates_removed = initial_rows - len(df)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', utc=True)
    df.set_index('Timestamp', inplace=True)
    df.sort_index(inplace=True)
    df.dropna(subset=['Timestamp'], inplace=True)
    df.fillna(method='ffill', inplace=True)

    scaler = MinMaxScaler()
    df['Energy Consumption (kWh)_normalized'] = scaler.fit_transform(df[['Energy Consumption (kWh)']])
    
    output_file = 'cleaned_data.csv'
    df.to_csv(output_file)
    
    return duplicates_removed, output_file