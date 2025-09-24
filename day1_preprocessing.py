import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def run_day1_preprocessing(df):
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    duplicates_removed = initial_rows - len(df)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', utc=True)
    df.dropna(subset=['Timestamp'], inplace=True)
    df.set_index('Timestamp', inplace=True)
    df.sort_index(inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    scaler = MinMaxScaler()
    df['Energy Consumption (kWh)_normalized'] = scaler.fit_transform(df[['Energy Consumption (kWh)']])
    return duplicates_removed, df