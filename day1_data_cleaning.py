import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def clean_and_normalize_data(file_path):
    """
    Performs data cleaning and normalization on the energy consumption dataset.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return None

    print("### 🛠️ Day 1: Building a Flawless Foundation (Data Cleaning)")
    
    # 📂 Dataset Loading & Initial Inspection
    print("\n#### 📂 Dataset Loading & Initial Inspection")
    try:
        # Try loading with different encodings
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            print("UnicodeDecodeError: Trying 'latin1' encoding...")
            df = pd.read_csv(file_path, encoding='latin1')
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None
        
    print("\nInitial Info:")
    df.info()

    # ⏳ Datetime Conversion & Data Cleaning
    print("\n#### ⏳ Datetime Conversion & Data Cleaning")
    # Deduplication
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    duplicate_rows = initial_rows - len(df)
    print(f"Number of duplicate rows removed: {duplicate_rows}")

    # Datetime Conversion
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)
    df.set_index('Timestamp', inplace=True)
    df.sort_index(inplace=True)
    print("'Timestamp' column converted to datetime and set as index.")
    
    # Check for missing values and fill
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\nMissing values detected. Filling with forward fill.")
        df.fillna(method='ffill', inplace=True)
    else:
        print("\nNo missing values detected.")
        
    print("\nFinal Info after cleaning:")
    df.info()
    
    # ✨ Normalization (Min-Max Scaling)
    print("\n#### ✨ Normalization (Min-Max Scaling)")
    scaler = MinMaxScaler()
    df['Energy Consumption (kWh)_normalized'] = scaler.fit_transform(df[['Energy Consumption (kWh)']])
    print("'Energy Consumption (kWh)' column normalized successfully.")
    
    # Save the cleaned DataFrame to a new CSV file
    cleaned_file_path = 'cleaned_building_data.csv'
    df.to_csv(cleaned_file_path)
    print(f"\nCleaned data saved to '{cleaned_file_path}'.")
    
    return df

# To run the script:
if __name__ == "__main__":
    # Make sure to provide the full file name, including the extension.
    dataset_file = 'building_dataset.csv'
    cleaned_df = clean_and_normalize_data(dataset_file)
    if cleaned_df is not None:
        print("\nData cleaning and preparation for Day 1 is complete!")