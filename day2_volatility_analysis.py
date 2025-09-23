import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_and_visualize_volatility(file_path):
    """
    Performs volatility analysis and visualizes energy consumption.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        print("Please ensure you have run the Day 1 script to create this file.")
        return None

    print("### 📈 Day 2: Visualizing the Unseen (Interactive Fluctuation Analysis)")
    
    # Load the cleaned dataset from Day 1
    try:
        df_cleaned = pd.read_csv(file_path, index_col='Timestamp', parse_dates=True)
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None
    
    # Set plotting style to a futuristic dark theme
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

    # 📊 Calculation of Volatility
    print("\n#### 📊 Calculation of Volatility")
    # Calculate the rolling standard deviation over a 7-day period (168 hours)
    df_cleaned['Consumption Volatility'] = df_cleaned['Energy Consumption (kWh)'].rolling(window=168, center=False).std()
    print("'Consumption Volatility' column calculated successfully.")

    # 📉 Energy Consumption Chart
    print("\n#### 📉 Energy Consumption Chart")
    plt.figure(figsize=(15, 7))
    plt.plot(df_cleaned.index, df_cleaned['Energy Consumption (kWh)'], label='Energy Consumption', color='#66ccff', linewidth=1)
    plt.title('Building Energy Consumption Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Energy Consumption (kWh)', fontsize=12)
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig('energy_consumption_chart.png')
    plt.show()

    # 🌊 7-Day Rolling Volatility Chart
    print("\n#### 🌊 7-Day Rolling Volatility Chart")
    plt.figure(figsize=(15, 7))
    plt.plot(df_cleaned.index, df_cleaned['Consumption Volatility'], label='7-Day Volatility', color='#ff6666', linewidth=2)
    plt.title('7-Day Rolling Volatility of Energy Consumption', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Volatility (Standard Deviation of kWh)', fontsize=12)
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig('rolling_volatility_chart.png')
    plt.show()

    # Save the dataframe with new column
    output_file_path = 'cleaned_building_data_with_volatility.csv'
    df_cleaned.to_csv(output_file_path)
    print(f"\nUpdated data saved to '{output_file_path}'.")

    print("\nData analysis for Day 2 is complete. Two charts have been generated and saved.")
    
    return df_cleaned

# To run the script:
if __name__ == "__main__":
    cleaned_file = 'cleaned_building_data.csv'
    analyze_and_visualize_volatility(cleaned_file)