import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

def save_plot_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('latin-1')
    plt.close(fig)
    return image_base64

def run_day2_fluctuations(df):
    df['Consumption Volatility'] = df['Energy Consumption (kWh)'].rolling(window=168).std()
    plots = {}
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['Energy Consumption (kWh)'], label='Energy Consumption', color='#66ccff', linewidth=1)
    ax.set_title('Building Energy Consumption Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Energy Consumption (kWh)')
    ax.grid(True, linestyle='--')
    plots['fluctuation_plot_1'] = save_plot_to_base64(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['Consumption Volatility'], label='7-Day Volatility', color='#ff6666', linewidth=2)
    ax.set_title('7-Day Rolling Volatility of Energy Consumption')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility (Standard Deviation)')
    ax.grid(True, linestyle='--')
    plots['fluctuation_plot_2'] = save_plot_to_base64(fig)
    return plots, df