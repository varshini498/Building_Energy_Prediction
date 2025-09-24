import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

def save_plot_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('latin-1')
    plt.close(fig)
    return image_base64

def run_day10_wastage_analysis(df):
    results = {}
    
    # Ensure necessary columns are present after previous steps
    if 'Energy Consumption (kWh)_normalized' not in df.columns:
        print("Warning: 'Energy Consumption (kWh)_normalized' not found. Using raw consumption for thresholds.")
        df['Energy Consumption (kWh)_normalized'] = df['Energy Consumption (kWh)'] / df['Energy Consumption (kWh)'].max()

    if 'Residuals' not in df.columns:
        print("Warning: 'Residuals' not found. Cannot calculate wastage from anomalies.")
        df['Residuals'] = 0 # Default to no residuals

    # Define thresholds for high/low usage based on quantiles
    high_usage_threshold = df['Energy Consumption (kWh)_normalized'].quantile(0.95)
    low_usage_threshold = df['Energy Consumption (kWh)_normalized'].quantile(0.05)
    
    # High Usage Events
    high_usage_events_df = df[df['Energy Consumption (kWh)_normalized'] > high_usage_threshold]
    results['high_usage_count'] = len(high_usage_events_df)
    results['high_usage_details'] = high_usage_events_df[['Energy Consumption (kWh)', 'Energy Consumption (kWh)_normalized']].head().to_html()

    # Low Usage Events
    low_usage_events_df = df[df['Energy Consumption (kWh)_normalized'] < low_usage_threshold]
    results['low_usage_count'] = len(low_usage_events_df)
    results['low_usage_details'] = low_usage_events_df[['Energy Consumption (kWh)', 'Energy Consumption (kWh)_normalized']].head().to_html()

    # Wastage Events (from high positive residuals)
    # Assuming anomalies were already detected and marked in df['Is_Anomaly'] in day9
    if 'Is_Anomaly' in df.columns:
        wastage_events_df = df[(df['Is_Anomaly'] == -1) & (df['Residuals'] > 0)]
        results['wastage_count'] = len(wastage_events_df)
        results['wastage_details'] = wastage_events_df[['Energy Consumption (kWh)', 'Residuals']].head().to_html()
    else:
        results['wastage_count'] = 0
        results['wastage_details'] = "<p>Anomaly detection not performed or 'Is_Anomaly' column missing.</p>"

    # --- Plot 1: Energy Consumption Distribution with Thresholds ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Energy Consumption (kWh)_normalized'], kde=True, color='#66ccff', ax=ax1)
    ax1.axvline(high_usage_threshold, color='#ff6666', linestyle='--', label=f'High Usage (>{high_usage_threshold:.2f})')
    ax1.axvline(low_usage_threshold, color='#ffcc66', linestyle='--', label=f'Low Usage (<{low_usage_threshold:.2f})')
    ax1.set_title('Distribution of Normalized Energy Consumption with Thresholds')
    ax1.set_xlabel('Normalized Energy Consumption (kWh)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    results['consumption_distribution_plot'] = save_plot_to_base64(fig1)

    # --- Plot 2: Time Series High/Low/Wastage Events ---
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    ax2.plot(df.index, df['Energy Consumption (kWh)_normalized'], color='#66ccff', alpha=0.7, label='Normalized Consumption')
    
    if not high_usage_events_df.empty:
        ax2.scatter(high_usage_events_df.index, high_usage_events_df['Energy Consumption (kWh)_normalized'], 
                    color='#ff6666', s=50, label='High Usage Events', zorder=5)
    
    if not low_usage_events_df.empty:
        ax2.scatter(low_usage_events_df.index, low_usage_events_df['Energy Consumption (kWh)_normalized'], 
                    color='#ffcc66', s=50, label='Low Usage Events', zorder=5)

    if 'Is_Anomaly' in df.columns:
        if not wastage_events_df.empty:
            ax2.scatter(wastage_events_df.index, wastage_events_df['Energy Consumption (kWh)_normalized'], 
                        color='#b16286', marker='X', s=80, label='Potential Wastage (Anomalies)', zorder=6)
    
    ax2.set_title('Energy Consumption Events Over Time')
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Normalized Energy Consumption (kWh)')
    ax2.legend()
    results['events_time_series_plot'] = save_plot_to_base64(fig2)

    return results, df