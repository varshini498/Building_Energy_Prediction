import os
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import numpy as np
import warnings
import io
import base64
import joblib
import uuid

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

plt.style.use('dark_background')
plt.rcParams.update({'figure.facecolor': '#2b2b2b', 'axes.facecolor': '#2b2b2b', 'figure.edgecolor': '#2b2b2b'})

from day1_preprocessing import run_day1_preprocessing
from day2_fluctuations import run_day2_fluctuations
from day3_engineering import run_day3_engineering
from day4_selection import run_day4_selection
from day5_extraction import run_day5_extraction
from day6_optimization import run_day6_optimization
from day7_hybrid import run_day7_hybrid
from day8_residual_calculation import run_day8_residual_calculation
from day9_anomaly_detection import run_day9_anomaly_detection
from day10_wastage_analysis import run_day10_wastage_analysis
from day11_comparison import run_day11_comparison

app = Flask(__name__)
app.secret_key = 'your_strong_secret_key_here'
TEMP_DIR = 'temp'

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

def save_df_to_temp(df):
    filename = f"{uuid.uuid4()}.pkl"
    filepath = os.path.join(TEMP_DIR, filename)
    df.to_pickle(filepath)
    return filename

def load_df_from_temp(filename):
    filepath = os.path.join(TEMP_DIR, filename)
    return pd.read_pickle(filepath)

def check_df_and_redirect():
    filename = session.get('filename')
    if filename is None:
        return redirect(url_for('upload_dataset'))
    df = load_df_from_temp(filename)
    if df is None or df.empty:
        return "Error: DataFrame is empty. Please upload a new dataset.", 400
    return df

@app.route('/')
def home():
    return redirect(url_for('upload_dataset'))

@app.route('/upload', methods=['GET', 'POST'])
def upload_dataset():
    session.clear()
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part', 400
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400
        if file:
            try:
                df = pd.read_csv(file, encoding='latin1')
                filename = save_df_to_temp(df)
                session['filename'] = filename
                return redirect(url_for('preprocess_data'))
            except Exception as e:
                return f"Error reading file: {e}", 400
    return render_template('page1_upload.html')

@app.route('/preprocess', methods=['GET'])
def preprocess_data():
    df = check_df_and_redirect()
    if isinstance(df, tuple): return df
    duplicates_removed, processed_df = run_day1_preprocessing(df.copy())
    session['filename'] = save_df_to_temp(processed_df)
    results = {'duplicates_removed': duplicates_removed, 'sample_data': processed_df.head().to_html()}
    return render_template('page2_preprocessing.html', results=results)

@app.route('/fluctuations', methods=['GET'])
def show_fluctuations():
    df = check_df_and_redirect()
    if isinstance(df, tuple): return df
    plots, processed_df = run_day2_fluctuations(df.copy())
    session['filename'] = save_df_to_temp(processed_df)
    return render_template('page3_fluctuations.html', plots=plots)

@app.route('/engineering', methods=['GET'])
def show_engineering():
    df = check_df_and_redirect()
    if isinstance(df, tuple): return df
    feature_importance_table, feature_plot, processed_df = run_day3_engineering(df.copy())
    session['filename'] = save_df_to_temp(processed_df)
    return render_template('page4_engineering.html', table=feature_importance_table, plot=feature_plot['feature_importance_plot'])

@app.route('/selection', methods=['GET'])
def show_selection():
    df = check_df_and_redirect()
    if isinstance(df, tuple): return df
    results, processed_df = run_day4_selection(df.copy())
    session['filename'] = save_df_to_temp(processed_df)
    return render_template('page5_selection.html', results=results)

@app.route('/extraction', methods=['GET'])
def show_extraction():
    df = check_df_and_redirect()
    if isinstance(df, tuple): return df
    plot, processed_df = run_day5_extraction(df.copy())
    session['filename'] = save_df_to_temp(processed_df)
    return render_template('page6_extraction.html', plot=plot['pca_plot'])

@app.route('/optimization', methods=['GET'])
def run_optimization():
    df = check_df_and_redirect()
    if isinstance(df, tuple): return df
    metrics, processed_df = run_day6_optimization(df.copy())
    session['filename'] = save_df_to_temp(processed_df)
    return render_template('page7_optimization.html', metrics=metrics)

@app.route('/hybrid', methods=['GET'])
def run_hybrid():
    df = check_df_and_redirect()
    if isinstance(df, tuple): return df
    metrics, processed_df = run_day7_hybrid(df.copy())
    session['filename'] = save_df_to_temp(processed_df)
    return render_template('page8_hybrid.html', metrics=metrics)

@app.route('/anomaly', methods=['GET'])
def show_anomaly():
    df = check_df_and_redirect()
    if isinstance(df, tuple): return df
    
    # Run both Day 8 (Residuals) and Day 9 (Anomaly) here
    processed_df_with_residuals = run_day8_residual_calculation(df.copy())
    results, processed_df_with_anomaly = run_day9_anomaly_detection(processed_df_with_residuals.copy())
    
    session['filename'] = save_df_to_temp(processed_df_with_anomaly)
    return render_template('page9_anomaly.html', results=results)

@app.route('/wastage', methods=['GET'])
def show_wastage():
    df = check_df_and_redirect()
    if isinstance(df, tuple): return df
    wastage_results, processed_df = run_day10_wastage_analysis(df.copy())
    session['filename'] = save_df_to_temp(processed_df)
    return render_template('page10_wastage.html', results=wastage_results)

@app.route('/comparison', methods=['GET'])
def show_comparison():
    df = check_df_and_redirect()
    if isinstance(df, tuple): return df
    comparison_results, processed_df = run_day11_comparison(df.copy())
    session['filename'] = save_df_to_temp(processed_df)
    return render_template('page11_comparison.html', results=comparison_results)

if __name__ == '__main__':
    # Fix for the Tkinter RuntimeError
    if os.name == 'nt':  # Check if the OS is Windows
        try:
            import multiprocessing
            multiprocessing.freeze_support()
        except (ImportError, AttributeError):
            pass

    app.run(debug=True, use_reloader=False)