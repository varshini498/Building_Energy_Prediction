import os
import matplotlib.pyplot as plt
from flask import Flask, render_template

# Set Matplotlib style before any other imports
plt.style.use('dark_background')
plt.rcParams.update({'figure.facecolor': '#2b2b2b', 'axes.facecolor': '#2b2b2b'})

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
results = {}

def run_all_pipeline():
    global results
    results = {}
    
    print("Running the entire Smart Energy Analysis pipeline...")
    
    # Day 1: Preprocessing
    df = pd.read_csv('building_dataset.csv', encoding='latin1')
    duplicates, df = run_day1_preprocessing(df)
    results['day1_duplicates'] = duplicates
    
    # Day 2: Fluctuations
    fluctuation_plots, df = run_day2_fluctuations(df)
    results.update(fluctuation_plots)
    
    # Day 3: Feature Engineering
    feature_importance_table, feature_plot, df = run_day3_engineering(df)
    results['feature_importance'] = feature_importance_table
    results.update(feature_plot)

    # Day 4: Feature Selection
    selection_results, df = run_day4_selection(df)
    results.update(selection_results)
    
    # Day 5: Feature Extraction (PCA)
    extraction_plot, df = run_day5_extraction(df)
    results.update(extraction_plot)
    
    # Day 6: Optimization
    optimized_metrics, df = run_day6_optimization(df)
    results['optimized_r2'] = optimized_metrics['r2']
    results['optimized_mae'] = optimized_metrics['mae']
    
    # Day 7: Hybrid Models
    hybrid_metrics, y_test_hybrid, y_pred_hybrid, df = run_day7_hybrid(df)
    results['hybrid_r2'] = hybrid_metrics['r2']
    
    # Day 8: Residual Calculation
    df = run_day8_residual_calculation(df)

    # Day 9: Anomaly Detection
    anomaly_results, df = run_day9_anomaly_detection(df)
    results.update(anomaly_results)
    
    # Day 10: Wastage Analysis
    wastage_results = run_day10_wastage_analysis(df)
    results.update(wastage_results)
    
    # Day 11: Final Comparison
    comparison_results = run_day11_comparison(df)
    results.update(comparison_results)
    
    print("Pipeline complete. Results are ready to be displayed on the web page.")

@app.route('/')
def home():
    try:
        if not results:
            run_all_pipeline()
        return render_template('index.html', results=results)
    except FileNotFoundError as e:
        return f"Error: {e}. Please ensure all files are in the correct directory.", 404
    except Exception as e:
        return f"An error occurred: {e}", 500

if __name__ == '__main__':
    if os.path.exists('templates') and os.path.isdir('templates'):
        pass
    else:
        os.makedirs('templates')

    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Energy Analysis</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
        body {
            font-family: 'Inter', sans-serif;
            background-color: #0d1117;
            color: #c9d1d9;
            line-height: 1.6;
        }
        .container {
            max-width: 1200px;
            margin: auto;
            padding: 2rem;
        }
        .card {
            background-color: #161b22;
            border-radius: 0.5rem;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border: 1px solid #30363d;
        }
        .plot-container {
            width: 100%;
            height: auto;
            max-width: 800px;
            margin: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }
        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #30363d;
        }
        th {
            background-color: #1f242a;
            font-weight: 700;
        }
        .section-header {
            color: #58a6ff;
            border-bottom: 2px solid #58a6ff;
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
        }
    </style>
</head>
<body class="p-8">

    <div class="container">
        <header class="text-center mb-8">
            <h1 class="text-4xl font-bold text-white mb-2">Smart Energy Analysis Pipeline</h1>
            <p class="text-gray-400">Transforming data into a comprehensive report.</p>
        </header>

        <main class="space-y-8">
            <section class="card">
                <h2 class="text-2xl font-bold section-header">Day 1: Data Preprocessing</h2>
                <p>Initial rows removed: <strong>{{ results.day1_duplicates }}</strong></p>
                <p>Dataset cleaned and normalized. Ready for the next steps.</p>
            </section>

            <section class="card">
                <h2 class="text-2xl font-bold section-header">Day 2: Energy Fluctuations</h2>
                <div class="space-y-4 text-center">
                    <img class="plot-container" src="data:image/png;base64,{{ results.fluctuation_plot_1 }}" alt="Energy Consumption Plot">
                    <img class="plot-container" src="data:image/png;base64,{{ results.fluctuation_plot_2 }}" alt="Volatility Plot">
                </div>
            </section>
            
            <section class="card">
                <h2 class="text-2xl font-bold section-header">Day 3: Feature Engineering</h2>
                <h3 class="text-xl font-semibold mt-4 mb-2">Top 10 Most Important Features</h3>
                <pre>{{ results.feature_importance }}</pre>
                <div class="text-center">
                    <img class="plot-container mt-4" src="data:image/png;base64,{{ results.feature_importance_plot }}" alt="Feature Importance Plot">
                </div>
            </section>
            
            <section class="card">
                <h2 class="text-2xl font-bold section-header">Day 4: Feature Selection</h2>
                <p>Features removed due to high correlation: <strong>{{ results.highly_correlated_count }}</strong></p>
            </section>

            <section class="card">
                <h2 class="text-2xl font-bold section-header">Day 5: Feature Extraction</h2>
                <h3 class="text-xl font-semibold mt-4 mb-2">2D PCA Visualization</h3>
                <div class="text-center">
                    <img class="plot-container" src="data:image/png;base64,{{ results.pca_plot }}" alt="PCA Plot">
                </div>
            </section>

            <section class="card">
                <h2 class="text-2xl font-bold section-header">Day 6: Model Optimization</h2>
                <p>Optimized Random Forest Model R² Score: <strong>{{ '%.4f' | format(results.optimized_r2) }}</strong></p>
            </section>

            <section class="card">
                <h2 class="text-2xl font-bold section-header">Day 7: Hybrid Models</h2>
                <p>Hybrid LSTM Model R² Score: <strong>{{ '%.4f' | format(results.hybrid_r2) }}</strong></p>
            </section>

            <section class="card">
                <h2 class="text-2xl font-bold section-header">Day 8: Residual Calculation</h2>
                <p>Residuals (prediction errors) have been calculated and saved for anomaly detection.</p>
            </section>

            <section class="card">
                <h2 class="text-2xl font-bold section-header">Day 9: Anomaly Detection</h2>
                <p>Anomalies found in the data: <strong>{{ results.anomalies_found }}</strong></p>
                <div class="text-center">
                    <img class="plot-container" src="data:image/png;base64,{{ results.anomaly_plot }}" alt="Anomaly Plot">
                </div>
            </section>

            <section class="card">
                <h2 class="text-2xl font-bold section-header">Day 10: Energy Wastage Analysis</h2>
                <p>High Usage Events: <strong>{{ results.high_usage_count }}</strong></p>
                <p>Low Usage Events: <strong>{{ results.low_usage_count }}</strong></p>
                <p>Potential high wastage events: <strong>{{ results.wastage_count }}</strong></p>
            </section>

            <section class="card">
                <h2 class="text-2xl font-bold section-header">Day 11: Final Report & Comparison</h2>
                <h3 class="text-xl font-semibold mb-2">Model Performance Comparison</h3>
                <pre>{{ results.comparison_table }}</pre>
            </section>
        </main>
    </div>

</body>
</html>
        ''')

    app.run(debug=True, use_reloader=False)