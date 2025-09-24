import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.keras.models import load_model
import joblib
import numpy as np

def run_day11_comparison(file_path):
    df = pd.read_csv(file_path, index_col='Timestamp', parse_dates=True)
    df.dropna(inplace=True)
    
    features = [col for col in df.columns if col not in ['Energy Consumption (kWh)_normalized', 'Energy Consumption (kWh)', 'Consumption Volatility', 'Residuals', 'Is_Anomaly']]
    X = df[features]
    y = df['Energy Consumption (kWh)_normalized']
    
    # Base Random Forest model
    rfr = RandomForestRegressor(n_estimators=100, random_state=42)
    rfr.fit(X, y)
    y_pred_rfr = rfr.predict(X)
    
    # Optimized Random Forest model
    optimized_model = joblib.load('optimized_model.pkl')
    y_pred_rfr_optimized = optimized_model.predict(X)
    
    # Hybrid LSTM model
    hybrid_model = load_model('hybrid_model.h5')
    X_sequence = np.reshape(X[['lag_24h', 'lag_168h', 'rolling_mean_24h']].values, (X.shape[0], 1, 3))
    y_pred_hybrid = hybrid_model.predict([X_sequence, X.drop(columns=['lag_24h', 'lag_168h', 'rolling_mean_24h'])])
    
    results = {}
    
    results['comparison_table'] = pd.DataFrame({
        'Model': ['RandomForest', 'RandomForest_Optimized', 'Hybrid_LSTM'],
        'R² Score': [r2_score(y, y_pred_rfr), r2_score(y, y_pred_rfr_optimized), r2_score(y, y_pred_hybrid)],
        'MAE': [mean_absolute_error(y, y_pred_rfr), mean_absolute_error(y, y_pred_rfr_optimized), mean_absolute_error(y, y_pred_hybrid)]
    }).to_markdown(index=False)
    
    return results