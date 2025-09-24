import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import randint
import joblib

def run_day6_optimization(file_path):
    df = pd.read_csv(file_path, index_col='Timestamp', parse_dates=True)
    df.dropna(inplace=True)
    
    features = [col for col in df.columns if col not in ['Energy Consumption (kWh)_normalized', 'Energy Consumption (kWh)', 'Consumption Volatility']]
    X = df[features]
    y = df['Energy Consumption (kWh)_normalized']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    param_dist = {
        'n_estimators': randint(low=50, high=200),
        'max_depth': randint(low=10, high=50),
        'min_samples_leaf': randint(low=1, high=10)
    }
    
    random_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions=param_dist, n_iter=10, cv=3, n_jobs=1, verbose=0, random_state=42
    )
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    metrics = {
        'model': 'RandomForest_Optimized',
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred)
    }
    
    joblib.dump(best_model, 'optimized_model.pkl')
    
    return metrics, 'optimized_model.pkl'