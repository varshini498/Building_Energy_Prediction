import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import randint

def optimize_model_randomized(file_path):
    """
    Performs a very fast hyperparameter tuning using RandomizedSearchCV.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        print("Please ensure you have run the Day 4 script to create this file.")
        return None

    print("### 🤖 Days 6-7: The Fastest Optimization Technique")
    
    try:
        df = pd.read_csv(file_path, index_col='Timestamp', parse_dates=True)
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None
    
    df.dropna(inplace=True)

    features = df.columns.tolist()
    features.remove('Energy Consumption (kWh)_normalized')
    
    X = df[features]
    y = df['Energy Consumption (kWh)_normalized']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    print("\n#### ⚙️ Hyperparameter Tuning with RandomizedSearchCV")
    
    # --- RANDOMIZED HYPERPARAMETER GRID ---
    # This defines the distribution of values to sample from
    param_dist = {
        'n_estimators': randint(low=50, high=200),
        'max_depth': randint(low=10, high=50),
        'min_samples_leaf': randint(low=1, high=10),
    }

    # Initialize the RandomizedSearchCV object
    random_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=10, # Number of random combinations to test
        cv=3,
        scoring='r2',
        n_jobs=1,
        verbose=1,
        random_state=42
    )

    print("Starting a faster randomized search...")
    random_search.fit(X_train, y_train)

    print("\nRandomized search completed.")
    print(f"Best hyperparameters found: {random_search.best_params_}")
    print(f"Best R² score from random search: {random_search.best_score_:.4f}")

    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n#### 📈 Optimized Model Performance Metrics:")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    metrics_df = pd.DataFrame({
        'Model': ['RandomForest_Optimized_Random'], 
        'R² Score': [r2], 
        'MAE': [mae], 
        'RMSE': [rmse]
    })
    metrics_df.to_csv('optimized_model_metrics.csv', index=False)
    print("\nOptimized model metrics saved to 'optimized_model_metrics.csv'.")
    print("\nModel optimization is complete.")

# To run the script:
if __name__ == "__main__":
    file_for_optimization = 'selected_features_building_data.csv'
    optimize_model_randomized(file_for_optimization)