import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def run_day7_hybrid(file_path):
    df = pd.read_csv(file_path, index_col='Timestamp', parse_dates=True)
    df.dropna(inplace=True)
    
    features = [col for col in df.columns if col not in ['Energy Consumption (kWh)_normalized', 'Energy Consumption (kWh)', 'Consumption Volatility']]
    X = df[features]
    y = df['Energy Consumption (kWh)_normalized']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    sequence_features = ['lag_24h', 'lag_168h', 'rolling_mean_24h']
    static_features = [col for col in features if col not in sequence_features]
    
    scaler = StandardScaler()
    X_train_static = scaler.fit_transform(X_train[static_features])
    X_test_static = scaler.transform(X_test[static_features])
    
    X_train_sequence = np.reshape(X_train[sequence_features].values, (X_train.shape[0], 1, len(sequence_features)))
    X_test_sequence = np.reshape(X_test[sequence_features].values, (X_test.shape[0], 1, len(sequence_features)))
    
    lstm_input = Input(shape=(1, len(sequence_features)))
    static_input = Input(shape=(len(static_features),))
    lstm_layer = LSTM(64, activation='relu')(lstm_input)
    concat_layer = Concatenate()([lstm_layer, static_input])
    dense_layer = Dense(64, activation='relu')(concat_layer)
    output_layer = Dense(1, activation='linear')(dense_layer)
    hybrid_model = Model(inputs=[lstm_input, static_input], outputs=output_layer)
    hybrid_model.compile(optimizer='adam', loss='mse')
    hybrid_model.fit([X_train_sequence, X_train_static], y_train, epochs=5, verbose=0)
    
    y_pred = hybrid_model.predict([X_test_sequence, X_test_static])
    
    metrics = {
        'model': 'Hybrid_LSTM',
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred)
    }
    
    hybrid_model.save('hybrid_model.h5')
    
    return metrics, 'hybrid_model.h5', y_test, y_pred