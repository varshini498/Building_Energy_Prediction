import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from sklearn.metrics import r2_score, mean_absolute_error
import os
import matplotlib.pyplot as plt
import io
import base64

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def save_plot_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('latin-1')
    plt.close(fig)
    return image_base64

def run_day7_hybrid(df):
    df.dropna(inplace=True)
    df_numeric = df.select_dtypes(include=['number'])
    features = [col for col in df_numeric.columns if col not in ['Energy Consumption (kWh)_normalized', 'Energy Consumption (kWh)', 'Consumption Volatility', 'target', 'Residuals', 'Is_Anomaly']]
    sequence_features = ['lag_24h', 'lag_168h', 'rolling_mean_24h']
    static_features = [col for col in features if col not in sequence_features]
    if not all(f in df_numeric.columns for f in sequence_features) or not all(f in df_numeric.columns for f in static_features):
        print("Warning: Missing features for hybrid model. Skipping.")
        return {'model': 'Hybrid_LSTM', 'r2': -1, 'mae': -1, 'loss_plot': None}, df
    X = df_numeric[features]
    y = df_numeric['Energy Consumption (kWh)_normalized']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
    
    # Train the model and get the history
    history = hybrid_model.fit([X_train_sequence, X_train_static], y_train, epochs=2, verbose=0)
    
    y_pred = hybrid_model.predict([X_test_sequence, X_test_static])
    
    # Create the loss plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history.history['loss'], label='Training Loss')
    ax.set_title('Hybrid Model Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.legend()
    loss_plot_base64 = save_plot_to_base64(fig)
    
    metrics = {
        'model': 'Hybrid_LSTM',
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'loss_plot': loss_plot_base64
    }
    hybrid_model.save('hybrid_model.h5')
    return metrics, df