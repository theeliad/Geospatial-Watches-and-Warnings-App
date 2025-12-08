# model_trainer.py

import os
import logging
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.preprocessing import MinMaxScaler
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --- CUSTOM KERAS CALLBACK ---
class StreamlitCallback(Callback):
    def __init__(self, progress_bar, total_epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        percent_complete = (epoch + 1) / self.total_epochs
        val_loss = logs.get('val_loss', 0)
        self.progress_bar.progress(
            percent_complete,
            text=f"Training Epoch {epoch + 1}/{self.total_epochs} – Validation Loss: {val_loss:.4f}"
        )


# --- HELPER FUNCTION ---
def create_lstm_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i: i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


# --- MAIN TRAINING FUNCTION ---
def train_model(
        station_id: str,
        years: int = 1,
        seq_len: int = 72,
        epochs: int = 10,
        batch_size: int = 64,
        progress_bar=None
):
    """
    Trains a univariate LSTM model to predict future water levels.
    """

    # ============================================================
    # CHANGE 1: FLEXIBLE FILE SEARCH (from old version)
    # ============================================================
    file_dir = "data/historical"
    file_candidates = [
        f"{file_dir}/{station_id}_historical_{years}y.csv",
        f"{file_dir}/{station_id}_historical_1y.csv",
        f"{file_dir}/{station_id}_historical_5y.csv",
        f"{file_dir}/{station_id}_historical.csv"
    ]

    df = None
    for file_path in file_candidates:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, parse_dates=['date_time'], index_col='date_time')
            logging.info(f"✅ Found historical data: {file_path}")
            break

    if df is None:
        raise FileNotFoundError(
            f"No historical data found for station {station_id}. Please load data first (Step 1)."
        )

    if "water_level" not in df.columns:
        raise ValueError("Historical dataset missing 'water_level' column.")

    df = df[["water_level"]].dropna()
    if df.empty:
        raise ValueError("Data is empty after cleaning.")

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # Create sequences and split data
    X, y = create_lstm_sequences(scaled_data, seq_len)
    if len(X) < 100:
        raise ValueError("Not enough data to create training/test sets.")

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    logging.info(f"Training LSTM with {X_train.shape} sequences.")

    # Build the model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Define callbacks
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # ============================================================
    # CHANGE 3: USE .keras EXTENSION CONSISTENTLY
    # ============================================================
    model_path = f"{model_dir}/{station_id}_waterlevel_lstm.keras"

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min')

    callbacks_list = [early_stop, checkpoint]
    if progress_bar:
        callbacks_list.append(StreamlitCallback(progress_bar, epochs))

    # Train the model
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=0
    )

    logging.info(f"Best LSTM model saved to {model_path}")

    # ============================================================
    # CHANGE 2: SAVE SCALER AFTER TRAINING (from old version)
    # ============================================================
    scaler_path = f"{model_dir}/{station_id}_waterlevel_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logging.info(f"Scaler saved to {scaler_path}")

    return model_path
