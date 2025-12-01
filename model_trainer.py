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


# -----------------------------------------------------
# STREAMLIT TRAINING PROGRESS CALLBACK
# -----------------------------------------------------
class StreamlitCallback(Callback):
    """Updates Streamlit progress bar each epoch."""

    def __init__(self, progress_bar, total_epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        pct = (epoch + 1) / self.total_epochs
        loss = logs.get("loss", 0)
        self.progress_bar.progress(
            pct,
            text=f"Epoch {epoch+1}/{self.total_epochs} – Loss: {loss:.4f}"
        )


# -----------------------------------------------------
# LSTM SEQUENCING
# -----------------------------------------------------
def create_lstm_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)


# -----------------------------------------------------
# LSTM MODEL TRAINING
# -----------------------------------------------------
def train_model(
    station_id: str,
    seq_len: int = 72,
    epochs: int = 10,
    batch_size: int = 64,
    progress_bar=None
):
    """
    Train LSTM model on historical water-level data.
    Compatible with current main.py structure.
    """

    # ------------ Load Cached Data ------------
    file_dir = "data/historical"
    file_candidates = [
        f"{file_dir}/{station_id}_historical_1y.csv",
        f"{file_dir}/{station_id}_historical_5y.csv",
        f"{file_dir}/{station_id}_historical.csv"
    ]

    df = None
    for f in file_candidates:
        if os.path.exists(f):
            df = pd.read_csv(f, parse_dates=["date_time"], index_col="date_time")
            break

    if df is None:
        raise FileNotFoundError(
            f"No historical data is loaded for station {station_id}. Step 1 must be run first."
        )

    if "water_level" not in df.columns:
        raise ValueError("Historical dataset missing 'water_level' column.")

    df = df[["water_level"]].dropna()

    # ------------ Scale Data ------------
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = create_lstm_sequences(scaled_data, seq_len)
    if len(X) < 200:
        raise ValueError("Not enough data to train LSTM.")

    train_cut = int(len(X) * 0.8)
    X_train, X_test = X[:train_cut], X[train_cut:]
    y_train, y_test = y[:train_cut], y[train_cut:]

    # ------------ Build LSTM Model ------------
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    # ------------ Callbacks ------------
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # ✅ CHANGED: Use .keras extension instead of .h5
    model_path = f"{model_dir}/{station_id}_waterlevel_lstm.keras"

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss")
    ]

    # ADD STREAMLIT CALLBACK
    if progress_bar is not None:
        callbacks.append(StreamlitCallback(progress_bar, epochs))

    # ------------ Train ------------
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0  # must be silent so Streamlit handles UI
    )

    # ------------ Save Scaler ------------
    scaler_path = f"{model_dir}/{station_id}_waterlevel_scaler.pkl"
    joblib.dump(scaler, scaler_path)

    logging.info(f"✅ LSTM model saved: {model_path}")
    logging.info(f"✅ Scaler saved: {scaler_path}")

    return model_path
