# predictor.py
"""
Run live prediction:
- load model + scaler
- load last seq_len historical water levels (NOAA does NOT provide live)
- run LSTM → base prediction
- fetch NWS precipitation forecast (12h) and apply heuristic adjustment
- classify flood risk using station-specific thresholds
"""

import os
import logging
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from datetime import timedelta

from data_loader import load_last_hours_from_cache
from data_fetcher import get_flood_levels_from_noaa, classify_flood_risk
from nws_fetcher import get_precipitation_forecast
from stations_df_func import get_stations_df

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------
# PRECIPITATION IMPACT
# ---------------------------------------------------------
def _calculate_precipitation_impact(precip_df: pd.DataFrame) -> dict:
    """
    Simple mapping of precipitation probability/amount → expected water rise in feet.
    """
    if precip_df is None or precip_df.empty:
        return {
            'expected_water_rise_ft': 0.0,
            'max_prob': 0.0,
            'avg_prob': 0.0,
            'hours_with_precip': 0
        }

    if 'precipitation_probability' in precip_df.columns:
        max_prob = float(precip_df['precipitation_probability'].max())
        avg_prob = float(precip_df['precipitation_probability'].mean())
        hours_with = int((precip_df['precipitation_probability'] > 30).sum())

        if max_prob >= 80:
            rise = 2.0
        elif max_prob >= 60:
            rise = 1.0
        elif max_prob >= 30:
            rise = 0.5
        else:
            rise = 0.0

        return {
            'expected_water_rise_ft': rise,
            'max_prob': max_prob,
            'avg_prob': avg_prob,
            'hours_with_precip': hours_with
        }

    elif 'precipitation_amount' in precip_df.columns:
        max_amt = float(precip_df['precipitation_amount'].max())
        rise = (max_amt / 0.5) * 0.25
        return {
            'expected_water_rise_ft': float(rise),
            'max_amt': max_amt,
            'avg_amt': float(precip_df['precipitation_amount'].mean())
        }

    return {'expected_water_rise_ft': 0.0}


# ---------------------------------------------------------
# MAIN LIVE PREDICTOR
# ---------------------------------------------------------
def run_live_prediction(station_id: str, seq_len: int = 72) -> dict:
    """
    Live forecast (based on latest available historical NOAA water-level).
    """

    # Load model + scaler
    model_path_keras = f"models/{station_id}_waterlevel_lstm.keras"
    model_path_h5 = f"models/{station_id}_waterlevel_lstm.h5"

    if os.path.exists(model_path_keras):
        model_path = model_path_keras
    elif os.path.exists(model_path_h5):
        model_path = model_path_h5
    else:
        raise FileNotFoundError("LSTM model file not found. Train a model first.")

    scaler_path = f"models/{station_id}_waterlevel_scaler.pkl"
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Scaler file not found. Train a model first.")

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    logging.info(f"✅ Loaded LSTM model ({model_path}) and scaler.")

    # Load last seq_len hours from cached historical file
    live_df = load_last_hours_from_cache(station_id, hours=seq_len)
    if live_df.empty:
        raise ValueError("No data available for prediction (last-hours cache empty).")

    if 'water_level' not in live_df.columns:
        raise ValueError("Dataset missing 'water_level' column.")

    # ============================================================
    # HANDLE SHAPE ISSUES ROBUSTLY
    # ============================================================

    # Extract water level values
    seq_vals = live_df['water_level'].values

    # Validate we have enough data
    if len(seq_vals) < seq_len:
        raise ValueError(
            f"Not enough data for prediction. Need {seq_len} hours, but only have {len(seq_vals)} hours. "
            f"Please ensure historical data covers at least {seq_len} hours."
        )

    # Take only the last seq_len values (in case we have more)
    seq_vals = seq_vals[-seq_len:]

    # Reshape to 2D array for scaler: (seq_len, 1)
    seq_vals_2d = seq_vals.reshape(-1, 1)

    # Scale the data
    scaled_seq = scaler.transform(seq_vals_2d)

    # Validate scaled sequence length
    if len(scaled_seq) != seq_len:
        raise ValueError(f"Scaling error: Expected {seq_len} timesteps, got {len(scaled_seq)}")

    # Reshape for LSTM input: (batch_size=1, timesteps=seq_len, features=1)
    X_pred = scaled_seq.reshape(1, seq_len, 1)

    logging.info(f"📊 Input shape for prediction: {X_pred.shape}")

    # Predict using LSTM
    pred_scaled = model.predict(X_pred, verbose=0)

    # ============================================================
    # ROBUST INVERSE TRANSFORM
    # ============================================================

    # Handle different prediction output shapes
    if pred_scaled.ndim == 2:
        # Shape is (1, 1) or (1, n_features)
        pred_val_scaled = pred_scaled[0, 0]
    elif pred_scaled.ndim == 1:
        # Shape is (1,)
        pred_val_scaled = pred_scaled
    else:
        # Unexpected shape, flatten and take first value
        pred_val_scaled = pred_scaled.flatten()

    # Reshape for inverse transform: (1, 1)
    pred_val_2d = np.array([[pred_val_scaled]])

    # Inverse transform to get actual water level
    inv = scaler.inverse_transform(pred_val_2d)
    predicted_level_base = float(inv[0, 0])

    logging.info(f"📊 Base LSTM prediction: {predicted_level_base:.2f} ft")

    # ============================================================
    # CORRECT LAT/LON EXTRACTION
    # ============================================================

    # Fetch precipitation forecast (requires lat/lon)
    stations_df = get_stations_df()
    lat = lon = None

    if not stations_df.empty:
        row = stations_df[stations_df['station_id'].astype(str) == str(station_id)]

        if not row.empty:
            # Extract values using bracket notation
            station_data = row.iloc[0]
            lat = station_data['latitude']
            lon = station_data['longitude']

            # Validate lat/lon are numbers
            if pd.notna(lat) and pd.notna(lon):
                logging.info(f"📍 Station coordinates: lat={lat}, lon={lon}")
            else:
                logging.warning(f"⚠️ Station {station_id} has invalid coordinates")
                lat = lon = None
        else:
            logging.warning(f"⚠️ Station {station_id} not found in metadata")
    else:
        logging.warning("⚠️ Station metadata is empty")

    precipitation_summary = {}
    predicted_level_adjusted = predicted_level_base

    if lat is not None and lon is not None:
        try:
            precip_df = get_precipitation_forecast(float(lat), float(lon), hours=12)
            precipitation_summary = _calculate_precipitation_impact(precip_df)
            predicted_level_adjusted = predicted_level_base + precipitation_summary.get('expected_water_rise_ft', 0.0)
            logging.info(
                f"🌧️ Precipitation adjustment: +{precipitation_summary.get('expected_water_rise_ft', 0.0):.2f} ft")
        except Exception as e:
            logging.warning(f"⚠️ NWS precipitation fetch failed: {e}")
            precipitation_summary = {'expected_water_rise_ft': 0.0, 'status': 'unavailable'}
    else:
        logging.warning("⚠️ Station lacks lat/lon; skipping precipitation.")
        precipitation_summary = {'expected_water_rise_ft': 0.0, 'status': 'no_coords'}

    # ============================================================
    # FLOOD RISK CLASSIFICATION (UPDATED TO USE TWO-TIER SYSTEM)
    # ============================================================

    # Get flood thresholds using two-tier approach
    # Option 1: NOAA metadata (most accurate)
    # Option 2: Statistical calculation from historical data
    thresholds = get_flood_levels_from_noaa(station_id)

    # Create prediction dataframe
    df_pred = pd.DataFrame({'water_level': [predicted_level_adjusted]})

    # Classify flood risk
    if thresholds is not None:
        df_class = classify_flood_risk(df_pred, thresholds)
        predicted_risk = df_class['risk_level'].iloc[0]
        threshold_method = thresholds.get('method', 'unknown')

        # Include threshold details in result
        threshold_info = {
            'method': threshold_method,
            'minor_threshold': thresholds.get('Minor'),
            'moderate_threshold': thresholds.get('Moderate'),
            'major_threshold': thresholds.get('Major')
        }

        # Add statistical info if available
        if threshold_method == 'statistical':
            threshold_info['mean'] = thresholds.get('mean')
            threshold_info['std'] = thresholds.get('std')
            threshold_info['data_points'] = thresholds.get('data_points')
    else:
        # No thresholds available
        predicted_risk = "Unknown (No Thresholds Available)"
        threshold_info = {
            'method': 'none',
            'error': 'Could not determine flood thresholds from NOAA or historical data'
        }
        logging.error(f"❌ Cannot classify flood risk for {station_id} - no thresholds available")

    # Calculate prediction timestamp
    last_obs_time = pd.to_datetime(live_df['date_time'].iloc[-1])
    pred_time = last_obs_time + timedelta(hours=1)

    # ============================================================
    # RETURN COMPREHENSIVE RESULTS
    # ============================================================

    return {
        "station_id": station_id,
        "prediction_for_timestamp": pred_time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_lstm_prediction_ft": round(predicted_level_base, 3),
        "precipitation_impact_ft": round(float(precipitation_summary.get('expected_water_rise_ft', 0.0)), 3),
        "predicted_water_level_ft": round(predicted_level_adjusted, 3),
        "predicted_risk_level": predicted_risk,
        "threshold_info": threshold_info,
        "features_used_for_last_step": live_df.tail(1).to_dict(orient='records'),
        "precipitation_summary": precipitation_summary
    }
