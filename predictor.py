# predictor.py

import os
import logging
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime, timedelta

from data_loader import load_noaa_live
from data_fetcher import get_flood_levels_from_noaa, classify_flood_risk
from nws_fetcher import get_precipitation_forecast, calculate_precipitation_impact
from stations_df_func import get_stations_df

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run_live_prediction(station_id: str, seq_len: int = 72) -> dict:
    """
    Predicts next hour's water level using:
    1. LSTM prediction based on past 72 hours of water levels
    2. Adjustment based on forecasted precipitation from NWS

    Args:
        station_id: NOAA station ID
        seq_len: Sequence length for LSTM (default 72 hours)

    Returns:
        dict with prediction results and precipitation impact
    """
    # ✅ CHANGED: Look for .keras file first, fallback to .h5
    model_path_keras = f"models/{station_id}_waterlevel_lstm.keras"
    model_path_h5 = f"models/{station_id}_waterlevel_lstm.h5"
    scaler_path = f"models/{station_id}_waterlevel_scaler.pkl"

    # Check which model format exists
    if os.path.exists(model_path_keras):
        model_path = model_path_keras
    elif os.path.exists(model_path_h5):
        model_path = model_path_h5
    else:
        raise FileNotFoundError(f"LSTM model not found for {station_id}. Train first.")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found for {station_id}. Train first.")

    # Load model and scaler
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    logging.info(f"✅ Loaded model from {model_path}")

    # ========================================
    # STEP 1: Get last 72 hours of water levels
    # ========================================
    live_df = load_noaa_live(station_id, seq_len=seq_len)

    if live_df.empty or len(live_df) < seq_len:
        raise ValueError(f"Not enough live data ({len(live_df)} points). Need {seq_len} for prediction.")

    # Prepare sequence (water_level only)
    recent_data = live_df['water_level'].tail(seq_len).values.reshape(-1, 1)

    # Scale the sequence
    scaled_sequence = scaler.transform(recent_data)

    # Reshape for LSTM input [1, seq_len, 1]
    X_pred = np.reshape(scaled_sequence, (1, seq_len, 1))

    # Make base LSTM prediction
    logging.info("🔮 Running LSTM prediction...")
    predicted_scaled = model.predict(X_pred, verbose=0)

    # Inverse transform to get actual water level
    predicted_level_base = scaler.inverse_transform(predicted_scaled)[0, 0]

    logging.info(f"📊 Base LSTM prediction: {predicted_level_base:.2f} ft")

    # ========================================
    # STEP 2: Get station coordinates for NWS forecast
    # ========================================
    precipitation_impact = None
    predicted_level_adjusted = predicted_level_base

    try:
        stations_df = get_stations_df()
        station_row = stations_df[stations_df['station_id'] == station_id]

        if station_row.empty:
            logging.warning(f"⚠️ Station {station_id} not found in master CSV")
            lat, lon = None, None
        else:
            # ✅ FIXED: Use .iloc to get the row, then access columns
            lat = station_row.iloc['latitude']
            lon = station_row.iloc['longitude']

            if pd.isna(lat) or pd.isna(lon):
                logging.warning(f"⚠️ No coordinates for station {station_id}")
                lat, lon = None, None
            else:
                # Fetch precipitation forecast
                logging.info(f"🌧️ Fetching precipitation forecast for ({lat}, {lon})...")
                precip_df = get_precipitation_forecast(lat, lon, hours=72)

                if not precip_df.empty:
                    # Calculate precipitation impact
                    precipitation_impact = calculate_precipitation_impact(precip_df)

                    # Adjust prediction based on expected water rise
                    water_rise = precipitation_impact['expected_water_rise_ft']
                    predicted_level_adjusted = predicted_level_base + water_rise

                    logging.info(f"🌧️ Precipitation adjustment: +{water_rise:.2f} ft")
                    logging.info(f"📊 Adjusted prediction: {predicted_level_adjusted:.2f} ft")
                else:
                    logging.warning("⚠️ No precipitation forecast available, using base prediction")

    except Exception as e:
        logging.error(f"❌ Error fetching precipitation data: {e}")
        import traceback
        logging.error(traceback.format_exc())
        logging.info("Using base LSTM prediction without precipitation adjustment")

    # ========================================
    # STEP 3: Classify flood risk
    # ========================================
    thresholds = get_flood_levels_from_noaa(station_id)

    if thresholds and 'Minor' in thresholds:
        df_pred = pd.DataFrame({'water_level': [predicted_level_adjusted]})
        df_pred_classified = classify_flood_risk(df_pred, thresholds)
        predicted_risk = df_pred_classified['risk_level'].iloc  # ✅ FIXED: Added
    else:
        predicted_risk = "Unknown"
        logging.warning(f"⚠️ No flood thresholds available for {station_id}")

    # ========================================
    # STEP 4: Get timestamp for prediction
    # ========================================
    last_timestamp = pd.to_datetime(live_df['date_time'].iloc[-1])
    prediction_time = last_timestamp + timedelta(hours=1)

    # ========================================
    # STEP 5: Build result dictionary
    # ========================================
    result = {
        "station_id": station_id,
        "predicted_for_timestamp": prediction_time.strftime('%Y-%m-%d %H:%M:%S'),
        "predicted_water_level_ft": float(predicted_level_adjusted),
        "base_lstm_prediction_ft": float(predicted_level_base),
        "predicted_risk_level": predicted_risk,
        "features_used_for_last_step": live_df.tail(1).to_dict(orient='records')
    }

    # Add precipitation impact if available
    if precipitation_impact:
        result["precipitation_forecast"] = {
            "max_precip_probability": precipitation_impact['max_precip_probability'],
            "avg_precip_probability": precipitation_impact['avg_precip_probability'],
            "expected_water_rise_ft": precipitation_impact['expected_water_rise_ft'],
            "flood_risk_multiplier": precipitation_impact['flood_risk_multiplier'],
            "hours_with_precip": precipitation_impact['hours_with_precip']
        }
    else:
        result["precipitation_forecast"] = {
            "status": "Not available - using base LSTM prediction only"
        }

    return result
