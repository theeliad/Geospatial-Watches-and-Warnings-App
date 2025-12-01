# data_loader.py

import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta
import logging
import time

from data_fetcher import get_tide_data, FLOOD_LEVELS, classify_flood_risk

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_noaa_historical(station_id: str, years: int = 1, progress_bar=None):
    """
    Tries to fetch real historical NOAA data. If it lacks flood events,
    it falls back to generating realistic simulated data for model training.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)
    logging.info(f"Attempting to fetch {years} year(s) of real historical data for station {station_id}...")

    output_dir = "data/historical"
    os.makedirs(output_dir, exist_ok=True)
    file_path = f"{output_dir}/{station_id}_historical_{years}y.csv"

    if os.path.exists(file_path):
        logging.info(f"Loading cached historical data from {file_path}")
        if progress_bar:
            progress_bar.progress(100, text="Loaded data from cache.")
        return pd.read_csv(file_path)

    if progress_bar:
        progress_bar.progress(10, text=f"Fetching {years} year(s) of water level data...")
        time.sleep(0.5)

    df_water = get_tide_data(
        station_id,
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d")
    )

    use_simulated_data = False
    if df_water.empty:
        logging.warning("No real water level data retrieved. Falling back to simulation.")
        use_simulated_data = True
    else:
        df = df_water.reset_index().rename(columns={"index": "date_time"})

        df["wind_speed"] = np.random.uniform(0, 40, len(df))
        df["air_pressure"] = np.random.uniform(980, 1040, len(df))
        df["air_temperature"] = np.random.uniform(10, 95, len(df))

        thresholds = FLOOD_LEVELS.get(station_id)
        if thresholds and "Minor" in thresholds:
            df["flood_flag"] = (df["water_level"] >= thresholds["Minor"]).astype(int)
            if df["flood_flag"].nunique() < 2:
                logging.warning(f"Real data for {station_id} has no flood events. Falling back to simulation.")
                use_simulated_data = True
        else:
            df["flood_flag"] = 0
            logging.warning(f"No flood thresholds for {station_id}. Using 0 for all flood flags.")
            use_simulated_data = True

    if use_simulated_data:
        logging.info("Generating 1 year of realistic simulated data...")
        if progress_bar:
            progress_bar.progress(40, text="Generating simulated data...")
            time.sleep(0.5)

        num_rows = 24 * 365 * 1
        dates = pd.to_datetime(pd.date_range(start=start_date, end=end_date, periods=num_rows))

        thresholds = FLOOD_LEVELS.get(station_id, {})
        minor_level = thresholds.get('Minor', 12.0)

        base_tide_center = minor_level / 2.0
        tide_amplitude = base_tide_center + 1.0
        storm_surge_std = 1.0

        base_water = base_tide_center + tide_amplitude * np.sin(np.linspace(0, 1 * 2 * np.pi * 365.25, num_rows))
        storm_surge = np.random.normal(0, storm_surge_std, num_rows)
        surge_indices = np.random.randint(0, num_rows, int(num_rows * 0.01))
        storm_surge[surge_indices] += np.random.uniform(2.0, 5.0, int(num_rows * 0.01))
        water_level = base_water + storm_surge

        df = pd.DataFrame({
            'date_time': dates,
            'water_level': water_level,
            'wind_speed': np.random.uniform(0, 40, num_rows),
            'air_pressure': np.random.uniform(980, 1040, num_rows),
            'air_temperature': np.random.uniform(10, 95, num_rows),
        })

        if thresholds and "Minor" in thresholds:
            df["flood_flag"] = (df["water_level"] >= thresholds["Minor"]).astype(int)
        else:
            df["flood_flag"] = 0

    if thresholds:
        df = classify_flood_risk(df, thresholds)
    else:
        df["risk_level"] = "No Threshold"

    flood_event_count = df['flood_flag'].sum()
    logging.info(f"Final dataset contains {flood_event_count} flood events.")

    if progress_bar:
        progress_bar.progress(80, text="Saving data to cache...")

    df.to_csv(file_path, index=False)
    logging.info(f"Historical data saved to {file_path}")

    if progress_bar:
        progress_bar.progress(100, text="Historical data loaded!")

    return df


def load_noaa_live(station_id: str, seq_len: int = 72):
    """
    Fetches the latest `seq_len` observations for a station for live prediction.

    NEW APPROACH:
    - Fetches ONLY water level data (no met data needed)
    - LSTM uses water levels to predict base tide
    - Precipitation forecast (from NWS) adjusts the prediction

    Returns:
        DataFrame with columns: date_time, water_level
    """
    logging.info(f"🌊 Fetching last {seq_len} hours of live water level data for station {station_id}...")

    # Define time window with larger buffer for reliability
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=seq_len + 24)  # 24-hour buffer

    # ✅ FIX: Use date-only format (more reliable for NOAA API)
    df_water = get_tide_data(
        station_id,
        start_date=start_date.strftime("%Y%m%d"),  # Format: "20251122"
        end_date=end_date.strftime("%Y%m%d")  # Format: "20251125"
    )

    if df_water.empty:
        logging.error(f"❌ No live water level data available for station {station_id}")
        logging.error(f"   Possible reasons:")
        logging.error(f"   - Station is offline or not reporting")
        logging.error(f"   - Station does not support real-time data")
        logging.error(f"   - Temporary API issue")
        return pd.DataFrame()

    # Reset index to get date_time as a column
    df = df_water.reset_index().rename(columns={"index": "date_time"})

    # Ensure datetime format
    df['date_time'] = pd.to_datetime(df['date_time'])

    # Sort by date and get most recent data
    df = df.sort_values('date_time', ascending=True)

    # Check if we have enough recent data
    if len(df) < seq_len:
        logging.warning(f"⚠️ Only {len(df)} hours available (need {seq_len})")
        logging.warning(f"   Using all available data for prediction")

    # Get the most recent seq_len hours
    df = df.tail(seq_len).reset_index(drop=True)

    logging.info(f"✅ Retrieved {len(df)} hours of water level data")
    logging.info(f"   Date range: {df['date_time'].min()} to {df['date_time'].max()}")

    return df
