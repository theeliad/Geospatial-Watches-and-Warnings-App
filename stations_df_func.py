# stations_df_func.py

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# This is the single source of truth for station metadata.
MASTER_STATIONS_FILE = "master_coastal_stations.csv"


def get_stations_df():
    """
    Loads and normalizes the master station list from the pre-generated CSV.
    """
    logging.info(f"Loading master station list from '{MASTER_STATIONS_FILE}'...")
    master_path = Path(MASTER_STATIONS_FILE)

    if not master_path.is_file():
        logging.error(f"❌ Master stations file not found: {MASTER_STATIONS_FILE}")
        logging.error("Please run create_master_station_list.py first to generate this file.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(master_path, dtype={'noaa_station_id': str, 'nws_station_id': str})

        if df.empty:
            logging.error(f"❌ Master stations file '{MASTER_STATIONS_FILE}' is empty.")
            return pd.DataFrame()

        # --- NORMALIZE COLUMN NAMES for consistent use in the app ---
        # The app will internally use 'station_id', 'latitude', and 'longitude'.
        df.rename(columns={
            'noaa_station_id': 'station_id',
            'noaa_station_name': 'station_name',
            'lat': 'latitude',
            'lon': 'longitude'
        }, inplace=True)

        # Ensure the essential columns exist after renaming
        required_cols = ['station_id', 'station_name', 'latitude', 'longitude', 'display_name']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Master CSV is missing required columns after renaming: {missing}")

        logging.info(f"✅ Loaded and normalized {len(df)} stations.")
        return df

    except Exception as e:
        logging.error(f"❌ Error loading or processing master stations file: {e}")
        return pd.DataFrame()
