a# stations_df_func.py

import pandas as pd
import noaa_coops as nc
import logging
import os


def get_stations_df() -> pd.DataFrame:
    """
    Tries to fetch the latest NOAA station list. If it fails, it loads a
    fallback list from a local CSV file.
    """
    # First, try the live API call
    try:
        logging.info("Attempting to fetch live station metadata from NOAA...")
        stations = nc.get_stations(product='water_level', active=True)
        if stations:
            station_data = [{
                "station_id": s.id, "station_name": s.name, "state": s.state,
                "latitude": s.lat, "longitude": s.lon, "station_type": s.station_type,
            } for s in stations]
            df = pd.DataFrame(station_data)
            logging.info(f"Successfully fetched {len(df)} live stations.")
            return df
    except Exception as e:
        logging.warning(f"Live NOAA station fetch failed: {e}. Attempting to load from local fallback.")

    # If the live call fails, use the local fallback file
    fallback_path = os.path.join('data', 'noaa_stations.csv')
    try:
        logging.info(f"Loading station metadata from fallback file: {fallback_path}")
        if not os.path.exists(fallback_path):
            logging.error(f"Fallback file not found at {fallback_path}")
            return pd.DataFrame()
        df = pd.read_csv(fallback_path)
        logging.info(f"Successfully loaded {len(df)} stations from fallback CSV.")
        return df
    except Exception as e:
        logging.error(f"Failed to load fallback station data: {e}")
        return pd.DataFrame()
