# stations_df_func.py
"""
Load and return NOAA station metadata from master CSV file.
Used by main.py for station selection and by predictor.py for lat/lon lookup.
"""

import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------
# GET STATIONS DATAFRAME
# ---------------------------------------------------------
def get_stations_df(csv_path: str = "all_noaa_coops_stations_with_states.csv") -> pd.DataFrame:
    """
    Load NOAA station metadata from CSV file.

    Args:
        csv_path: Path to NOAA stations CSV file

    Returns:
        DataFrame with columns: station_id, station_name, state, latitude, longitude, station_type, display_name
    """

    try:
        logging.info(f"📋 Loading station metadata from '{csv_path}'...")

        df = pd.read_csv(csv_path, dtype={'station_id': str})

        # Normalize column names (strip whitespace, lowercase)
        df.columns = [c.strip().lower() for c in df.columns]

        # Validate required columns
        required_cols = ['station_id', 'station_name', 'state', 'latitude', 'longitude']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logging.error(f"❌ CSV missing required columns: {missing_cols}")
            return pd.DataFrame()

        # Ensure station_id is string type (in case dtype didn't work)
        df['station_id'] = df['station_id'].astype(str)

        # Remove rows with missing critical data
        df = df.dropna(subset=['station_id', 'station_name', 'latitude', 'longitude'])

        # Add station_type column if not present, otherwise keep existing
        if 'station_type' not in df.columns:
            df['station_type'] = 'Water Level'

        # Create display_name column (required by main.py)
        df['display_name'] = df['station_name'] + " (" + df['station_id'] + ")"

        # NOTE: We do NOT filter by supports_water_level because that column
        # appears to be unreliable in the source CSV. Instead, we'll let users
        # select any station and handle errors during data fetching if a station
        # doesn't actually support water level data.

        logging.info(f"✅ Loaded {len(df)} stations from metadata file")
        return df

    except FileNotFoundError:
        logging.error(f"❌ Station metadata file not found: {csv_path}")
        logging.error(f"   Please ensure 'all_noaa_coops_stations_with_states.csv' is in the project root directory")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"❌ Error loading station metadata: {e}")
        return pd.DataFrame()
