# data_fetcher.py

import pandas as pd
import noaa_coops as nc
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Flood level thresholds by station (fallback if API doesn't provide)
FLOOD_LEVELS = {
    '8443970': {'Major': 14.0, 'Moderate': 13.0, 'Minor': 12.0},  # Boston, MA
    '8452660': {'Major': 8.5, 'Moderate': 6.5, 'Minor': 5.5},  # Newport, RI
}


def get_flood_levels_from_noaa(station_id: str, datum='MLLW') -> dict:
    """
    Retrieves official flood level thresholds from NOAA.
    Falls back to FLOOD_LEVELS dict if API doesn't have data.

    Args:
        station_id: NOAA station ID
        datum: Datum reference (default: MLLW)

    Returns:
        dict with keys: Major, Moderate, Minor (flood levels in feet)
    """
    try:
        logging.info(f"🔍 Fetching flood levels for station {station_id}...")
        station = nc.Station(station_id)
        flood_data = station.floodlevels

        if not flood_data:
            logging.warning(f"⚠️ No flood data object returned for {station_id}")
            return FLOOD_LEVELS.get(station_id, {})

        if 'floodlevel' not in flood_data:
            logging.warning(f"⚠️ 'floodlevel' key not found in flood data for {station_id}")
            return FLOOD_LEVELS.get(station_id, {})

        for level_set in flood_data['floodlevel']:
            if level_set.get('datum') == datum:
                levels = {
                    'Major': float(level_set.get('major', 999)),
                    'Moderate': float(level_set.get('moderate', 999)),
                    'Minor': float(level_set.get('minor', 999))
                }
                logging.info(f"✅ Found flood levels for {station_id}: {levels}")
                return levels

        logging.warning(f"⚠️ No flood levels found for datum {datum} at {station_id}")
        return FLOOD_LEVELS.get(station_id, {})

    except Exception as e:
        logging.warning(f"❌ Could not fetch flood levels for {station_id}: {e}")
        return FLOOD_LEVELS.get(station_id, {})


def get_tide_data(station_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches, cleans, and resamples hourly water-level data from NOAA.

    Args:
        station_id: NOAA station ID
        start_date: Start date in format YYYYMMDD
        end_date: End date in format YYYYMMDD

    Returns:
        DataFrame with datetime index and 'water_level' column
    """
    try:
        logging.info(f"🌊 Fetching water level data for station {station_id}: {start_date} → {end_date}")

        station = nc.Station(station_id)
        df = station.get_data(
            begin_date=start_date,
            end_date=end_date,
            product="water_level",
            datum="MLLW",
            units="english",
            time_zone="lst_ldt"
        )

        if df.empty or "water_level" not in df.columns:
            logging.warning(f"⚠️ No valid water-level data for {station_id}.")
            return pd.DataFrame()

        # Keep only water_level column
        df = df[["water_level"]]
        df.index = pd.to_datetime(df.index)

        # Convert to hourly, clean interpolation
        df = df.resample("H").mean()
        df["water_level"] = df["water_level"].interpolate(method="time")
        df.dropna(inplace=True)

        logging.info(f"✅ Retrieved {len(df)} hours of water level data")

        return df

    except Exception as e:
        logging.error(f"❌ Error loading tide data for {station_id}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return pd.DataFrame()


def check_station_has_live_data(station_id: str) -> bool:
    """
    Checks if a station has recent live data available (within last 24 hours).

    Args:
        station_id: NOAA station ID

    Returns:
        True if station has recent data, False otherwise
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=24)

        df = get_tide_data(
            station_id,
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d")
        )

        if df.empty:
            logging.warning(f"⚠️ Station {station_id} has no recent live data")
            return False

        # Check if data is recent (within last 6 hours)
        df_recent = df[df.index > (end_date - timedelta(hours=6))]

        if df_recent.empty:
            logging.warning(f"⚠️ Station {station_id} data is stale (older than 6 hours)")
            return False

        logging.info(f"✅ Station {station_id} has recent live data")
        return True

    except Exception as e:
        logging.error(f"❌ Error checking station {station_id}: {e}")
        return False


def classify_flood_risk(df: pd.DataFrame, flood_levels: dict) -> pd.DataFrame:
    """
    Assigns flood-risk level (Major, Moderate, Minor, No Flood) based on thresholds.

    Args:
        df: DataFrame with 'water_level' column
        flood_levels: dict with Major, Moderate, Minor threshold values

    Returns:
        DataFrame with added 'risk_level' column
    """
    if df.empty:
        logging.warning("⚠️ Cannot classify flood risk: DataFrame is empty")
        return df

    if not flood_levels or 'Minor' not in flood_levels:
        logging.warning("⚠️ No valid flood levels provided, setting risk to 'Unknown'")
        df['risk_level'] = "Unknown"
        return df

    def classify(level):
        if pd.isna(level):
            return "Unknown"
        if level >= flood_levels.get('Major', 999):
            return "Major"
        if level >= flood_levels.get('Moderate', 999):
            return "Moderate"
        if level >= flood_levels.get('Minor', 999):
            return "Minor"
        return "No Flood"

    df["risk_level"] = df["water_level"].apply(classify)

    # Log distribution of risk levels
    risk_counts = df['risk_level'].value_counts()
    logging.info(f"📊 Risk level distribution: {risk_counts.to_dict()}")

    return df
