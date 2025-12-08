# data_fetcher.py

import logging
import pandas as pd
import noaa_coops as nc
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ----------------------------------------------------
# NOAA HISTORICAL WATER LEVEL FETCHER
# ----------------------------------------------------
def fetch_noaa_water(station_id: str, start_date: str, end_date: str):
    """Fetch and normalize NOAA CO-OPS water-level data."""

    logging.info(f"🌊 Fetching NOAA water levels: station={station_id}, {start_date} → {end_date}")

    try:
        coops = nc.Station(station_id)
        df = coops.get_data(
            begin_date=start_date,
            end_date=end_date,
            product="water_level",
            datum="MLLW",  # Mean Lower Low Water (standard for US coastal stations)
            units="english",  # Feet (not meters)
            time_zone="lst_ldt"  # Local Standard/Daylight Time (valid timezone)
        )
    except Exception as e:
        logging.error(f"❌ NOAA request failed: {e}")
        return None

    if df is None or df.empty:
        logging.error("❌ NOAA returned an empty dataframe.")
        return None

    # ----------------------------------------------------
    # FLEXIBLE COLUMN DETECTION (handles version differences)
    # ----------------------------------------------------

    # The noaa_coops library returns data with index as datetime
    # and columns that may vary by version

    # Reset index to make datetime a column
    df = df.reset_index()

    time_col = None
    value_col = None

    # Check for time column
    for col in ['date_time', 't', 'time', 'datetime', 'index']:
        if col in df.columns:
            time_col = col
            break

    # Check for value column
    for col in ['water_level', 'v', 'value', 'water level']:
        if col in df.columns:
            value_col = col
            break

    if time_col is None or value_col is None:
        logging.error(f"❌ NOAA response missing expected columns. Found: {df.columns.tolist()}")
        return None

    # Rename to standard names
    df = df.rename(columns={time_col: "date_time", value_col: "water_level"})

    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df["water_level"] = pd.to_numeric(df["water_level"], errors="coerce")

    df = df.dropna(subset=["date_time", "water_level"])

    if df.empty:
        logging.error("❌ After cleaning NOAA data, no valid water levels remain.")
        return None

    return df[["date_time", "water_level"]]


# ----------------------------------------------------
# GET STATION METADATA (NEW FUNCTION)
# ----------------------------------------------------
def get_station_metadata(station_id: str) -> dict:
    """
    Fetch comprehensive station metadata including elevation and flood thresholds.

    Returns:
        dict with keys: station_id, elevation_ft, datum, flood_thresholds, metadata_available
    """
    try:
        coops = nc.Station(station_id)
        meta = coops.metadata

        metadata = {
            'station_id': station_id,
            'elevation_ft': None,
            'datum': None,
            'flood_thresholds': None,
            'metadata_available': False
        }

        if meta:
            metadata['metadata_available'] = True

            # Extract elevation (various possible keys)
            for key in ['elevation', 'met_site_elevation', 'height', 'sensor_elevation']:
                if key in meta:
                    try:
                        metadata['elevation_ft'] = float(meta[key])
                        logging.info(f"📏 Station {station_id} elevation: {metadata['elevation_ft']:.2f} ft")
                        break
                    except (ValueError, TypeError):
                        continue

            # Check for datum
            if 'datum' in meta:
                metadata['datum'] = meta['datum']

            # Check for flood levels
            if 'flood_levels' in meta:
                metadata['flood_thresholds'] = meta['flood_levels']
                logging.info(f"🌊 Station {station_id} has NOAA flood thresholds")

        return metadata

    except Exception as e:
        logging.error(f"❌ Error fetching station metadata for {station_id}: {e}")
        return {
            'station_id': station_id,
            'elevation_ft': None,
            'datum': None,
            'flood_thresholds': None,
            'metadata_available': False
        }


# ----------------------------------------------------
# NOAA FLOOD THRESHOLDS (TWO-TIER APPROACH)
# ----------------------------------------------------
def get_flood_levels_from_noaa(station_id: str) -> dict:
    """
    Get flood thresholds using two-tier approach:
    1. Try NOAA metadata first (most accurate, station-specific)
    2. Fall back to statistical calculation from historical data

    NO GENERIC DEFAULTS - Always uses real data.

    Returns:
        dict with keys: Minor, Moderate, Major, method
        Returns None if both methods fail
    """

    # ============================================================
    # OPTION 1: Try NOAA metadata first (MOST ACCURATE)
    # ============================================================
    try:
        coops = nc.Station(station_id)
        meta = coops.metadata

        if meta and "flood_levels" in meta:
            thresholds = meta["flood_levels"]
            thresholds['method'] = 'noaa_metadata'
            logging.info(f"✅ Using NOAA official flood thresholds for {station_id}")
            logging.info(
                f"   Minor: {thresholds.get('Minor', 'N/A')} ft, Moderate: {thresholds.get('Moderate', 'N/A')} ft, Major: {thresholds.get('Major', 'N/A')} ft")
            return thresholds
    except Exception as e:
        logging.warning(f"⚠️ Could not fetch NOAA metadata: {e}")

    # ============================================================
    # OPTION 2: Calculate from historical data (FALLBACK)
    # ============================================================
    try:
        # Import here to avoid circular dependency
        from data_loader import calculate_flood_thresholds_from_data

        thresholds = calculate_flood_thresholds_from_data(station_id)
        if thresholds:
            logging.info(f"✅ Using statistically calculated flood thresholds for {station_id}")
            logging.info(
                f"   Minor: {thresholds['Minor']:.2f} ft, Moderate: {thresholds['Moderate']:.2f} ft, Major: {thresholds['Major']:.2f} ft")
            return thresholds
    except ImportError:
        logging.error("❌ data_loader module not available for threshold calculation")
    except Exception as e:
        logging.warning(f"⚠️ Could not calculate thresholds from historical data: {e}")

    # ============================================================
    # BOTH METHODS FAILED - Return None
    # ============================================================
    logging.error(f"❌ Could not determine flood thresholds for {station_id} using any method.")
    logging.error(f"   Please ensure historical data is loaded or NOAA metadata is available.")
    return None


# ----------------------------------------------------
# CLASSIFY FLOOD RISK
# ----------------------------------------------------
def classify_flood_risk(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """
    Adds 'risk_level' column based on flood thresholds.

    Args:
        df: DataFrame with 'water_level' column
        thresholds: dict with 'Minor', 'Moderate', 'Major' keys (or None)

    Returns:
        DataFrame with added 'risk_level' column
    """

    df = df.copy()

    # If no thresholds available, mark as Unknown
    if thresholds is None:
        df["risk_level"] = "Unknown (No Thresholds Available)"
        logging.warning("⚠️ Flood classification unavailable - no thresholds provided")
        return df

    def classify(level):
        if "Major" in thresholds and level >= thresholds["Major"]:
            return "Major Flood"
        if "Moderate" in thresholds and level >= thresholds["Moderate"]:
            return "Moderate Flood"
        if "Minor" in thresholds and level >= thresholds["Minor"]:
            return "Minor Flood"
        return "Normal"

    df["risk_level"] = df["water_level"].apply(classify)

    # Add threshold method info if available
    if 'method' in thresholds:
        logging.info(f"📊 Flood classification using '{thresholds['method']}' thresholds")

    return df
