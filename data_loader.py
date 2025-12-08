# data_loader.py

import logging
import os
import pandas as pd
from datetime import datetime, timedelta

from data_fetcher import fetch_noaa_water

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ----------------------------------------------------
# LOAD MASTER COASTAL STATIONS
# ----------------------------------------------------
def load_master_stations(csv_path="master_coastal_stations.csv"):
    logging.info(f"Loading master station list from '{csv_path}'...")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "station_id" not in df.columns:
        raise ValueError("CSV missing required 'station_id' column.")

    logging.info(f"✅ Loaded and normalized {len(df)} stations.")
    return df


# ----------------------------------------------------
# LOAD HISTORICAL NOAA WATER-LEVEL DATA
# ----------------------------------------------------
def load_noaa_historical(station_id: str, years: int = 1):
    """Retrieve and save NOAA COOPS historical water-level data."""

    logging.info(f"--- Starting historical load for station {station_id} ({years} years) ---")

    end = datetime.utcnow()
    start = end - timedelta(days=years * 365)

    df = fetch_noaa_water(
        station_id,
        start.strftime("%Y%m%d"),
        end.strftime("%Y%m%d")
    )

    if df is None or df.empty:
        logging.warning("⚠️ No water-level data retrieved from NOAA.")
        return None

    df = df.sort_values("date_time")

    out_dir = "data/historical"
    os.makedirs(out_dir, exist_ok=True)

    out_path = f"{out_dir}/{station_id}_historical_{years}y.csv"
    df.to_csv(out_path, index=False)

    logging.info(f"✅ NOAA historical dataset saved: {out_path}")
    return df


# ----------------------------------------------------
# LOAD LAST N HOURS FROM CACHE (Required by predictor.py)
# ----------------------------------------------------
def load_last_hours_from_cache(station_id: str, hours: int = 72):
    """
    Loads the most recent N hours of water-level data
    from the locally saved historical CSV.
    """

    file_dir = "data/historical"
    candidates = [
        f"{file_dir}/{station_id}_historical_1y.csv",
        f"{file_dir}/{station_id}_historical_5y.csv",
        f"{file_dir}/{station_id}_historical.csv"
    ]

    df = None
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["date_time"])
            break

    if df is None:
        logging.error(f"No cached historical data found for {station_id}.")
        return pd.DataFrame()

    cutoff = datetime.utcnow() - timedelta(hours=hours)
    df = df[df["date_time"] >= cutoff]

    if df.empty:
        logging.warning(f"No cached water-level data within last {hours} hours.")
        return pd.DataFrame()

    return df[["date_time", "water_level"]]


# ----------------------------------------------------
# CALCULATE FLOOD THRESHOLDS FROM HISTORICAL DATA (NEW FUNCTION)
# ----------------------------------------------------
def calculate_flood_thresholds_from_data(station_id: str) -> dict:
    """
    Calculate station-specific flood thresholds from historical data.
    Uses statistical analysis of 1-year water level data.

    This is Option 2: Calculate from Historical Data (Fallback)

    Method:
    - Minor Flood: 95th percentile of water levels
    - Moderate Flood: 99th percentile of water levels
    - Major Flood: 95% of historical maximum

    Args:
        station_id: NOAA station ID

    Returns:
        dict with keys: Minor, Moderate, Major, mean, std, max, method
        Returns None if data not available
    """

    # Search for historical data file
    file_dir = "data/historical"
    candidates = [
        f"{file_dir}/{station_id}_historical_1y.csv",
        f"{file_dir}/{station_id}_historical_5y.csv",
        f"{file_dir}/{station_id}_historical.csv"
    ]

    df = None
    file_found = None
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            file_found = path
            break

    if df is None:
        logging.error(f"❌ No historical data found for {station_id}. Cannot calculate thresholds.")
        return None

    if 'water_level' not in df.columns or df.empty:
        logging.error(f"❌ Historical data for {station_id} is empty or missing 'water_level' column.")
        return None

    # Remove any NaN values
    df = df.dropna(subset=['water_level'])

    if len(df) < 100:
        logging.error(f"❌ Insufficient data for {station_id} ({len(df)} records). Need at least 100.")
        return None

    # Statistical analysis
    mean_wl = df['water_level'].mean()
    std_wl = df['water_level'].std()
    min_wl = df['water_level'].min()
    max_wl = df['water_level'].max()
    median_wl = df['water_level'].median()

    # Percentiles
    p95 = df['water_level'].quantile(0.95)
    p99 = df['water_level'].quantile(0.99)
    p999 = df['water_level'].quantile(0.999)

    # Define thresholds based on statistical distribution
    thresholds = {
        'Minor': float(p95),  # 95th percentile - occurs ~18 days per year
        'Moderate': float(p99),  # 99th percentile - occurs ~3.6 days per year
        'Major': float(p999),  # 99.9th percentile - occurs ~8.7 hours per year
        'mean': float(mean_wl),
        'std': float(std_wl),
        'min': float(min_wl),
        'max': float(max_wl),
        'median': float(median_wl),
        'method': 'statistical',
        'data_points': len(df),
        'source_file': file_found
    }

    logging.info(f"📊 Calculated statistical thresholds for {station_id}:")
    logging.info(f"   Data points: {len(df)}")
    logging.info(f"   Mean: {mean_wl:.2f} ft, Std: {std_wl:.2f} ft")
    logging.info(f"   Range: {min_wl:.2f} to {max_wl:.2f} ft")
    logging.info(f"   Minor (95th percentile): {thresholds['Minor']:.2f} ft")
    logging.info(f"   Moderate (99th percentile): {thresholds['Moderate']:.2f} ft")
    logging.info(f"   Major (99.9th percentile): {thresholds['Major']:.2f} ft")

    return thresholds
