# nws_fetcher.py

import pandas as pd
import logging
from nwsapy import api_connector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

api_connector.set_user_agent('Geospatial-Watches-And-Warnings-App', 'policapee@gmail.com')


def get_precipitation_forecast(lat: float, lon: float, hours: int = 72) -> pd.DataFrame:
    """
    Fetches NWS precipitation forecast for the next X hours.

    Args:
        lat: Latitude of NOAA station
        lon: Longitude of NOAA station
        hours: Forecast hours (default 72 = 3 days)

    Returns:
        DataFrame with columns: forecast_time, precipitation_probability, precipitation_amount
    """
    try:
        logging.info(f"🌧️ Fetching {hours}h precipitation forecast for ({lat}, {lon})...")

        # Get NWS grid point data
        point_data = api_connector.get_point(lat, lon)
        point_dict = point_data.to_dict()

        # Get hourly forecast URL
        forecast_hourly_url = point_dict.get('forecastHourly')

        if not forecast_hourly_url:
            logging.error("❌ No hourly forecast URL found")
            return pd.DataFrame()

        # Fetch hourly forecast
        response = api_connector.make_request(forecast_hourly_url)

        if response.status_code != 200:
            logging.error(f"❌ NWS API error: {response.status_code}")
            return pd.DataFrame()

        data = response.json()

        if 'properties' not in data or 'periods' not in data['properties']:
            logging.error("❌ Invalid forecast response")
            return pd.DataFrame()

        periods = data['properties']['periods'][:hours]

        forecast_data = []
        for period in periods:
            forecast_time = pd.to_datetime(period.get('startTime'))

            # Extract precipitation probability (%)
            precip_prob = period.get('probabilityOfPrecipitation', {})
            precip_probability = precip_prob.get('value', 0) if precip_prob else 0

            # Extract precipitation amount (if available)
            # Note: NWS doesn't always provide quantitative precipitation in hourly forecast
            # We'll use probability as a proxy
            precip_amount = precip_probability / 100.0  # Convert % to 0-1 scale

            forecast_data.append({
                'forecast_time': forecast_time,
                'precipitation_probability': precip_probability,  # 0-100%
                'precipitation_amount': precip_amount  # 0-1 scale for model
            })

        df = pd.DataFrame(forecast_data)

        if df.empty:
            logging.warning("⚠️ No precipitation forecast data")
            return pd.DataFrame()

        logging.info(f"✅ Fetched {len(df)} hours of precipitation forecast")
        logging.info(f"   Max precip probability: {df['precipitation_probability'].max()}%")

        return df

    except Exception as e:
        logging.error(f"❌ Error fetching precipitation forecast: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return pd.DataFrame()


def calculate_precipitation_impact(precip_df: pd.DataFrame) -> dict:
    """
    Calculates the expected water level rise from forecasted precipitation.

    Simple heuristic:
    - Low precip (0-30%): +0.0 ft
    - Moderate precip (30-60%): +0.5 ft
    - High precip (60-80%): +1.0 ft
    - Extreme precip (80-100%): +2.0 ft

    Args:
        precip_df: DataFrame from get_precipitation_forecast()

    Returns:
        dict with precipitation impact metrics
    """
    if precip_df.empty:
        return {
            'max_precip_probability': 0,
            'avg_precip_probability': 0,
            'expected_water_rise_ft': 0.0,
            'flood_risk_multiplier': 1.0,
            'hours_with_precip': 0
        }

    max_prob = precip_df['precipitation_probability'].max()
    avg_prob = precip_df['precipitation_probability'].mean()

    # Calculate expected water rise based on precipitation
    if max_prob >= 80:
        water_rise = 2.0
        risk_multiplier = 1.5
    elif max_prob >= 60:
        water_rise = 1.0
        risk_multiplier = 1.3
    elif max_prob >= 30:
        water_rise = 0.5
        risk_multiplier = 1.1
    else:
        water_rise = 0.0
        risk_multiplier = 1.0

    return {
        'max_precip_probability': float(max_prob),
        'avg_precip_probability': float(avg_prob),
        'expected_water_rise_ft': water_rise,
        'flood_risk_multiplier': risk_multiplier,
        'hours_with_precip': int((precip_df['precipitation_probability'] > 30).sum())
    }
