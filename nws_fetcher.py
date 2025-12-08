# nws_fetcher.py
"""
Fetch precipitation forecast from National Weather Service API.
Returns hourly precipitation probability and/or amounts for the next 24 hours,
with aggregation support for 6-hour intervals.
"""

import logging
import requests
import pandas as pd
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------
# NWS PRECIPITATION FORECAST (ENHANCED FOR 24 HOURS)
# ---------------------------------------------------------
def get_precipitation_forecast(lat: float, lon: float, hours: int = 24) -> pd.DataFrame:
    """
    Fetch precipitation forecast from NWS API for given coordinates [2][6].

    Args:
        lat: Latitude
        lon: Longitude
        hours: Number of hours to forecast (default 24, max recommended 36 for accuracy)

    Returns:
        DataFrame with columns: valid_time, precipitation_probability, precipitation_amount
    """

    logging.info(f"🌧️ Fetching NWS precipitation forecast for ({lat}, {lon}), {hours} hours")

    # Step 1: Get grid point metadata [2][6]
    points_url = f"https://api.weather.gov/points/{lat},{lon}"

    try:
        headers = {'User-Agent': '(NOAA Coastal Flood Viewer, contact@example.com)'}
        response = requests.get(points_url, headers=headers, timeout=10)
        response.raise_for_status()
        points_data = response.json()
    except Exception as e:
        logging.error(f"❌ NWS points API failed: {e}")
        return pd.DataFrame()

    # Step 2: Extract forecastGridData URL [2]
    if 'properties' not in points_data or 'forecastGridData' not in points_data['properties']:
        logging.error("❌ NWS response missing forecastGridData URL")
        return pd.DataFrame()

    grid_data_url = points_data['properties']['forecastGridData']

    # Step 3: Fetch grid data [2]
    try:
        response = requests.get(grid_data_url, headers=headers, timeout=10)
        response.raise_for_status()
        grid_data = response.json()
    except Exception as e:
        logging.error(f"❌ NWS grid data API failed: {e}")
        return pd.DataFrame()

    # Step 4: Extract precipitation data [2][3]
    properties = grid_data.get('properties', {})

    # Try quantitativePrecipitation first (hourly amounts) [3]
    precip_data = properties.get('quantitativePrecipitation', {}).get('values', [])

    # Fallback to probabilityOfPrecipitation [2]
    prob_data = properties.get('probabilityOfPrecipitation', {}).get('values', [])

    if not precip_data and not prob_data:
        logging.warning("⚠️ No precipitation data available from NWS")
        return pd.DataFrame()

    # Step 5: Parse precipitation data
    records = []
    cutoff_time = datetime.utcnow() + timedelta(hours=hours)

    # Parse quantitative precipitation [3]
    for item in precip_data:
        valid_time_str = item.get('validTime', '')
        value = item.get('value')

        if not valid_time_str or value is None:
            continue

            # Parse ISO 8601 duration format: "2025-12-05T18:00:00+00:00/PT1H"
            try:
                time_part = valid_time_str.split('/')
                valid_time = pd.to_datetime(time_part)

                if valid_time > cutoff_time:
                    break

                # Convert mm to inches
                precip_inches = value * 0.0393701 if value else 0.0

                records.append({
                    'valid_time': valid_time,
                    'precipitation_amount': precip_inches
                })
            except Exception:
                continue

        # Parse probability of precipitation
        prob_records = []
        for item in prob_data:
            valid_time_str = item.get('validTime', '')
            value = item.get('value')

            if not valid_time_str or value is None:
                continue

            try:
                time_part = valid_time_str.split('/')
                valid_time = pd.to_datetime(time_part)

                if valid_time > cutoff_time:
                    break

                prob_records.append({
                    'valid_time': valid_time,
                    'precipitation_probability': value
                })
            except Exception:
                continue

        # Step 6: Combine into DataFrame
        df_amount = pd.DataFrame(records) if records else pd.DataFrame()
        df_prob = pd.DataFrame(prob_records) if prob_records else pd.DataFrame()

        if df_amount.empty and df_prob.empty:
            logging.warning("⚠️ No valid precipitation records parsed")
            return pd.DataFrame()

        # Merge on valid_time if both exist
        if not df_amount.empty and not df_prob.empty:
            df = pd.merge(df_amount, df_prob, on='valid_time', how='outer')
        elif not df_amount.empty:
            df = df_amount
        else:
            df = df_prob

        df = df.sort_values('valid_time').reset_index(drop=True)

        logging.info(f"✅ Retrieved {len(df)} precipitation forecast records")
        return df

    # ---------------------------------------------------------
    # NEW: AGGREGATE PRECIPITATION BY 6-HOUR INTERVALS
    # ---------------------------------------------------------
    def aggregate_precipitation_by_interval(precip_df: pd.DataFrame, interval_hours: int = 6) -> list:
        """
        Aggregate precipitation forecast into fixed time intervals (e.g., 6-hour buckets).

        Args:
            precip_df: DataFrame from get_precipitation_forecast()
            interval_hours: Size of each interval in hours (default 6)

        Returns:
            List of dicts, one per interval:
            [
                {
                    'interval_start': datetime,
                    'interval_end': datetime,
                    'interval_label': '0-6h',
                    'max_probability': float,
                    'avg_probability': float,
                    'total_precipitation': float (inches),
                    'hours_with_precip': int,
                    'expected_water_rise_ft': float
                },
                ...
            ]
        """

        if precip_df is None or precip_df.empty:
            logging.warning("⚠️ No precipitation data to aggregate")
            return []

        # Ensure valid_time is datetime
        if 'valid_time' not in precip_df.columns:
            logging.error("❌ Precipitation data missing 'valid_time' column")
            return []

        precip_df['valid_time'] = pd.to_datetime(precip_df['valid_time'])

        # Calculate number of intervals needed for 24 hours
        num_intervals = 24 // interval_hours  # 24 / 6 = 4 intervals

        # Get current time as reference
        now = datetime.utcnow()

        intervals = []

        for i in range(num_intervals):
            interval_start = now + timedelta(hours=i * interval_hours)
            interval_end = now + timedelta(hours=(i + 1) * interval_hours)

            # Filter data for this interval
            mask = (precip_df['valid_time'] >= interval_start) & (precip_df['valid_time'] < interval_end)
            interval_data = precip_df[mask]

            # Create interval label
            start_hour = i * interval_hours
            end_hour = (i + 1) * interval_hours
            interval_label = f"{start_hour}-{end_hour}h"

            # Initialize interval summary
            interval_summary = {
                'interval_start': interval_start,
                'interval_end': interval_end,
                'interval_label': interval_label,
                'max_probability': 0.0,
                'avg_probability': 0.0,
                'total_precipitation': 0.0,
                'hours_with_precip': 0,
                'expected_water_rise_ft': 0.0
            }

            if interval_data.empty:
                logging.warning(f"⚠️ No precipitation data for interval {interval_label}")
                intervals.append(interval_summary)
                continue

            # Calculate statistics for probability
            if 'precipitation_probability' in interval_data.columns:
                probs = interval_data['precipitation_probability'].dropna()
                if not probs.empty:
                    interval_summary['max_probability'] = float(probs.max())
                    interval_summary['avg_probability'] = float(probs.mean())
                    interval_summary['hours_with_precip'] = int((probs > 30).sum())

            # Calculate statistics for amount
            if 'precipitation_amount' in interval_data.columns:
                amounts = interval_data['precipitation_amount'].dropna()
                if not amounts.empty:
                    interval_summary['total_precipitation'] = float(amounts.sum())

            # Calculate expected water rise based on precipitation
            interval_summary['expected_water_rise_ft'] = _calculate_water_rise_from_precip(
                interval_summary['max_probability'],
                interval_summary['total_precipitation']
            )

            intervals.append(interval_summary)

            logging.info(
                f"📊 Interval {interval_label}: "
                f"Max Prob={interval_summary['max_probability']:.0f}%, "
                f"Total Precip={interval_summary['total_precipitation']:.2f}in, "
                f"Water Rise={interval_summary['expected_water_rise_ft']:.2f}ft"
            )

        return intervals

    # ---------------------------------------------------------
    # NEW: CALCULATE WATER RISE FROM PRECIPITATION
    # ---------------------------------------------------------
    def _calculate_water_rise_from_precip(max_probability: float, total_precip_inches: float) -> float:
        """
        Calculate expected water level rise based on precipitation forecast.

        Uses a heuristic model combining probability and amount.

        Args:
            max_probability: Maximum precipitation probability in interval (0-100)
            total_precip_inches: Total precipitation amount in inches

        Returns:
            Expected water rise in feet
        """

        # Base calculation on probability
        if max_probability >= 80:
            prob_factor = 2.0
        elif max_probability >= 60:
            prob_factor = 1.5
        elif max_probability >= 40:
            prob_factor = 1.0
        elif max_probability >= 20:
            prob_factor = 0.5
        else:
            prob_factor = 0.0

        # Base calculation on amount (if available)
        # Rule of thumb: 1 inch of rain can raise water level by ~0.5 feet in coastal areas
        amount_factor = total_precip_inches * 0.5 if total_precip_inches > 0 else 0.0

        # Combine both factors (use maximum of the two approaches)
        water_rise = max(prob_factor, amount_factor)

        return round(water_rise, 2)

    # ---------------------------------------------------------
    # GET 24-HOUR FORECAST SUMMARY
    # ---------------------------------------------------------
    def get_24h_precipitation_summary(lat: float, lon: float) -> dict:
        """
        Get a comprehensive 24-hour precipitation forecast summary.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Dict with keys:
            - 'raw_forecast': Full hourly DataFrame
            - 'intervals': List of 6-hour interval summaries
            - 'total_expected_rise': Total cumulative water rise over 24h
            - 'max_risk_interval': Interval with highest precipitation risk
            - 'status': 'available', 'unavailable', or 'no_coords'
        """

        logging.info(f"🌧️ Fetching 24-hour precipitation summary for ({lat}, {lon})")

        # Fetch 24-hour forecast
        precip_df = get_precipitation_forecast(lat, lon, hours=24)

        if precip_df.empty:
            logging.warning("⚠️ No precipitation forecast data available")
            return {
                'raw_forecast': pd.DataFrame(),
                'intervals': [],
                'total_expected_rise': 0.0,
                'max_risk_interval': None,
                'status': 'unavailable'
            }

        # Aggregate into 6-hour intervals
        intervals = aggregate_precipitation_by_interval(precip_df, interval_hours=6)

        if not intervals:
            logging.warning("⚠️ Could not aggregate precipitation into intervals")
            return {
                'raw_forecast': precip_df,
                'intervals': [],
                'total_expected_rise': 0.0,
                'max_risk_interval': None,
                'status': 'unavailable'
            }

        # Calculate total expected rise (cumulative)
        total_rise = sum(interval['expected_water_rise_ft'] for interval in intervals)

        # Find interval with highest risk
        max_risk_interval = max(intervals, key=lambda x: x['expected_water_rise_ft'])

        logging.info(f"✅ 24h precipitation summary complete:")
        logging.info(f"   Total expected rise: {total_rise:.2f} ft")
        logging.info(
            f"   Max risk interval: {max_risk_interval['interval_label']} (+{max_risk_interval['expected_water_rise_ft']:.2f} ft)")

        return {
            'raw_forecast': precip_df,
            'intervals': intervals,
            'total_expected_rise': round(total_rise, 2),
            'max_risk_interval': max_risk_interval,
            'status': 'available'
        }
