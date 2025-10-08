import requests
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CSV filename ---
stations_csv = "all_noaa_coops_stations_xml.csv"

# --- NOAA XML URL (Corrected) ---
# This is the URL where NOAA serves the current prediction stations data in XML format.
stations_url = "https://opendap.co-ops.nos.noaa.gov/axis/webservices/currentpredictionstations/response.jsp?format=xml"

def get_stations_df():
    """
    Fetch NOAA CO-OPS station data from XML, parse into a DataFrame,
    save locally as CSV, and return the DataFrame.
    """
    logging.info("Attempting to get NOAA stations data.")

    # If CSV already exists and is non-empty, load it
    csv_path = Path(stations_csv)
    if csv_path.is_file() and csv_path.stat().st_size > 0:
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                logging.info(f"Loaded stations data from existing CSV: {stations_csv}")
                return df
            else:
                logging.warning(f"Existing CSV '{stations_csv}' was empty. Fetching new data.")
        except pd.errors.EmptyDataError:
            logging.warning(f"Existing CSV '{stations_csv}' is empty or malformed. Fetching new data.")
        except Exception as e:
            logging.error(f"Error reading existing CSV '{stations_csv}': {e}. Fetching new data.")

    try:
        logging.info(f"Fetching XML data from NOAA URL: {stations_url}")
        # Fetch XML data from NOAA
        response = requests.get(stations_url, timeout=30) # Increased timeout for potentially slow responses
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

        # Parse XML
        root = ET.fromstring(response.content)

        # Namespaces used in the XML
        # Note: The 'ns' namespace might vary slightly depending on the exact XML structure.
        # Based on your notebook and the corrected URL, this should be correct.
        namespaces = {
            "soapenv": "http://schemas.xmlsoap.org/soap/envelope/",
            "ns": "https://opendap.co-ops.nos.noaa.gov/axis/webservices/currentpredictionstations/wsdl"
        }

        stations_list = []

        # Loop through stations
        for station in root.findall(".//ns:station", namespaces):
            station_dict = {
                "station_id": station.attrib.get("ID"),
                "station_name": station.attrib.get("name"),
                "bin_number": station.findtext(".//ns:binNumber", namespaces=namespaces),
                "depth": station.findtext(".//ns:depth", namespaces=namespaces),
                "latitude": station.findtext(".//ns:lat", namespaces=namespaces),
                "longitude": station.findtext(".//ns:long", namespaces=namespaces),
                "station_type": station.findtext(".//ns:stationType", namespaces=namespaces)
            }
            stations_list.append(station_dict)

        # Create DataFrame
        stations_df = pd.DataFrame(stations_list)

        # Convert numeric columns, coercing errors will turn invalid parsing into NaN
        stations_df["latitude"] = pd.to_numeric(stations_df["latitude"], errors="coerce")
        stations_df["longitude"] = pd.to_numeric(stations_df["longitude"], errors="coerce")
        stations_df["bin_number"] = pd.to_numeric(stations_df["bin_number"], errors="coerce")
        stations_df["depth"] = pd.to_numeric(stations_df["depth"], errors="coerce")

        # Drop rows where essential coordinates are missing
        stations_df.dropna(subset=['latitude', 'longitude'], inplace=True)

        # Save to CSV
        stations_df.to_csv(csv_path, index=False)
        logging.info(f"Successfully fetched new data and saved to CSV: {stations_csv}")

        return stations_df

    except requests.exceptions.RequestException as req_err:
        logging.error(f"Network or HTTP error fetching NOAA stations: {req_err}")
        return pd.DataFrame()  # return empty DataFrame on error
    except ET.ParseError as parse_err:
        logging.error(f"XML parsing error from NOAA response: {parse_err}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"An unexpected error occurred while fetching/processing NOAA stations: {e}")
        return pd.DataFrame()  # return empty DataFrame on error