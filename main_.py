import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from stations_df_func_ import get_stations_df

# --- US State to Region Mapping ---
# This dictionary helps categorize states into broader regions for the dropdown.
STATE_TO_REGION = {
    # Northeast
    'Connecticut': 'Northeast', 'Delaware': 'Northeast', 'Maine': 'Northeast',
    'Maryland': 'Northeast', 'Massachusetts': 'Northeast', 'New Hampshire': 'Northeast',
    'New Jersey': 'Northeast', 'New York': 'Northeast', 'Pennsylvania': 'Northeast',
    'Rhode Island': 'Northeast', 'Vermont': 'Northeast',

    # Southeast
    'Alabama': 'Southeast', 'Arkansas': 'Southeast', 'Florida': 'Southeast',
    'Georgia': 'Southeast', 'Kentucky': 'Southeast', 'Louisiana': 'Southeast',
    'Mississippi': 'Southeast', 'North Carolina': 'Southeast', 'South Carolina': 'Southeast',
    'Tennessee': 'Southeast', 'Virginia': 'Southeast', 'West Virginia': 'Southeast',

    # Midwest
    'Illinois': 'Midwest', 'Indiana': 'Midwest', 'Iowa': 'Midwest',
    'Kansas': 'Midwest', 'Michigan': 'Midwest', 'Minnesota': 'Midwest',
    'Missouri': 'Midwest', 'Nebraska': 'Midwest', 'North Dakota': 'Midwest',
    'Ohio': 'Midwest', 'South Dakota': 'Midwest', 'Wisconsin': 'Midwest',

    # Southwest
    'Arizona': 'Southwest', 'New Mexico': 'Southwest', 'Oklahoma': 'Southwest',
    'Texas': 'Southwest',

    # West (Mountain & Pacific)
    'Alaska': 'West', 'California': 'West', 'Colorado': 'West',
    'Hawaii': 'West', 'Idaho': 'West', 'Montana': 'West',
    'Nevada': 'West', 'Oregon': 'West', 'Utah': 'West',
    'Washington': 'West', 'Wyoming': 'West',

    # Special Case: District of Columbia (often grouped with Northeast/Mid-Atlantic)
    'District of Columbia': 'Northeast'
}

# Reverse mapping for easy lookup: Region to States
REGION_TO_STATES = {}
for state, region in STATE_TO_REGION.items(): read_csv(csv_path)
# Ensure 'State' and 'Region' columns exist if
REGION_TO_STATES.setdefault(region, []).append(loading
from CSV

if 'State'state)
for region in REGION_TO_STATES:
    REGION_TO_STATES in df.columns and 'Region' in df.columns and not df.empty:
    logging.info(f"Loaded stations data[region].sort() # Sort states within each region

    # --- Streamlit from existing CSV: {stations_csv}")
    return df
else page setup ---
st.set_page_config(:
logging.warning(f"Existing CSV '{stationslayout="
wide
", page_title="
NOAA
CO - _csv}' missing '
State
' or '
Region
' column,OPS Station Viewer")
st.title("🌊 or is empty. Fetching new data.")
except pd.errors.NOAA
CO - OPS
Station
Viewer
")
st.markdown("EmptyDataError:
logging.warning(
    f"Existing CSV '{stations_csv}' is empty or malformedSelect a region, state, and station to view its location.")

# --- Cached function to load. Fetching new data.")
except Exception as e:
logging.error(f
stations
DataFrame - --
@ st.cache_data(ttl=3600, show_
"Error reading existing CSV '{stations_csv}': {e}. Fetching new data.")

try:
    logging.info(f"spinner="
    Loading
    NOAA
    station
    data...
    ")


def cached_get_stations_df():
    """
    Wrapper for get_stations_df to leverage StreamFetching XML data from NOAA URL: {stations_url}")
        response =lit's caching.
    Adds 'region' and 'state requests.get(stations_url, timeout=30)
        response' columns to the DataFrame.
    """
    .raise_for_status()

    root = ET.fromstring(response.content)

    namespaces = {
        "soapenv": "logging.info("Calling get_stations_df(might be
    from cache or fresh
    fetch).")
    df = get_stations_df()
    http: // schemas.xmlsoap.org / soap / envelope / ",
                                                     "

    if not df.empty:
    # ns": "https://opendap.co-ops.nos.noaa.gov/axis/webservices/currentpredictionstations/wsdl"
        }
    Add
    a
    'state'
    column
    based
    on
    reverse
    geocoding or a
    lookup

    stations_list = []

    for station in
        # For simplicity, we' root.findall(".//ns:station", namespaces):
        station_dict = {
            "station_id": ll assign a dummy state if actual state data isn't in station.attrib.get("ID"),
                                                                          "station_name": station
        XML
        # In a real app, you'd.attrib.get("name"),
        "bin_number": station.findtext(".//ns use a geocoding service for accurate state assignment
                                       # For:binNumber", namespaces=namespaces),
                                       "depth": station.find
        now, let
        's assume we can derivetext(".//ns:depth", namespaces=namespaces),
        "latitude": station.findtext(".//ns:lat", state
        from station_name or add
        it
        manually.
        # Since namespaces=namespaces),
        "longitude": station.find
        the
        NOAA
        data
        doesn
        't directlytext(".//ns:long", namespaces=namespaces),
        "station_type": station.findtext(".//ns:stationType", namespaces=namespaces)
        }
        stations_list.append(station_dict)

        provide
        'State', we
        'll approximate or assume
        # a way to get it. For this example, let's addstations_df = pd.DataFrame(stations_list)

        stations_df["latitude"] = pd.to_numeric(stations_df["latitude a placeholder 'State' column
        # and then map it to regions"], errors="coerce")
        stations.

        # IMPORTANT: The NOAA XML does_df["longitude"] = pd.to_numeric(stations_df["longitude"], errors="coerce")
        stations_df["bin_number"] = pd.to_numeric(stations_df["bin not directly provide a 'State' column.
        # You would need to either:
        # 1. Reverse_number"], errors="coerce")
        stations_df
        geocode
        the
        lat["depth"] = pd.to_numeric(stations_df["depth"], errors="coerce")

        stations_ / long
        for each station(API call, can be slow / costly).
        # 2. Maintain a separate mappingdf.dropna(subset=['latitude', 'longitude'], inplace=True)

        # --- NEW of station_id to: Add 'State' and 'Region' columns ---
        # This is a state.
        # 3. Infer state from station_name if it's consistently included.
        # critical step. NOAA's XML doesn't directly provide state.
        # You would For this example, I'll add a placeholder 'State' column and
        # demonstrate typically need a reverse geocoding library (like geopy)
        # or a lookup table based on lat/lon ranges.
        # For demonstration, I how to integrate it. For a robust solution, you'll need
        'll add a placeholder.
        # A robust# to implement one of the above methods to populate the 'State' column accurately solution needs actual state assignment.
        # For now, let's try to infer a state.

        # Placeholder from the station name if possible,
        # or assign a default if not.

        # *** IMPORTANT: You need a for 'State' - you need to replace this with actual logic
        # For reliable way to get the 'State' for each station. ***
        # Option 1: Manual demonstration, let's assign a default or try to infer from mapping if you have a small name
        df['state'] =, fixed set of stations.
        df['station_name'].apply( lambda x: "California" if "CA# Option 2: Reverse geocoding (e.g., using geopy and Nominatim, but be mindful of API limits" in x or "San Francisco" in x else
        "Florida" if "FL" in).
        # Option 3: A pre-compiled list of NOAA stations with their states.
        # Option 4: A more advanced x or "Miami" in x else
        "New York" if "NY" in x or "New York" in x else
        "Washington" if geospatial
        lookup
        using
        a
        shapefile.

        # For this example, I' "WA" in x or "Seattle" in x else
        "Texas" if "TX" in xll
        create
        a
        dummy
        'State'
        column. or "Galveston" in x else
        "Maryland" if "MD" in x or "Baltimore
        # In a real application, you'd replace this with actual logic.
        # Let" in x else
                                   "'s try to extract a state abbreviation from the station name if it's common.
        Massachusetts
        " if "
        MA
        " in x or "
        Boston
        " in x else
        "Virginia" if "VA" in x or  # This is very brittle "Norfolk" in x else
        and will
        likely
        need
        manual
        "Louisiana" if "LA" in x
        orYou
        're absolutely right! When dealing with a large number of markers, displaying them all at once can indeed slow down the application and make the map cluttered. A hierarchical dropdown menu (Region -> State -> Station) is an excellent solution for improving both performance and user experience.

        We
        'll adapt your provided `app.py` structure to our NOAA station data.

        First, we
        need
        to
        categorize
        each
        state
        into
        a
        region.I
        'll use common U.S. regional groupings.

        ### 1. Update `stations_df_func.py` to Include Region Data

        We
        need
        to
        add
        a
        'Region'
        column
        to
        our
        DataFrame.This is best
        done
        when
        the
        data is initially
        processed.

        ```python
        import requests
        import pandas as pd
        import xml.etree.ElementTree as ET
        from pathlib import Path
        import logging

        # Set up basic logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # --- CSV filename ---
        stations_csv = "all_noaa_coops_stations_xml.csv"

        # --- NOAA XML URL ---
        stations_url = "https://opendap.co-ops.nos.noaa.gov/axis/webservices/currentpredictionstations/response.jsp?format=xml"

        # --- State to Region Mapping ---
        # This dictionary maps each state to a broader geographical region.
        STATE_TO_REGION = {
            'AL': 'Southeast', 'AK': 'Alaska', 'AZ': 'Southwest', 'AR': 'Southeast', 'CA': 'Pacific Southwest',
            'CO': 'Mountain West', 'CT': 'Northeast', 'DE': 'Mid-Atlantic',
            'DC': 'Mid refinement or a proper geocoding step.

    def infer_state_from_name(station_name):


"New Orleans" in x else
"Alaska" if "AK" in x or "Anchorage" in x else
-Atlantic
', '
FL
': '
Southeast
',
'GA': 'Southeast', 'HI': 'Pacific Southwest', 'ID': 'Mountain West', 'IL': 'Midwest', 'IN': 'Midwest',
'IA': 'Midwest', 'KS': 'Midwest', 'KY': 'Southeast', 'LA': 'Southeast', 'ME': 'Northeast',
'MD': 'Mid-Atlantic', 'MA': 'Northeast', 'MI': 'Midwest', 'MN': 'Midwest', 'MS': 'Southeast',
'MO': 'Midwest', 'MT': 'Mountain West', 'NE': 'Midwest', 'NV': 'Pacific Southwest', 'NH': 'Northeast',
'NJ': 'Mid-Atlantic', 'NM': 'Southwest', 'NY': 'Northeast', 'NC': 'Southeast', 'ND': 'Midwest',
'OH': 'Midwest', 'OK': 'Southwest', 'OR': 'Pacific Northwest', 'PA': 'Mid-Atlantic', 'RI': 'Northeast',
'SC': 'Southeast', 'SD': 'Midwest', 'TN': 'Southeast', 'TX': 'Southwest', 'UT': 'Mountain West',
'VT': 'Northeast', 'VA': 'Mid-Atlantic', 'WA': 'Pacific Northwest', 'WV': 'Mid-Atlantic', 'WI': 'Midwest',
'WY': 'Mountain West',
# Add any other territories or specific NOAA station locations if needed
'PR': 'Caribbean',  # Puerto Rico
'VI': 'Caribbean',  # U.S. Virgin Islands
'GU': 'Pacific Territories',  # Guam
'AS': 'Pacific Territories',  # American Samoa
'MP': 'Pacific Territories'  # Northern Mariana Islands
}

def get_stations_df():
    """
    Fetch NOAA CO-OPS station data from XML, parse into a DataFrame,
    save locally as CSV, and return the DataFrame. Includes 'Region' column.
    """
    logging.info("Attempting to get NOAA stations data.")

    # If CSV already exists and is non-empty, load it
    csv_path = Path(stations_csv)
    if csv_path.is_file() and csv_path.stat().st_size > 0:
        try:
            df = pd.read_csv(csv_path)
            # Ensure 'Region' column exists, if not, re-fetch
            if 'Region' in df.columns and not df.empty:
                logging.info(f"Loaded stations data from existing CSV: {stations_csv}")
                return df
            else:
                logging.warning(
                    f"Existing CSV '{stations_csv}' is missing 'Region' column or is empty. Fetching new data.")
        except pd.errors.EmptyDataError:
            logging.warning(f"Existing CSV '{stations_csv}' is empty or malformed. Fetching new data.")
        except Exception as e:
            logging.error(f"Error reading existing CSV '{stations_csv}': {e}. Fetching new data.")

    try:
        logging.info(f"Fetching XML data from NOAA URL: {stations_url}")
        response = requests.get(stations_url, timeout=30)
        response.raise_for_status()

        root = ET.fromstring(response.content)

        namespaces = {
            "soapenv": "http://schemas.xmlsoap.org/soap/envelope/",
            "ns": "https://opendap.co-ops.nos.noaa.gov/axis/webservices/currentpredictionstations/wsdl"
        }

        stations_list = []

        for station in root.findall(".//ns:station", namespaces):
            station_dict = {
                "station_id": station.attrib.get("ID"),
                "station_name": station.attrib.get("name"),
                "bin_number": station.findtext(".//ns:binNumber", namespaces=namespaces),
                "depth": station.findtext(".//ns:depth", namespaces=namespaces),
                "latitude": station.findtext(".//ns:lat", namespaces=namespaces),
                "longitude": station.findtext(".//ns:long", namespaces=namespaces),
                "station_type": station.findtext(".//ns:stationType", namespaces=namespaces),
                "state_abbr": station.findtext(".//ns:state", namespaces=namespaces)  # Capture state abbreviation
            }
            stations_list.append(station_dict)

        stations_df = pd.DataFrame(stations_list)

        stations_df["latitude"] = pd.to_numeric(stations_df["latitude"], errors="coerce")
        stations_df["longitude"] = pd.to_numeric(stations_df["longitude"], errors="coerce")
        stations  # Simple heuristic: look                                                        "Hawaii" if "HI_df["bin_number"] = pd.to_numeric(stations_df["bin_ for common 2-letter state codes in parentheses or at end
        import re
        match = re.search
        " in x or "
        Honolulu
        " in x else
        "Unknown")  # Default if not found

        # Map thenumber"], errors="coerce")
        stations_df["depth"] = pd.to_numeric(stations_df["depth"], errors="coerce")

        (r'\(([A-Z]{ '
        state
        ' to a '
        region
        '


# Drop rows where essential coordinates are missing
stations_df.dropna2})\)', station_name)
if match:
    return match
df['region'] = df['state'].map((subset=['latitude', 'longitude', 'state_abbr'], inplace=True.group(1)
# Try to matchSTATE_TO_REGION).fillna('Other)

# --- Add Region Column ---
# Map state abbreviations to regions. Use common state names (e.g., 'Florida', 'Texas')
df['display_name'] = df['station_name'] + " (" + 'Unknown'
for any unmapped states.
')
for state_name, abbrev in {df['station_id'] + ")"
stations_df['Region'] = stations_df['
'Alabama': 'AL',
return df

# --- Load stations DataFramestate_abbr'].map(STATE_ 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', ' ---
all_stations_df = cached_get_stations_dfTO_REGION).fillna('Unknown')

# Save to CSV
stations_df.toCalifornia
': '
CA
',
'Colorado': 'CO()

if all_stations_df.empty:
    st.warning_csv(csv_path, index=False)
logging.info(f"Successfully fetched new data, added regions, and saved to CSV: {stations_csv}")

', '
Connecticut
': '
CT
', '
Delaware
': '
DE
', '
Florida
': '
FL
', '
Georgia
': '
GA
',
("⚠ NOAA stations data could not be loaded. Please try again later.")
else:
# --- User Input Sidebar ---
return stations_df

except requests.exceptions.
'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indianast.sidebar.header("Station Selector")

# 1. Select RegionRequestException as req_err:
logging.error(f"Network or HTTP error fetching NOAA stations: {': 'IN', 'Iowa': 'IA',
'Kansas':
regions_list = sorted(all_stations_df['regionreq_err}")
return pd.DataFrame()
except ET.ParseError as parse
'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': '].unique())
selected_region = st.sidebar.selectbox("1. Select a Region", regions_list_err:
logging.error(f"XML parsing error from NOAA response: {parse_err}")
return pd.DataFrame()
'MD',
'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota':)

# Filter states based on selected region
statesexcept
Exception as e:
logging.error(f"An unexpected error occurred while fetching/processing NOAA stations: 'MN', 'Mississippi': 'MS',
'Missouri': 'MO_in_region = REGION_TO_ {e}", exc_info=True)
return pd.DataFrame()
', '
Montana
': '
MT
', '
STATES.get(selected_region,