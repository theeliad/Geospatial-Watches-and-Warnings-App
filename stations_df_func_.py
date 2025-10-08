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
stations_url = "https://opendap.co-ops.nos.noaa.gov/axis/webservices/currentpredictionstations/response.jsp?format=xml"

# --- State to Region Mapping ---
# This dictionary maps each state to a broader geographical region.
STATE_TO_REGION = {
    # Northeast
    'ME': 'Northeast', 'VT': 'Northeast', 'NH': 'Northeast', 'MA': 'Northeast', 'RI': 'Northeast',
    'CT': 'Northeast', 'NY': 'Northeast', 'PA': 'Northeast', 'NJ': 'Northeast', 'DE': 'Northeast',
    'MD': 'Northeast', 'DC': 'Northeast',
    # Southeast
    'VA': 'Southeast', 'WV': 'Southeast', 'NC': 'Southeast', 'SC': 'Southeast', 'GA': 'Southeast',
    'FL': 'Southeast', 'AL': 'Southeast', 'MS': 'Southeast', 'TN': 'Southeast', 'KY': 'Southeast',
    'AR': 'Southeast', 'LA': 'Southeast',
    # Midwest
    'OH': 'Midwest', 'IN': 'Midwest', 'IL': 'Midwest', 'MI': 'Midwest', 'WI': 'Midwest',
    'MN': 'Midwest', 'IA': 'Midwest', 'MO': 'Midwest', 'ND': 'Midwest', 'SD': 'Midwest',
    'NE': 'Midwest', 'KS': 'Midwest',
    # Southwest
    'AZ': 'Southwest', 'NM': 'Southwest', 'TX': 'Southwest', 'OK': 'Southwest',
    # West
    'MT': 'West', 'ID': 'West', 'WY': 'West', 'CO': 'West', 'UT': 'West',
    'NV': 'West', 'CA': 'West', 'OR': 'WA': 'West', 'HI': 'West', 'AK': 'West',
    # For any states not explicitly listed or territories, you might want a 'Other' category
    # Note: NOAA data might include territories like PR (Puerto Rico), GU (Guam), VI (Virgin Islands)
    # You'll need to decide how to categorize these or add them to the mapping.
    # For now, I'll add common ones.
    'PR': 'Caribbean', 'VI': 'Caribbean', 'GU': 'Pacific Territories', 'AS': 'Pacific Territories'
}

# You'll notice the NOAA data doesn't directly provide a 'State' column.
# We'll need to infer the state from the station's latitude/longitude or its name.
# For simplicity and to match your example, I'll assume we can get a state abbreviation
# from the station name or you might add a reverse geocoding step if truly needed.
# However, the NOAA XML itself does not contain a state field.
# The most reliable way to get a state would be to perform a reverse geocode
# or map based on lat/long. For this example, I'll add a placeholder
# and suggest how you might get state if not directly available.

def get_state_from_lat_lon(latitude, longitude):
    """
    Placeholder for a function to get state abbreviation from lat/lon.
    This would typically involve an external geocoding service or a local shapefile.
    For demonstration, we'll return a generic state or infer from common NOAA locations.
    """
    # This is a simplification. In a real app, you'd use a library like
    # geopy with a geocoding service (e.g., Nominatim) or a local shapefile.
    # For now, we'll map based on approximate coordinates or return a default.
    # Since NOAA stations are mostly coastal, we can make some educated guesses
    # or rely on a pre-defined list if available.
    # For the purpose of this example, we will assume a way to get the state.
    # A robust solution would involve adding a 'State' column during data processing.
    # For now, let's just return a placeholder or try to infer from station name if possible.
    return "Unknown" # Placeholder. Will need actual state data.


def get_stations_df():
    """
    Fetch NOAA CO-OPS station data from XML, parse into a DataFrame,
    save locally as CSV, and return the DataFrame.
    Includes adding 'State' and 'Region' columns.
    """
    logging.info("Attempting to get NOAA stations data.")

    csv_path = Path(stations_csv)
    if csv_path.is_file() and csv_pathOkay, this is a fantastic next step for improving the user experience and performance! Shifting from an "all markers on map" approach to a dropdown-driven selection for individual stations, categorized by region and state, will make.stat().st_size > 0:
        try:
            df = pd. the app much more responsive and user-friendly.

We'll adapt your provided NWS forecast app structure to our NOAA CO-OPS stations.

Here's how we'll modify `main.py` and introduce a helper function for region mapping:

**1. Update `main.py`:**
   *   We'll remove the initial map that displays all clusters.
   *   We'll create a `region_mapping` function (or embed it) to assign each state to a broader US region.
   *   The sidebar will now have three dropdowns: Region, State, and then Station.
   *   The map will display a *single* selected station, similar to your NWS example.
   *   We'll keep the `st.cache_data` for `get_stations_df` to ensure fast loading of the full dataset.

**2. No changes needed for `stations_df_func.py`**
   *   `stations_df_func.py` will remain as is, continuing to fetch and cache the full list of stations into `all_noaa_coops_stations_xml.csv`. This is exactly what we want: a comprehensive local dataset to power our dropdowns.

---

### Step-by-Step Implementation

First, let's define the regions and states. I'll use a common regional breakdown for the US.

```python
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
for state, region in STATE_TO_REGION.items():
    REGION_TO_STATES.setdefault(region, []).append(state)
for region in REGION_TO_STATES:
    REGION_TO_STATES[region].sort() # Sort states within each region