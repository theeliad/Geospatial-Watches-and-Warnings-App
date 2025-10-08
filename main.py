import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import the function to get the DataFrame
from stations_df_func import get_stations_df

# --- Streamlit page setup ---
st.set_page_config(
    page_title="NOAA CO-OPS Station Viewer",
    page_icon="🌊",
    layout="wide"
)


# --- Data Loading (Cached for Performance) ---
# Use st.cache_data to cache the output of get_stations_df.
# The cache will be re-run only if the function's arguments or its source code changes,
# or if the ttl (time to live) expires.
@st.cache_data(ttl=3600, show_spinner="Loading NOAA station data...")  # Cache for 1 hour (3600 seconds)
def cached_get_stations_df():
    """
    Wrapper for get_stations_df to leverage Streamlit's caching.
    """
    logging.info("Calling get_stations_df (might be from cache or fresh fetch).")
    df = get_stations_df()

    # Add a 'display_name' column for the selectbox
    if not df.empty:
        df['display_name'] = df['station_name'] + " (" + df['station_id'] + ")"
    return df


# --- Main App Logic ---
st.title("🌊 NOAA CO-OPS Station Viewer")
st.markdown("Select a station to view its details on the map.")

# Load stations DataFrame using the cached function
all_stations_df = cached_get_stations_df()

if all_stations_df.empty:
    st.warning("⚠ NOAA stations data could not be loaded. Please try again later.")
else:
    # --- Add a 'State' or 'Region' column for filtering (if not already present) ---
    # NOAA data doesn't directly provide 'State' in the XML.
    # We'll create a dummy 'Region' for demonstration or
    # you could implement a reverse geocoding lookup here if needed.
    # For now, let's create a simple region based on longitude for demonstration.
    # In a real app, you might use a pre-computed state column or a geocoding service.

    # Simple region assignment based on longitude for demonstration purposes
    def assign_region(lon):
        if lon > -70:
            return "East Coast"
        elif lon > -95:
            return "Central US"
        elif lon > -125:
            return "West Coast"
        else:
            return "Alaska/Hawaii/Other"  # This might need refinement for actual states/territories


    # Ensure 'region' column exists for dropdown. If you have actual state data, use that.
    if 'region' not in all_stations_df.columns:
        all_stations_df['region'] = all_stations_df['longitude'].apply(assign_region)
        logging.info("Assigned dummy 'region' column based on longitude.")

    # --- User Input Sidebar ---
    st.sidebar.header("Station Selector")

    # 1. Select a Region/State
    regions_list = sorted(all_stations_df['region'].unique())
    selected_region = st.sidebar.selectbox("1. Select a Region", regions_list)

    # Filter stations based on the selected region
    stations_in_region = all_stations_df[all_stations_df['region'] == selected_region].copy()

    # 2. Select a Station within the chosen region
    selected_station_display_name = st.sidebar.selectbox(
        "2. Select a Station",
        options=stations_in_region['display_name']
    )

    # --- Author and Contact Info Sidebar ---
    st.sidebar.markdown("---")
    st.sidebar.title("About")
    st.sidebar.info("""
        This app displays NOAA CO-OPS station locations and details.
    """)
    st.sidebar.title("Contact")
    st.sidebar.info("""
        [Your GitHub](https://github.com/yourusername) | [Your LinkedIn](https://www.linkedin.com/in/yourusername/)
    """)

    # --- Display selected station data and map ---
    if selected_station_display_name:
        selected_station_data = stations_in_region[
            stations_in_region['display_name'] == selected_station_display_name
            ].iloc[0]

        lat = selected_station_data['latitude']
        lon = selected_station_data['longitude']

        st.header(f"Details for: {selected_station_data['station_name']}")
        st.subheader(f"({selected_station_data['station_id']}) - {selected_region}")

        # Display key station details in a table
        st.markdown("**Station Information**")
        station_info_df = pd.DataFrame({
            "Property": ["Station ID", "Station Name", "Station Type", "Bin Number", "Depth", "Latitude", "Longitude"],
            "Value": [
                selected_station_data['station_id'],
                selected_station_data['station_name'],
                selected_station_data['station_type'],
                selected_station_data['bin_number'],
                selected_station_data['depth'],
                f"{lat:.4f}",
                f"{lon:.4f}"
            ]
        })
        st.dataframe(station_info_df.set_index('Property'), use_container_width=True)

        st.markdown("**Station Location**")
        # Create a map centered on the selected station
        m = folium.Map(location=[lat, lon], zoom_start=12)
        folium.Marker(
            [lat, lon],
            popup=f"<b>{selected_station_data['station_name']}</b><br>ID: {selected_station_data['station_id']}<br>Type: {selected_station_data['station_type']}",
            tooltip=selected_station_data['station_name']
        ).add_to(m)

        # Render the map
        st_folium(m, height=450, width=725)

    else:
        st.warning(f"No stations found in the selected region: **{selected_region}**")
