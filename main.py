import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import pandas as pd
import logging

# Set up basic logging for main.py as well
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from stations_df_func import get_stations_df

# --- Streamlit page setup ---
st.set_page_config(layout="wide")
st.title("NOAA CO-OPS Station Clusters")

# --- Cached function to load stations DataFrame ---
# Use st.cache_data to cache the output of get_stations_df.
# The cache will be re-run only if the function's arguments or its source code changes.
@st.cache_data(ttl=3600, show_spinner="Loading NOAA station data...") # Cache for 1 hour (3600 seconds)
def cached_get_stations_df():
    """
    Wrapper for get_stations_df to leverage Streamlit's caching.
    """
    logging.info("Calling get_stations_df (might be from cache or fresh fetch).")
    return get_stations_df()

# --- Load stations DataFrame ---
try:
    stations_df = cached_get_stations_df()

    if stations_df.empty:
        st.warning("⚠ NOAA stations data could not be loaded. Please try again later.")
    else:
        # --- Create Folium map ---
        # Calculate the bounds to fit all markers
        min_lat, max_lat = stations_df['latitude'].min(), stations_df['latitude'].max()
        min_lon, max_lon = stations_df['longitude'].min(), stations_df['longitude'].max()

        # Initialize map with a reasonable default center and zoom
        # Folium will adjust the zoom to fit bounds later
        m = folium.Map(
            location=[stations_df['latitude'].mean(), stations_df['longitude'].mean()],
            zoom_start=4 # Start with a wider zoom
        )

        # Fit the map to the bounds of all stations
        # This will automatically set the center and zoom to show all markers
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])


        # --- Add marker cluster ---
        marker_cluster = MarkerCluster().add_to(m)
        for idx, row in stations_df.iterrows():
            popup_text = f"<b>{row['station_name']}</b> ({row['station_id']})<br>Type: {row['station_type']}<br>Lat: {row['latitude']:.2f}, Lon: {row['longitude']:.2f}"
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=popup_text
            ).add_to(marker_cluster)

        # --- Render map in Streamlit ---
        st_data = st_folium(m, width=1200, height=800)

except Exception as e:
    st.error(f"An error occurred while loading NOAA stations: {e}")
    logging.error(f"Error in main.py: {e}", exc_info=True)