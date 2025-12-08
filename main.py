# main.py

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import logging
from datetime import datetime
import os
import glob

# --- Project modules ---
from stations_df_func import get_stations_df
from model_trainer import train_model
from predictor import run_live_prediction
from data_loader import load_noaa_historical

# --------------------------
# Setup
# --------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
st.set_page_config(page_title="NOAA Coastal Flood Viewer", page_icon="🌊", layout="wide")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

SEQ_LEN = 72  # LSTM sequence length

# --------------------------
# Initialize Session State
# --------------------------
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'current_station' not in st.session_state:
    st.session_state.current_station = None


# --------------------------
# Cache Station Metadata
# --------------------------
@st.cache_data(ttl=3600, show_spinner="Loading NOAA station list...")
def cached_get_stations_df():
    df = get_stations_df()
    if not df.empty:
        df["station_id"] = df["station_id"].astype(str)
        df["display_name"] = df["station_name"] + " (" + df["station_id"] + ")"
    return df


# --------------------------
# Page Title & Initial Load
# --------------------------
st.title("🌊 Geospatial Watches & Warnings Web-App")
st.markdown("NOAA historical water-level data, train an LSTM model, & run a live water-level forecast for flood risk.")

stations_df = cached_get_stations_df()

if stations_df.empty:
    st.error("❌ Could not load NOAA station metadata.")
    st.stop()

# --------------------------
# Sidebar Station Selection
# --------------------------
st.sidebar.header("Station Selection")
states = sorted(stations_df["state"].dropna().unique())
selected_state = st.sidebar.selectbox("Select a State", states)

stations_in_state = stations_df[stations_df["state"] == selected_state]
station_name = st.sidebar.selectbox("Select a Station", stations_in_state["display_name"])

selected_station = stations_in_state[stations_in_state["display_name"] == station_name].iloc[0]
station_id = str(selected_station["station_id"])
lat, lon = selected_station["latitude"], selected_station["longitude"]

# Check if station changed (reset data_loaded flag)
if st.session_state.current_station != station_id:
    st.session_state.current_station = station_id
    st.session_state.data_loaded = False

# --------------------------
# Sidebar Contact Information (at bottom)
# --------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 👤 Contact")
st.sidebar.info("""
    **Eli Policape**

    🔗 [GitHub](https://github.com/theeliad)  
    💼 [LinkedIn](https://www.linkedin.com/in/eli-p-96312163/)
""")

# --------------------------
# Station Header
# --------------------------
st.header(f"Station: {selected_station['station_name']} ({station_id})")

# --------------------------
# Station Info + Map
# --------------------------
col1, col2 = st.columns([1, 1.4])

with col1:
    st.markdown("### Station Information")
    st.dataframe(pd.DataFrame({
        "Property": ["Station Name", "Station ID", "State", "Type"],
        "Value": [
            selected_station["station_name"],
            station_id,
            selected_station["state"],
            selected_station["station_type"]
        ],
    }).set_index("Property"))

with col2:
    st.markdown("### Station Map")
    if pd.notna(lat) and pd.notna(lon):
        m = folium.Map(location=[lat, lon], zoom_start=11)
        folium.Marker([lat, lon], tooltip=selected_station["station_name"]).add_to(m)
        st_folium(m, height=260, use_container_width=True)
    else:
        st.info("No map coordinates available.")

# =====================================================================
# 📘 STEP 1: LOAD HISTORICAL DATA
# =====================================================================
st.markdown("---")
st.subheader("📘 Step 1: Load Historical NOAA Data")

st.info("Loads **1 year of historical water-level data** from NOAA CO-OPS.", icon="ℹ️")

if st.button("📥 Load 1 Year of Historical Data", key="load_data_button"):

    with st.spinner("Fetching NOAA water-level data..."):
        df_hist = load_noaa_historical(station_id, years=1)

    if df_hist is None or df_hist.empty:
        st.error("❌ No historical NOAA data retrieved.")
        st.session_state.data_loaded = False
    else:
        st.session_state.data_loaded = True
        st.success(f"✔ Loaded {len(df_hist)} water-level records.")

        # ✅ SOLUTION 1: Use Expander for Data Display
        with st.expander("📊 View Data Sample (Most Recent Datasets)", expanded=False):
            st.dataframe(df_hist.tail(20))

        # ✅ SOLUTION 1: Downsample Chart Data
        with st.expander("📈 View Water Level Chart (Daily Average)", expanded=False):
            df_chart = df_hist.copy()
            df_chart['date_time'] = pd.to_datetime(df_chart['date_time'])
            df_chart = df_chart.set_index("date_time")
            # Show every 24th point (daily instead of hourly)
            st.line_chart(df_chart["water_level"].iloc[::24])

# ✅ SOLUTION 2: Display Data Load Status
if st.session_state.data_loaded:
    st.success(f"✅ Historical data loaded for station {station_id}")
else:
    st.warning("⚠️ No data loaded yet. Please load historical data first.")

# =====================================================================
# 🧠 STEP 2: TRAIN LSTM MODEL
# =====================================================================
st.markdown("---")
st.subheader("🧠 Step 2: Train Water Level Prediction Model (LSTM)")

if st.button("🧠 Train LSTM Model", key="train_model_button"):

    if not st.session_state.data_loaded:
        st.error("❌ Please load historical data first (Step 1).")
    else:
        progress_bar = st.progress(0.0, text="Initializing...")

        # ✅ SOLUTION 4: Delete Old Model Files
        old_models = glob.glob(f"models/{station_id}_waterlevel_lstm.*")
        for old_file in old_models:
            try:
                os.remove(old_file)
                logging.info(f"🗑️ Deleted old model: {old_file}")
            except Exception as e:
                logging.warning(f"Could not delete {old_file}: {e}")

        try:
            # ✅ SOLUTION 3: Re-enable Progress Bar
            model_path = train_model(
                station_id=station_id,
                years=1,
                seq_len=SEQ_LEN,
                progress_bar=progress_bar
            )

            st.success(f"✔ LSTM Model trained and saved to `{model_path}`")

        except FileNotFoundError as e:
            st.error(f"❌ Cannot train model — load historical data first. Details: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected training error: {e}")

# =====================================================================
# 🌊 STEP 3: LIVE WATER LEVEL FORECAST
# =====================================================================
st.markdown("---")
st.subheader("🌊 Step 3: Live Water Level Forecast")

st.info(f"Predicts the next hour's water level using the last {SEQ_LEN} hours of NOAA observations.",
        icon="ℹ️")

if st.button("🔮 Forecast Next Hour's Water Level", key="predict_button"):

    try:
        with st.spinner("Running live forecast..."):
            result = run_live_prediction(station_id, seq_len=SEQ_LEN)

        st.success("✔ Forecast complete!")

        st.write(f"**Prediction for: {result['prediction_for_timestamp']}**")

        col1, col2 = st.columns(2)
        col1.metric("Predicted Water Level", f"{result['predicted_water_level_ft']:.2f} ft")
        col2.metric("Predicted Risk Level", result['predicted_risk_level'])

        # Display base prediction and precipitation impact
        col3, col4 = st.columns(2)
        col3.metric("Base LSTM Prediction", f"{result['base_lstm_prediction_ft']:.2f} ft")
        col4.metric("Precipitation Impact", f"+{result['precipitation_impact_ft']:.2f} ft")

        st.write("---")

        # Display threshold information
        with st.expander("📊 Flood Threshold Information", expanded=False):
            threshold_info = result.get('threshold_info', {})
            method = threshold_info.get('method', 'unknown')

            st.write(f"**Threshold Method:** `{method}`")

            if method == 'noaa_metadata':
                st.info("Using official NOAA flood thresholds for this station.")
            elif method == 'statistical':
                st.info("Using statistically calculated thresholds from historical data.")
                st.write(f"**Data Points Used:** {threshold_info.get('data_points', 'N/A')}")
                st.write(f"**Historical Mean:** {threshold_info.get('mean', 'N/A'):.2f} ft")
                st.write(f"**Historical Std Dev:** {threshold_info.get('std', 'N/A'):.2f} ft")

            st.write("**Thresholds:**")
            st.write(f"- Minor Flood: {threshold_info.get('minor_threshold', 'N/A')} ft")
            st.write(f"- Moderate Flood: {threshold_info.get('moderate_threshold', 'N/A')} ft")
            st.write(f"- Major Flood: {threshold_info.get('major_threshold', 'N/A')} ft")

        # Display precipitation summary
        with st.expander("🌧️ Precipitation Forecast Details", expanded=False):
            st.json(result['precipitation_summary'])

        # Display most recent NOAA features used
        with st.expander("📈 Most Recent NOAA Features Used", expanded=False):
            st.json(result['features_used_for_last_step'])

    except FileNotFoundError:
        st.error("❌ No model found — train one first (Step 2).")
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
# =====================================================================
# FOOTER
# =====================================================================
st.markdown("---")
st.markdown("""
    ### 📚 About This Application

    This application provides 24-hour coastal flood forecasting using:
    - **NOAA CO-OPS** historical water level data
    - **LSTM Neural Network** for time-series prediction
    - **NWS API** for precipitation forecasts
    - **Station-specific flood thresholds** (NOAA metadata or statistical)

    **Forecast Intervals:** Predictions are made at 1-hour intervals

    **Data Sources:**
    - Water Level: NOAA Center for Operational Oceanographic Products and Services
    - Precipitation: National Weather Service API
    - Station Metadata: NOAA CO-OPS Station Registry

    **Disclaimer:** This is a demonstration tool for educational purposes. 
    For official flood warnings and forecasts, please consult NOAA and NWS official sources.
    """)

st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666;">Built with Streamlit | Powered by NOAA & NWS Data</div>',
    unsafe_allow_html=True
)
