# main.py

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import logging
from datetime import datetime, timedelta
import os

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
st.title("🌊 NOAA Coastal Flood Risk & LSTM Prediction Viewer")
st.markdown(
    "Select a NOAA station, load historical data, train a model to predict water levels, and run a live forecast.")

stations_df = cached_get_stations_df()

if stations_df.empty:
    st.error(
        "❌ Could not load NOAA station metadata. Please ensure 'stations_df_func.py' is in the directory and works.")
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

# --- CHANGE: Updated info message and button text to reflect 1 year ---
st.info(
    "Loads 1 year of real historical data. If no flood events are found, it uses simulated data to ensure the model can be trained.",
    icon="ℹ️")

if st.button("📥 Load 1 Year of Historical Data"):

    progress_bar = st.progress(0, text="Starting historical data load...")

    # We call the function with years=1 explicitly for clarity
    df_hist = load_noaa_historical(station_id, years=1, progress_bar=progress_bar)

    if df_hist.empty:
        st.error("❌ No historical NOAA data retrieved.")
    else:
        st.success(f"✔ Loaded {len(df_hist)} historical rows.")

        display_cols = ["date_time", "water_level", "wind_speed", "air_pressure", "air_temperature", "risk_level"]
        cols_to_show = [col for col in display_cols if col in df_hist.columns]
        df_display = df_hist[cols_to_show]

        st.dataframe(df_display.head())

        if 'date_time' in df_hist.columns:
            df_chart = df_hist.copy()
            df_chart['date_time'] = pd.to_datetime(df_chart['date_time'])
            df_chart = df_chart.set_index("date_time")
            st.line_chart(df_chart["water_level"])

# =====================================================================
# 🧠 STEP 2: TRAIN LSTM MODEL
# =====================================================================
st.markdown("---")
st.subheader("🧠 Step 2: Train Water Level Prediction Model (LSTM)")

if st.button("🧠 Train LSTM Model"):

    progress_bar = st.progress(0, text="Initializing LSTM training...")

    try:
        model_path = train_model(
            station_id,
            seq_len=SEQ_LEN,
            progress_bar=progress_bar
        )

        st.success(f"✔ LSTM Model trained and saved to `{model_path}`")

    except FileNotFoundError as e:
        st.error(f"❌ Could not train model. Please load historical data first. Details: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during LSTM training: {e}")


# =====================================================================
# 🌊 STEP 3: LIVE WATER LEVEL FORECAST
# =====================================================================
st.markdown("---")
st.subheader("🌊 Step 3: Live Water Level Forecast")
st.info(f"Uses the last {SEQ_LEN} hours of data to predict the water level for the next hour.", icon="ℹ️")

if st.button("🔮 Forecast Next Hour's Water Level"):
    try:
        with st.spinner("Running live LSTM forecast..."):
            result = run_live_prediction(station_id, seq_len=SEQ_LEN)

        st.success("✔ Forecast complete!")

        st.write(f"**Prediction for: {result['predicted_for_timestamp']}**")

        col1, col2 = st.columns(2)

        col1.metric("Predicted Water Level", f"{result['predicted_water_level_ft']:.2f} ft")
        col2.metric("Predicted Risk Level", result['predicted_risk_level'])

        st.write("Prediction based on the most recent known observation:")
        st.json(result['features_used_for_last_step'])

    except FileNotFoundError as e:
        st.error(f"❌ Model or scaler not found. Please train the model first. Details: {e}")
    except ValueError as e:
        st.error(f"❌ Data Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")
