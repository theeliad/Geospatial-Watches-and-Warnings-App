# About Me: Geospatial Flood Watches and Weather Web Application
 This is a Climatebase Capstone Project

This project presents a comprehensive web application for predicting coastal flood risk using machine learning models and NOAA data. The application is built using Python tools such as Streamlit, LSTM model, NOAA API, NWS API, Folium, and Matplotlib.

# Project Overview

The project involves developing a three-step workflow:

1.  **Loading Historical Data**: Fetching historical data from NOAA using the NOAA API.
2.  **Training an LSTM Model**: Training an LSTM model on the historical data to predict future flood risk.
3.  **Making Live Predictions**: Making live predictions using recent NOAA data and NWS precipitation forecasts.

# Technical Details

*   Python Tools: Streamlit, LSTM model, NOAA API, NWS API, Folium, and Matplotlib.
*   Machine Learning Model: LSTM model trained on historical data to predict future flood risk.
*   Data Sources: NOAA API and NWS API for fetching historical and live data.

Geospatial Watches and Weather Web Application Overview
===========================================================

## 🗂️ **Module Architecture**

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| **`main.py`** | Streamlit UI, user flow orchestration | Station selection, 3-step workflow (Load → Train → Predict) |
| **`stations_df_func.py`** | Station metadata loader | `get_stations_df()` → Returns DataFrame with station_id, name, lat, lon, state |
| **`data_fetcher.py`** | NOAA water-level fetcher + flood classification | `fetch_noaa_water()`, `get_flood_levels_from_noaa()`, `classify_flood_risk()` |
| **`data_loader.py`** | Historical data loader & cache manager | `load_noaa_historical()`, `load_last_hours_from_cache()` |
| **`model_trainer.py`** | LSTM model training (univariate: water level only) | `train_model()`, `create_lstm_sequences()`, `StreamlitCallback()` |
| **`predictor.py`** | Live prediction orchestrator | `run_live_prediction()`, `_calculate_precipitation_impact()` |
| **`nws_fetcher.py`** | NWS precipitation forecast retriever | `get_precipitation_forecast(lat, lon, hours)` |

______________________________________________________________________________

# Purpose
main.py
UI, user flow, orchestration of all steps

data_fetcher.py
Retrieve NOAA water-level data + flood thresholds + risk classification

data_loader.py
Historical & live data loading, simulation fallback

model_trainer.py
LSTM model definition, training, progress callbacks

predictor.py (not shown above but used by main.py)
Loads trained model & runs live prediction

nws_fetcher.py
Fetches NWS precipitation forecast & calculates impacts

stations_df_func.py
Loads station metadata used in dropdowns & maps

______________________________________________________________________________

# Three-Step Loop

The application follows a clean three-step loop:

1.  Load Historical Data
Uses load_noaa_historical() to fetch historical data from NOAA.
Saves the data to `data/historical/<station>_historical.csv`.

2.  Train Machine Learning Model
Utilizes model_trainer.train_model() to train the machine learning model.
Loads the historical data from Step 1.
Saves the trained model to models/<station>_flood_model.pkl.

3.  Live Prediction
    *   Uses predictor.run_live_prediction() to make live predictions.
    *   Loads the trained model from Step 2.
    *   Loads live single-timestamp placeholder observation.
    *   Predicts flood probability.

# 🌊 Comprehensive Web Application Architecture & Workflow
---

## 🔄 Complete User Workflow

### Step 1: Station Selection
```
User Action:
├─ Select State (dropdown)
├─ Select Station (dropdown)
└─ View station info + map

Backend:
├─ stations_df_func.get_stations_df()
│   └─ Returns: station_id, station_name, state, latitude, longitude, station_type
└─ Display station metadata + Folium map
```

---

### Step 2: Load Historical Data
```
User Action:
└─ Click "📥 Load 1 Year of Historical Data"

Backend Flow:
main.py
  └─ Calls: data_loader.load_noaa_historical(station_id, years=1)
       │
       └─ Calls: data_fetcher.fetch_noaa_water(station_id, start_date, end_date)
            │
            ├─ Uses: noaa_coops.Station(station_id).get_data(product="water_level")
            │
            ├─ Handles column name variations:
            │   • If columns = ["t", "v"] → rename to ["date_time", "water_level"]
            │   • If columns = ["date_time", "water_level"] → keep as-is
            │   • If columns = ["time", "value"] → rename to ["date_time", "water_level"]
            │
            └─ Returns: DataFrame with ["date_time", "water_level"]

       └─ Saves to: data/historical/{station_id}_historical_1y.csv

UI Output:
├─ Success message: "✔ Loaded {N} water-level records"
├─ Display: df.head() table
└─ Display: Line chart of water_level over time
```

File Structure After Step 2:
```
data/
└── historical/
    └── 8724580_historical_1y.csv  # Example for Key West station
```

---

### Step 3: Train LSTM Model
```
User Action:
└─ Click "🧠 Train LSTM Model"

Backend Flow:
main.py
  └─ Calls: model_trainer.train_model(station_id, years=1, seq_len=72, epochs=20)
       │
       ├─ Loads: data/historical/{station_id}_historical_1y.csv
       │
       ├─ Validates: "water_level" column exists
       │
       ├─ Preprocessing:
       │   ├─ Scale data: MinMaxScaler(feature_range=(0, 1))
       │   ├─ Save scaler: models/{station_id}_waterlevel_scaler.pkl
       │   └─ Create sequences: create_lstm_sequences(data, seq_len=72)
       │       • X shape: (num_samples, 72, 1)  # 72 hours of water level
       │       • y shape: (num_samples, 1)      # Next hour's water level
       │
       ├─ Train/Test Split: 80% train, 20% test
       │
       ├─ LSTM Architecture:
       │   └─ LSTM(50, return_sequences=True) → Dropout(0.2)
       │       └─ LSTM(50, return_sequences=False) → Dropout(0.2)
       │           └─ Dense(25, relu) → Dense(1)
       │
       ├─ Training:
       │   ├─ Optimizer: Adam
       │   ├─ Loss: Mean Squared Error
       │   ├─ Callbacks: EarlyStopping(patience=5), ModelCheckpoint, StreamlitCallback
       │   └─ Progress bar updates in Streamlit UI
       │
       └─ Saves:
           ├─ models/{station_id}_waterlevel_lstm.keras  (or .h5)
           └─ models/{station_id}_waterlevel_scaler.pkl

UI Output:
├─ Progress bar: "Training Epoch 15/20 – Validation Loss: 0.0023"
└─ Success message: "✔ LSTM Model trained and saved to models/{station_id}_waterlevel_lstm.keras"
```

File Structure After Step 3:
```
models/
├── 8724580_waterlevel_lstm.keras
└── 8724580_waterlevel_scaler.pkl
```

---

### Step 4: Live Water Level Forecast
```
User Action:
└─ Click "🔮 Forecast Next Hour's Water Level"

Backend Flow:
main.py
  └─ Calls: predictor.run_live_prediction(station_id, seq_len=72)
       │
       ├─ PHASE 1: Load Model & Scaler
       │   ├─ Load: models/{station_id}_waterlevel_lstm.keras
       │   └─ Load: models/{station_id}_waterlevel_scaler.pkl
       │
       ├─ PHASE 2: Get Recent Water Level Data
       │   └─ Calls: data_loader.load_last_hours_from_cache(station_id, hours=72)
       │        │
       │        ├─ Searches for:
       │        │   • data/historical/{station_id}_historical_1y.csv
       │        │   • data/historical/{station_id}_historical_5y.csv
       │        │   • data/historical/{station_id}_historical.csv
       │        │
       │        ├─ Filters: Last 72 hours from current UTC time
       │        │
       │        └─ Returns: DataFrame with ["date_time", "water_level"]
       │
       ├─ PHASE 3: Base LSTM Prediction
       │   ├─ Prepare input: Scale last 72 hours of water_level
       │   ├─ Reshape: (1, 72, 1)
       │   ├─ Predict: model.predict(X_pred)
       │   ├─ Inverse transform: scaler.inverse_transform()
       │   └─ Result: base_lstm_prediction_ft (e.g., 2.45 ft)
       │
       ├─ PHASE 4: Fetch Precipitation Forecast
       │   ├─ Get station coordinates:
       │   │   └─ Calls: stations_df_func.get_stations_df()
       │   │        └─ Extract: lat, lon for station_id
       │   │
       │   └─ Calls: nws_fetcher.get_precipitation_forecast(lat, lon, hours=12)
       │        │
       │        ├─ NWS API: /points/{lat},{lon}
       │        │   └─ Get: forecastGridData URL
       │        │
       │        ├─ Fetch: quantitativePrecipitation (hourly)
       │        │
       │        └─ Returns: DataFrame with:
       │            • valid_time (datetime)
       │            • precipitation_probability (%)
       │            • precipitation_amount (inches, if available)
       │
       ├─ PHASE 5: Calculate Precipitation Impact
       │   └─ Calls: _calculate_precipitation_impact(precip_df)
       │        │
       │        ├─ Heuristic Rules:
       │        │   • max_prob >= 80% → +2.0 ft
       │        │   • max_prob >= 60% → +1.0 ft
       │        │   • max_prob >= 30% → +0.5 ft
       │        │   • else → 0.0 ft
       │        │
       │        └─ Returns: {
       │              '


# Machine Learning Model:

LSTM Machine Learning (Official)

The official version uses an LSTM (Long Short-Term Memory) machine learning model to predict the water level at a given time. The model is trained on historical data such as water level, precipitation, temperature, and wind speed. The model is designed to answer "What will the water level be at 3:00 PM?" and is well-suited for this task due to its ability to capture long-term dependencies in the data.

The LSTM machine learning model used in the official version is a type of recurrent neural network (RNN) that is particularly well-suited for time series forecasting tasks. The model is trained on a sequence of data points and makes predictions based on the patterns and relationships it has learned from the data.

The LSTM model is particularly well-suited for this task because it can capture long-term dependencies in the data and is robust to noise and variability. The model is also relatively fast to train and can handle large datasets.

Model Trainer (LSTM)

The model_trainer.py module contains the train_model() function, which trains an LSTM model to predict the water level at a given time. The function takes in the station ID, sequence length, epochs, batch size, and progress bar as inputs. It loads the historical data, scales the data, creates sequences, trains the model, and saves the trained model and scaler to disk.

Predictor (LSTM)

The predictor.py module contains the run_live_prediction() function, which takes in the station ID, sequence length, and live data as inputs. It loads the trained model, scales the live data, makes predictions, and returns the predicted water level and flood risk.

Accuracy Comparison

For predicting "What will the water level be at 3:00 PM?", the LSTM model will be more accurate due to its ability to capture long-term dependencies in the data.

NOAA Live Data Fetcher and Loader 

The data_fetcher.py module contains the load_noaa_live() function, which fetches the latest NOAA live measurements for a given station. The function takes in the station ID and sequence length as inputs and returns the live data.

The data_loader.py module contains the load_noaa_historical() function, which fetches historical data from NOAA for a given station. The function takes in the station ID and lookback days as inputs and returns the historical data.

Plotting Function 

The plot_results() function is used to plot the actual vs. predicted water levels for visual inspection. The function takes in the actual and predicted water levels as inputs and plots them using matplotlib.

# Data Sources

The project uses two data sources:

NOAA API: For fetching historical and live data.
NWS API: For fetching precipitation forecasts.
