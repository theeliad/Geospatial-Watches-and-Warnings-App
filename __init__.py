# __init__.py

"""
Geospatial Coastal Water Level Prediction Package

This package provides:

- Streamlit application entry point (`main.py`)
- NOAA + NWS station metadata loading (`stations_df_func.py`)
- NOAA historical data loader (`data_loader.py`)
- NWS live data fetcher (`nws_fetcher.py`)
- Flood classification + utilities (`data_fetcher.py`)
- Machine learning model trainer (`model_trainer.py`)
- Live forecasting pipeline (`predictor.py`)
- State lookup helpers (`state_finder.py`)

All modules are exposed through __all__ for clean imports.
"""

__all__ = [
    "main",
    "stations_df_func",
    "state_finder",
    "data_loader",
    "data_fetcher",
    "nws_fetcher",
    "model_trainer",
    "predictor",
]
