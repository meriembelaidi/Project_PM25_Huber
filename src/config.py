# config.py

# Reproducibility
SEED = 42

# Project paths
file_names = ["combined_gee_pm25.csv", "gee_features_daily.csv", "pm25.csv"]


# target
target = "pm25"

# Predictor variables
predictors = [
    "temperature_celsius",
    "pressure_mb",
    "wind_u",
    "wind_v",
    "NO2",
    "CO",
    "O3",
    "AOD"
]

# Variables used for EDA visualizations
variables = [
    "temperature_celsius",
    "pressure_mb",
    "wind_u",
    "wind_v",
    "NO2",
    "CO",
    "O3",
    "AOD",
    "pm25"
]

# Numerical columns
numerical_cols = [
    "latitude",
    "longitude",
    "temperature_celsius",
    "pressure_mb",
    "wind_u",
    "wind_v",
    "NO2",
    "CO",
    "O3",
    "AOD",
    "pm25"
]

# Variables for missingness indicators
sentinel_col = ["NO2", "CO", "O3"]