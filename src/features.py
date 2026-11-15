# features.py
import pandas as pd
import numpy as np


def create_temporal_features(df):

    df = df.copy()

    # ensure valid timestamp
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["date"]).copy()

    # create daily timeline
    daily_df = (
        df.groupby([
            "sensor_name",
            pd.Grouper(
                key="date",
                freq="D"
            )
        ])
        .agg({
            "latitude": "first",
            "longitude": "first",
            "pm25": "median",
            "temperature_celsius": "median",
            "pressure_mb": "median",
            "wind_u": "median",
            "wind_v": "median",
            "NO2": "median",
            "CO": "median",
            "O3": "median",
            "AOD": "median"
        }).reset_index())

    daily_df = daily_df.sort_values(["sensor_name", "date"]).reset_index(drop=True)

    # gap between previous observation
    previous_date = (daily_df.groupby("sensor_name")["date"].shift(1))

    day_gap = (daily_df["date"] - previous_date).dt.days

    # lag features
    for col in ["AOD", "NO2"]:

        daily_df[f"{col}_lag_1"] = (daily_df.groupby("sensor_name")[col].shift(1))

        # Do not calculate lag across multi-day gaps
        daily_df.loc[day_gap != 1, f"{col}_lag_1"] = np.nan

    # Day-to-day change in AOD
    daily_df["AOD_difference_1"] = (daily_df.groupby("sensor_name")["AOD"].diff())

    daily_df.loc[day_gap != 1, "AOD_difference_1"] = np.nan

    # Past-only 3 day rolling mean
    daily_df["AOD_rolling_mean_3"] = (
        daily_df.groupby("sensor_name")["AOD"]
        .transform(lambda x: x.shift(1).rolling(window=3, min_periods=3).mean()))

    # Past-only 7 day rolling mean
    daily_df["AOD_rolling_mean_7"] = (
        daily_df.groupby("sensor_name")["AOD"]
        .transform(lambda x: x.shift(1).rolling(window=7, min_periods=7).mean()))

    # Cyclical calendar features
    daily_df["day_of_year"] = (daily_df["date"].dt.dayofyear)

    daily_df["day_of_year_sin"] = np.sin(2 * np.pi * daily_df["day_of_year"] / 365)

    daily_df["day_of_year_cos"] = np.cos(2 * np.pi * daily_df["day_of_year"] / 365)

    print("\nTemporal features created successfully")

    return daily_df