import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import linregress, ks_2samp
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import pearsonr, spearmanr
import json

file_names = ["combined_gee_pm25.csv", "gee_features_daily.csv", "pm25.csv"]

for file in file_names:
    print(f"\n{'='*25}")
    print(f"FILE: {file}")
    print(f"{'='*25}")
    
    
    df = pd.read_csv(file)
    
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nThe first 5 rows:\n", df.head())
    print("\nData type:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDuplicates rows:\n", df.duplicated().sum())


df = pd.read_csv("combined_gee_pm25.csv")

# target and predictors and variables
target = "pm25"

variables = [
    "temperature_celsius",
    "pressure_mb",
    "wind_u",
    "wind_v",
    "NO2",
    "CO",
    "O3",
    "AOD",
    "pm25"]

numerical_cols =[
    'latitude',
    'longitude',
    'temperature_celsius',
    'pressure_mb',
    'wind_u',
    'wind_v', 
    'NO2', 
    'CO', 
    'O3', 
    'AOD', 
    'pm25']


predictors = [
    "temperature_celsius",
    "pressure_mb",
    "wind_u",
    "wind_v",
    "NO2",
    "CO",
    "O3",
    "AOD"]

# display basic info about dataset

print("\nMain dataset info:")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Target:", target)
print("Date range:", df["date"].min(), "to", df["date"].max())

print("\nLatitude range:")
print(df['latitude'].min(), df['latitude'].max())

print("\nLongitude range:")
print(df['longitude'].min(), df['longitude'].max())

print("\nDataset information:", df.info())

print("\nDescripe:\n", df[target].describe().T)

# calculate summary statistics and skewness, kurtosis
summary = df[numerical_cols].describe().T

summary['skewness'] = df[numerical_cols].skew()
summary['kurtosis'] = df[numerical_cols].kurt()

print("\nDescribe \n:", summary.round(2))

print("\nObservation per sensor:", df['sensor_name'].value_counts())

print("\nNumber of unique sensors:", df['sensor_name'].nunique())

# visualization distribution od the main variables

# visualize the distribution using histogram
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(variables):
    sns.histplot(data=df, x=col, bins=40, kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
    
plt.tight_layout()
plt.show()

# visualize using boxplot
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(variables):
    sns.boxenplot(data=df, y=col, ax=axes[i])
    axes[i].set_title(f'Boxplot of {col}')
    axes[i].set_ylabel(col)
    
plt.tight_layout()
plt.show()

# visualize number of observations using count plot
top_sensors = df['sensor_name'].value_counts().head(10).index

plt.figure(figsize=(12, 6))

sns.countplot(
    data=df[df['sensor_name'].isin(top_sensors)],
    y='sensor_name',
    order=top_sensors)

plt.title("Top 10 sensors by number of observations")
plt.xlabel('Number of observations')
plt.ylabel('sensor')
plt.tight_layout()
plt.show()

# visualize the distribution using boxplot
plt.figure(figsize=(14, 7))
sns.boxplot(
    data=df[df['sensor_name'].isin(top_sensors)],
    x='sensor_name',
    y=target,
    order=top_sensors)

plt.title("PM2.5 Distributions")
plt.xlabel('sensor')
plt.ylabel('PM2.5')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

# check missing values
print("\nMissing count:\n", df.isnull().sum())
print("\nMissing percentage:\n", (df.isnull().mean() * 100).round(2))

# check for invalid or unsual values
print("\nDomain knowledge check:")
print("Negative PM2.5 values:", (df[target] < 0).sum())
print("PM2.5 values equal to -999:", (df[target] == -999).sum())
print("PM2.5 values equal to 1000:", (df[target] == 1000).sum())

print("Negative AOD values:", (df['AOD'] < 0).sum())
print("Negative pressure values:", (df['pressure_mb'] < 0).sum())

# detect outliers by using IQR 
print("\nIQR:")

iqr_results= []
for col in variables:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    
    iqr_results.append({
        'variable': col,
        'Lower bound': lower,
        "Upper bound": upper,
        'Outliers': outliers,
        'Percentage': outliers / len(df) * 100})
    
iqr_table = pd.DataFrame(iqr_results)
print(iqr_table.round(2).to_string(index=False))

# detect outliers by using Z-score
print("\nZ-SCORES:")

zscore = (df[variables] - df[variables].mean()) / df[variables].std()

zscore_results = pd.DataFrame({
    'Variable': variables,
    'Outliers': (zscore.abs() > 3).sum().values})

zscore_results['Percentage'] = (
    zscore_results['Outliers'] / len(df) * 100)

print(zscore_results.round(2).to_string(index=False))

# detect outliers by using Isolation forest
print("\nIsolation forest:")

X_if = df[predictors].copy()
imputer = SimpleImputer(strategy='median')
X_if_imputed = imputer.fit_transform(X_if)

if_scaler = StandardScaler()
X_if_scaled = if_scaler.fit_transform(X_if_imputed)

iso_forest = IsolationForest(
    n_estimators=100,
    contamination='auto',
    random_state=42,
    n_jobs=-1)

if_predictions = iso_forest.fit_predict(X_if_scaled)

n_out_if = (if_predictions == -1).sum()
pct_out_if = n_out_if / len(df) * 100

print("Potentials outliers:", n_out_if)
print("Normal obdervations:", (if_predictions == 1).sum())
print("Outliers perecentage:", round(pct_out_if, 2), "%")


df['isolation_outlier'] = if_predictions

print("\nFirst 10 potential outliers:")
print(df[df['isolation_outlier'] == -1][
    ['date', 'sensor_name', 'pm25', 'AOD', 'NO2', 'CO', 'O3']].head(10))



# DATA CLEANING
#remove missing and invalid values in PM2.5
initial_rows = len(df)

missing_pm25 = df[target].isna().sum()
sentinel_low = (df[target] <= 0).sum()
sentinel_high = (df[target] >= 500).sum()

invalid_pm25 = ((df[target] <= 0) | (df[target] >= 500))

invalid_pm25_count = invalid_pm25.sum()
df = df.dropna(subset=[target]).copy()

df = df.loc[~invalid_pm25].copy()

final_rows = len(df)
removed_rows = initial_rows - final_rows

print("\nCleaning results of PM2.5:")
print("Intial observations:", initial_rows)
print("Missing values:", missing_pm25)
print("Invalid values:", invalid_pm25_count)
print("Total observation removed:", removed_rows)
print("Remaining observations:", final_rows)
print("Percentage removed:", round(removed_rows / initial_rows * 100, 2), '%')

print("\nVerification:")
print("Missing PM2.5:", df[target].isna().sum())
print("Minimum PM2.5:", df[target].min())
print("Maximum PM2.5:", df[target].max())

# check for duplicates observations and key fields
print("\nDuplicates :")
print("Duplicates all columns:", df.duplicated().sum())

key_fields = [
    'date',
    'latitude',
    'longitude',
    'sensor_name']

n_duplicates_key = df.duplicated(subset=key_fields).sum()

print("Duplicates key fields:", n_duplicates_key)

print("\nMissing key fields:", df[key_fields].isna().sum())


# calculate Pearson and Spearman correlations
X= df[predictors]
pearson_corr = X.corr(method='pearson')
spearman_corr = X.corr(method='spearman')

print("Pearson correlation :")
print(df[predictors + [target]].corr()[target].sort_values(ascending=False))

print("\nSpearman correlation :")
print(df[predictors + [target]].corr(method='spearman')[target].sort_values(ascending=False))

# identify highly correlated predictors
upper_triangle = pd.DataFrame(
    np.triu(np.ones(pearson_corr.shape), k=1).astype(bool),
    index=pearson_corr.index,
    columns=pearson_corr.columns)

high_corr = pearson_corr.where(upper_triangle).stack()

print("\nHighly correlated predictors:")
print(high_corr[high_corr.abs() > 0.9])

# visualize correlations with heatmaps
plt.figure(figsize=(12, 8))
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', center=0)
plt.title("Pearson correlation heatmap")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', center=0)
plt.title("Spearman correlation heatmap")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, col in enumerate(predictors):
    sns.scatterplot(data=df, x=col, y=target, alpha=0.3, ax=axes[i])
    axes[i].set_title(f'PM2.5 vs {col}')

plt.tight_layout()
plt.show()

# Seasonal analysis
df['date'] = pd.to_datetime(df['date'], errors="coerce")
df['month'] = df['date'].dt.month

# create seasons
seasons = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}

df['seasons'] = df['month'].map(seasons)

# analysis PM2.5 variation
# visualize distribution by season using boxplot
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x='seasons', y=target, order=['Winter', 'Spring', 'Summer', 'Autumn'])
plt.title('PM2.5 distribution by season')
plt.show()    

# visualize PM2.5 target distribution using histogram
print("PM2.5 skewness:", round(df[target].skew(), 2))
sns.histplot(data=df, x=target, bins=40, kde=True)
plt.title('PM2.5 target distribution')
plt.xlabel('PM2.5 (µg/m³)')
plt.show()

print("\nAverage PM2.5 by seasons:")
print(df.groupby('seasons')[target].mean().round(2))

sns.boxplot(data=df, x='seasons', y=target, order=['Winter', 'Spring', 'Summer', 'Autumn'])
plt.title('PM2.5 seasonal variation')
plt.show()

# check pollution threshold Proportions
prop_25 = (df[target] > 25).mean() * 100
prop_50 = (df[target] > 50).mean() * 100

print("\nPM2.5 pollution thresholds:")
print("Proportion of PM2.5 > 25 :", round(prop_25, 2), '%')
print("Proportion of PM2.5 > 50:", round(prop_50, 2), '%')

# Exploratory_Data.csv file to create

X_lof = df[predictors].copy()

lof_imputer = SimpleImputer(strategy='median')
X_lof_imputed = lof_imputer.fit_transform(X_lof)

lof_scaler = StandardScaler()
X_lof_scaled = lof_scaler.fit_transform(X_lof_imputed)

lof = LocalOutlierFactor(n_neighbors=20, contamination='auto', n_jobs=-1)

# detect outliers
lof_predictions = lof.fit_predict(X_lof_scaled)

# count outliers
n_out_lof = (lof_predictions == -1).sum()
pct_out_lof = n_out_lof / len(df) * 100

# calculate VIF 
X_vif = df[predictors].copy()

X_vif = SimpleImputer(
    strategy='median'
).fit_transform(X_vif)

X_vif = pd.DataFrame(
    StandardScaler().fit_transform(X_vif),
    columns=predictors
)

vif_results = {}

for i, predictor in enumerate(predictors):

    try:
        vif_results[predictor] = variance_inflation_factor(
            X_vif.values,
            i
        )

    except (ValueError, np.linalg.LinAlgError):
        vif_results[predictor] = np.nan


print("\nVIF results:")

for predictor, value in vif_results.items():
    print(predictor, ":", round(value, 3) if pd.notna(value) else np.nan)


n_out_if = (if_predictions == -1).sum()
pct_out_if = n_out_if / len(df) * 100

# calculate ACF and PACF, ADF statistics
acf_lag1 = np.nan
acf_lag24 = np.nan
pacf_lag1 = np.nan
n_lags_recommended = np.nan
adf_p_global = np.nan



daily_target = (
    df.groupby(df['date'].dt.floor('D'))[target]
    .median()
    .sort_index()
    .dropna()
)

if len(daily_target) >= 5:

    # ACF
    try:
        max_acf_lags = min(24, len(daily_target) - 1)

        acf_values = acf(
            daily_target,
            nlags=max_acf_lags
        )

        if len(acf_values) > 1:
            acf_lag1 = acf_values[1]

        if len(acf_values) > 24:
            acf_lag24 = acf_values[24]

    except (ValueError, np.linalg.LinAlgError):
        pass


    # PACF
    try:
        max_pacf_lags = min(
            24,
            len(daily_target) // 2 - 1
        )

        if max_pacf_lags >= 1:

            pacf_values = pacf(
                daily_target,
                nlags=max_pacf_lags
            )

            if len(pacf_values) > 1:
                pacf_lag1 = pacf_values[1]

            threshold = 1.96 / np.sqrt(len(daily_target))

            significant_lags = np.where(
                np.abs(pacf_values[1:]) > threshold
            )[0] + 1

            if len(significant_lags) > 0:
                n_lags_recommended = int(significant_lags.max())
            else:
                n_lags_recommended = 0

    except (ValueError, np.linalg.LinAlgError):
        pass


    # ADF
    try:
        adf_p_global = adfuller(daily_target)[1]

    except (ValueError, np.linalg.LinAlgError):
        pass


summary_rows = []

correlation_matrix = df[numerical_cols].corr()

for col in numerical_cols:

    valid_values = df[col].dropna()
    
    # calculate again Q1, Q3, IQR...
    q1 = valid_values.quantile(0.25)
    q3 = valid_values.quantile(0.75)
    iqr_value = q3 - q1

    lower_bound = q1 - 1.5 * iqr_value
    upper_bound = q3 + 1.5 * iqr_value

    n_out_iqr_col = (
        (valid_values < lower_bound) |
        (valid_values > upper_bound)
    ).sum()

    pct_out_iqr_col = (
        n_out_iqr_col / len(valid_values) * 100
        if len(valid_values) > 0 else np.nan
    )

    # Z-score
    std_value = valid_values.std()

    if pd.notna(std_value) and std_value > 0:
        z_values = (
            valid_values - valid_values.mean()
        ) / std_value

        n_out_z_col = (z_values.abs() > 3).sum()

    else:
        n_out_z_col = 0

    pct_out_z_col = (
        n_out_z_col / len(valid_values) * 100
        if len(valid_values) > 0 else np.nan
    )

    # correlation with other variables
    other_corr = correlation_matrix[col].drop(
        labels=[col],
        errors='ignore'
    ).dropna()

    if len(other_corr) > 0:
        corr_partner = other_corr.abs().idxmax()
        max_abs_corr = other_corr.abs().max()
    else:
        corr_partner = np.nan
        max_abs_corr = np.nan

    # association with PM2.5
    if col == target:

        pearson_r = 1.0
        pearson_p = np.nan
        spearman_r = 1.0
        spearman_p = np.nan

    else:

        valid = df[[col, target]].dropna()

        if len(valid) >= 3 and valid[col].nunique() > 1:

            pearson_r, pearson_p = pearsonr(
                valid[col],
                valid[target]
            )

            spearman_r, spearman_p = spearmanr(
                valid[col],
                valid[target]
            )

        else:
            pearson_r = np.nan
            pearson_p = np.nan
            spearman_r = np.nan
            spearman_p = np.nan

    # variable family
    if col in ['latitude', 'longitude']:
        family = 'Spatial'

    elif col in [
        'temperature_celsius',
        'pressure_mb',
        'wind_u',
        'wind_v'
    ]:
        family = 'Meteorological'

    elif col == 'AOD':
        family = 'AOD'

    elif col in ['NO2', 'CO', 'O3']:
        family = 'Atmospheric pollutant'

    elif col == target:
        family = 'Target'

    else:
        family = 'Other'

    # units
    if col in ['latitude', 'longitude']:
        unit = 'degrees'

    elif col == 'temperature_celsius':
        unit = '°C'

    elif col == 'pressure_mb':
        unit = 'mb'

    elif col in ['wind_u', 'wind_v']:
        unit = 'm/s'

    elif col == target:
        unit = 'µg/m³'

    else:
        unit = 'dataset unit'

    # notes
    if col == target:
        notes = 'Target variable; temporal statistics calculated.'

    elif col in predictors:
        notes = 'Predictor variable.'

    else:
        notes = 'Spatial variable; not used as a model predictor.'

    # temporal statistics only for PM2.5
    if col == target:
        variable_acf1 = acf_lag1
        variable_acf24 = acf_lag24
        variable_pacf1 = pacf_lag1
        variable_n_lags = n_lags_recommended
        variable_adf = adf_p_global

    else:
        variable_acf1 = np.nan
        variable_acf24 = np.nan
        variable_pacf1 = np.nan
        variable_n_lags = np.nan
        variable_adf = np.nan

    target_assoc = (
        f"Pearson r={pearson_r:.4f}; "
        f"Spearman r={spearman_r:.4f}"
        if pd.notna(pearson_r)
        else np.nan
    )

    
    summary_rows.append({
        'model_id': 'huber_pm25',
        'model_group': 'G3',
        'variable': col,
        'family': family,
        'type': 'numeric',
        'unit': unit,
        'n_total': len(df),
        'n_missing': df[col].isna().sum(),
        'pct_missing': df[col].isna().mean() * 100,
        'n_unique': df[col].nunique(),
        'n_duplicates_key': n_duplicates_key,
        'min': valid_values.min(),
        'q1': q1,
        'median': valid_values.median(),
        'q3': q3,
        'max': valid_values.max(),
        'mean': valid_values.mean(),
        'std': valid_values.std(),
        'skewness': valid_values.skew(),
        'kurtosis': valid_values.kurt(),
        'iqr': iqr_value,
        'n_out_iqr': n_out_iqr_col,
        'pct_out_iqr': pct_out_iqr_col,
        'n_out_z': n_out_z_col,
        'pct_out_z': pct_out_z_col,
        'n_out_if': n_out_if,
        'pct_out_if': pct_out_if,
        'n_out_lof': n_out_lof,
        'pct_out_lof': pct_out_lof,
        'max_abs_corr': max_abs_corr,
        'corr_partner': corr_partner,
        'vif': vif_results.get(col, np.nan),
        'target_assoc': target_assoc,
        'target_pval': pearson_p,
        'acf_lag1': variable_acf1,
        'acf_lag24': variable_acf24,
        'pacf_lag1': variable_pacf1,
        'n_lags_recommended': variable_n_lags,
        'adf_p': variable_adf,
        'notes': notes
    })

data_summary = pd.DataFrame(summary_rows)

data_summary.to_csv("Exploratory_Data.csv", index=False)
print("Exploratory_Data.csv created successfully")

# Exploratory_temporal.csv file we trying to create

temporal_results = []

for sensor in df['sensor_name'].dropna().unique():

    data = df[df['sensor_name'] == sensor].sort_values('date').copy()
    daily = data.groupby(data['date'].dt.floor('D'))[target].median().dropna()

    if len(daily) < 10:
        continue

    dates = pd.date_range(daily.index.min(), daily.index.max(), freq='D')
    daily = daily.reindex(dates)

    clean = daily.dropna()

    intervals = data['date'].diff().dt.total_seconds().dropna()
    missing_pct = daily.isna().mean() * 100
    
    
    try:
        acf_values = acf(clean, nlags=min(24, len(clean)-1))
        acf1 = acf_values[1]
        acf7 = acf_values[7] if len(acf_values) > 7 else np.nan
        acf24 = acf_values[24] if len(acf_values) > 24 else np.nan
    except (ValueError, np.linalg.LinAlgError):
        acf1 = acf7 = acf24 = np.nan

    try:
        adf_p = adfuller(clean)[1]
    except:
        adf_p = np.nan

    try:
        lb_p = acorr_ljungbox(clean, lags=[min(7, len(clean)-1)], return_df=True)['lb_pvalue'].iloc[0]
    except:
        lb_p = np.nan

    trend = linregress(np.arange(len(clean)), clean)

    midpoint = len(clean) // 2
    drift_p = ks_2samp(clean.iloc[:midpoint], clean.iloc[midpoint:]).pvalue

    monthly_missing = {}
    for col in ['NO2', 'CO', 'O3']:
        monthly_missing[col] = (
            data.groupby(data['date'].dt.month)[col]
            .apply(lambda x: round(x.isna().mean() * 100, 2))
            .to_dict()
        )
        
    monthly_median = (data.groupby(data['date'].dt.month)[target].median())
    seasonal_amplitude = (monthly_median.max() - monthly_median.min())

    temporal_results.append({
        'model_id': 'huber_pm25',
        'model_group': 'G3',
        'variable': 'pm25',
        'node_id': sensor,
        'n_records': len(data),
        't_min': data['date'].min(),
        't_max': data['date'].max(),
        'sampling_interval_median_sec': intervals.median(),
        'sampling_interval_std_sec': intervals.std(),
        'pct_missing_intervals': missing_pct,
        'acf_lag1': acf1,
        'acf_lag7': acf7,
        'acf_lag24': acf24,
        'pacf_lag1': np.nan,
        'jung_box_p': lb_p,
        'adf_p': adf_p,
        'kpss_p': np.nan,
        'trend_slope_per_day': trend.slope,
        'trend_pval': trend.pvalue,
        'diurnal_amplitude': np.nan,
        'seasonal_amplitude': seasonal_amplitude,
        'target_drift_stat': ks_2samp(clean.iloc[:midpoint], clean.iloc[midpoint:]).statistic,
        'target_drift_pval': drift_p,
        'n_days_above_25': (clean > 25).sum(),
        'n_days_above_50': (clean > 50).sum(),
        'pct_days_above_25': (clean > 25).mean() * 100,
        'pct_days_above_50': (clean > 50).mean() * 100,
        'regime_breaks': np.nan,
        'cv_strategy_recommended': 'temporal',
        'gap_between_train_test_sec': intervals.median(),
        'n_cycles_observed': data['date'].dt.year.nunique(),
        'no2_pct_missing_by_month': json.dumps(monthly_missing['NO2']),
        'co_pct_missing_by_month': json.dumps(monthly_missing['CO']),
        'o3_pct_missing_by_month': json.dumps(monthly_missing['O3']),
        'notes': 'Daily analysis; diurnal amplitude and regime breaks not calculated.'
    })
    
temporal_df = pd.DataFrame(temporal_results)
temporal_df.to_csv("Exploratory_temporal.csv", index=False)
print("Exploratory_temporal.csv created successfully")

# Timestamp quality and time-series exploration

# check duplicate timestamps and missing dates
df = df.sort_values(['sensor_name', 'date'])

print("\nTimestamp quality")

print("Duplicate timestamps:", df.duplicated(['sensor_name', 'date']).sum())

for sensor, data in df.groupby('sensor_name'):
    
    dates = data['date'].dropna().sort_values().dt.normalize().unique()
    
    if len(dates) > 0:
        dates = pd.DatetimeIndex(dates)
        expected_dates = pd.date_range(
            start=dates.min(), end=dates.max(),freq='D')
        
        missing_dates = len(expected_dates.difference(dates))
        
        print(sensor,
              "Median interval:", data['date'].diff().dt.total_seconds().median(),
              "Missing dates:", missing_dates) 
    
# plot ACF and PACF for one sensor
daily =(df.set_index('date').groupby('sensor_name')[target].resample('D').median())
    
available_sensor = df['sensor_name'].dropna()
series = daily.loc[sensor].dropna()

if len(available_sensor) > 0:
    sensor = available_sensor.iloc[0]
    series = daily.loc[sensor].dropna()
    max_lags = min(30, len(series) // 2 - 1)
    if max_lags >= 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(series, lags=max_lags, ax=axes[0])
        plot_pacf(series, lags=max_lags, ax=axes[1], method='ywm')
        plt.tight_layout()
        plt.show()
else:
    print("Not sensor data availableF")


# test temporal dependence using Ljung-Box and ADF
if len(series) > 7:
    lb_result = acorr_ljungbox(series, lags=[7], return_df=True)
    print("Ljung-Box p-value:", lb_result['lb_pvalue'].iloc[0])
else:
    print("Not enough observation for Ljung_Box")
    
print("ADF p-value:", adfuller(series)[1])


# analyze monthly and daily PM2.5 variation
df['month_year'] = df['date'].dt.to_period('M')

monthly = df.groupby('month_year')[target].agg(
    median='median',
    IQR=lambda x: x.quantile(0.75) - x.quantile(0.25),
    high_frequency=lambda x: (x > 25).mean() * 100)    

print("\nMonthly statistics", monthly)

monthly['median'].plot(figsize=(12, 4), title='Monthly median PM2.5')
plt.show()

df['day'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

print("\nAverage by day of PM2.5")
print(df.groupby('day')[target].mean())

print("\nAverage by month of PM2.5")
print(df.groupby('month')[target].mean())

sns.boxenplot(data=df, x='month', y=target)
plt.title("Monthly variation PM2.5")
plt.show()

# explore correlations between PM2.5 and predictors at different time lags
print("\nLead-lag correlations")

lead_lag_result = []

for sensor, data in df.groupby('sensor_name'):
    data = data.sort_values('date').copy()
    
    daily_data = (
        data.set_index('date')[
            ['pm25', 'AOD', 'NO2', 'CO', 'O3', 'wind_u', 'wind_v', 'temperature_celsius']].resample('D').median())
    
    for col in ['AOD', 'NO2', 'CO', 'O3', 'wind_u', 'wind_v', 'temperature_celsius']:
        
        for lag in [0, 1, 2, 3]:

            shifted = daily_data[col].shift(lag)

            valid = pd.concat([
                daily_data[target],
                shifted], axis=1).dropna()
            
            valid.columns = ['pm25', 'predictor']
            
            if len(valid) >= 3:
                corr = valid[target].corr(valid['predictor'])
                
                lead_lag_result.append({
                    'sensor_name': sensor,
                    'predictor': col,
                    'lag_days': lag,
                    'correlation': corr,
                    'n_observations': len(valid)})
                
lead_lag_df = pd.DataFrame(lead_lag_result)
print(lead_lag_df.head())
print("\nNumber of lead-lag results:", len(lead_lag_df))