# PM2.5 Forecasting Using Huber Regression

## 1. Project Overview
This project focuses on forecasting PM2.5 air pollution levels using meteorological and air-quality variables from the Mediterranean Basin.
The main objective is to develop a forecasting model using **Huber Regression**, a robust regression method that is less sensitive to outliers than ordinary linear regression.

## 2. Dataset
- **Source:** [Kaggle – Air Quality Mediterranean Basin (2016–2026)](https://www.kaggle.com/datasets/sarahameziane/air-quality-mediterranean-basin-20162026/)
- **Target variable:** PM2.5
- **Predictor variables:** Temperature, pressure, wind components, NO2, CO, O3, and AOD.
- **Period:** 2016–2026.
- 
## 3. Technologies and Libraries
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Statsmodels
- SciPy

## 4. Project Structure
As it shown in my GitHub

## 5. Installation
Clone the repository:

```bash
git clone https://github.com/meriembelaidi/pm25-forecasting-Belaidi-Ayache-Lakel
cd pm25-forecasting-Belaidi-Ayache-Lakel
```

Install the required libraries:

```bash
pip install -r requirements.txt
```
## 6. Methodology

The project includes:
1. Dataset inspection and exploratory data analysis.
2. Data cleaning and missing-value treatment.
3. Outlier detection and analysis.
4. Time-series analysis.
5. Feature engineering and preprocessing.
6. Chronological data splitting.
7. Temporal cross-validation.
8. Huber Regression model development and evaluation.

## 8. Evaluation Metrics

The model will be evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)
