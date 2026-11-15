# preprocessing.py

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from src.config import (SEED, file_names, target, predictors, variables, numerical_cols, sentinel_col)

def load_and_clean_data(file):
    
    df = pd.read_csv("combined_gee_pm25.csv")

    # ensure date is valid
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # remove missing PM2.5
    df = df.dropna(subset=[target]).copy()

    # remove invalid PM2.5 values
    invalid_pm25 = ((df[target] <= 0) | (df[target] >= 500))

    df = df.loc[~invalid_pm25].copy()

    return df


def chronological_split(df):
    
    df = df.sort_values("date").reset_index(drop=True)

    n = len(df)

    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    validation_df = df.iloc[train_end:validation_end].copy()
    test_df = df.iloc[validation_end:].copy()

    return train_df, validation_df, test_df


def prepare_features(train_df, validation_df, test_df):
    
    X_train = train_df[predictors].copy()
    y_train = train_df[target].copy()

    X_validation = validation_df[predictors].copy()
    y_validation = validation_df[target].copy()

    X_test = test_df[predictors].copy()
    y_test = test_df[target].copy()

    # create missingness indicators
    for col in sentinel_col:
        X_train[col + "_missing"] = X_train[col].isna().astype(int)

        X_validation[col + "_missing"] = (X_validation[col].isna().astype(int))

        X_test[col + "_missing"] = (X_test[col].isna().astype(int))

    # Median imputation fitted only on training data
    imputer = SimpleImputer(strategy="median")

    X_train_imputed = imputer.fit_transform(X_train)
    X_validation_imputed = imputer.transform(X_validation)
    X_test_imputed = imputer.transform(X_test)

    # Scaling fitted only on training data
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_validation_scaled = scaler.transform(
        X_validation_imputed
    )
    X_test_scaled = scaler.transform(X_test_imputed)

    # Feature selection using training data only
    selector = SelectKBest(
        score_func=f_regression,
        k=8
    )

    X_train_selected = selector.fit_transform(
        X_train_scaled,
        y_train
    )

    X_validation_selected = selector.transform(
        X_validation_scaled
    )

    X_test_selected = selector.transform(
        X_test_scaled
    )

    feature_names = X_train.columns.tolist()

    selected_features = [
        feature_names[i]
        for i in selector.get_support(indices=True)
    ]

    print("\nSelected features:")
    print(selected_features)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_validation": X_validation,
        "y_validation": y_validation,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_selected": X_train_selected,
        "X_validation_selected": X_validation_selected,
        "X_test_selected": X_test_selected,
        "selected_features": selected_features,
        "imputer": imputer,
        "scaler": scaler,
        "selector": selector
    }


def compare_outlier_treatment(
    X_train,
    y_train,
    X_test,
    y_test
):
    
    # Calculate thresholds using training data only
    Q1 = X_train.quantile(0.25)
    Q3 = X_train.quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    X_train_treated = X_train.clip(
        lower=lower_bound,
        upper=upper_bound,
        axis=1
    )

    X_test_treated = X_test.clip(
        lower=lower_bound,
        upper=upper_bound,
        axis=1
    )

    # Model without outlier treatment
    model_original = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(
            score_func=f_regression,
            k=8
        )),
        ("model", HuberRegressor())
    ])

    model_original.fit(X_train, y_train)

    y_pred_original = model_original.predict(X_test)

    # Model with outlier treatment
    model_treated = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(
            score_func=f_regression,
            k=8
        )),
        ("model", HuberRegressor())
    ])

    model_treated.fit(
        X_train_treated,
        y_train
    )

    y_pred_treated = model_treated.predict(
        X_test_treated
    )

    # Evaluation
    comparison = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R2"],
        "Without outlier treatment": [
            mean_absolute_error(
                y_test,
                y_pred_original
            ),
            np.sqrt(mean_squared_error(
                y_test,
                y_pred_original
            )),
            r2_score(
                y_test,
                y_pred_original
            )
        ],
        "With outlier treatment": [
            mean_absolute_error(
                y_test,
                y_pred_treated
            ),
            np.sqrt(mean_squared_error(
                y_test,
                y_pred_treated
            )),
            r2_score(
                y_test,
                y_pred_treated
            )
        ]
    })

    return comparison