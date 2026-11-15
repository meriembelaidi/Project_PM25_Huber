# splits.py
import json

from sklearn.model_selection import (TimeSeriesSplit, GroupKFold, KFold)

from src.config import SEED


def create_validation_splits(
    train_df,
    validation_df,
    predictors,
    output_dir="outputs"
):

    # Development data
    development_df = train_df.copy()

    development_df = development_df._append(validation_df, ignore_index=True)

    development_df = development_df.sort_values("date").reset_index(drop=True)

    X_development = development_df[predictors]

    print("\nDevelopment observations:", len(development_df))

    # Temporal cross-validation
    tscv = TimeSeriesSplit(n_splits=5, gap=1)

    temporal_folds = list(tscv.split(X_development))

    temporal_summary = []

    for i, (train_idx, val_idx) in enumerate(temporal_folds, start=1):

        temporal_summary.append({
            "fold": i,
            "train_size": len(train_idx),
            "validation_size": len(val_idx)
        })

    with open(f"{output_dir}/temporal_cv.json", "w") as f:

        json.dump(temporal_summary, f, indent=4)

    # Spatial cross-validation
    groups = development_df["sensor_name"]

    gkf = GroupKFold(n_splits=5)

    spatial_folds = list(gkf.split(X_development, groups=groups))

    spatial_summary = []

    for i, (train_idx, val_idx) in enumerate(spatial_folds, start=1):

        spatial_summary.append({
            "fold": i,
            "train_size": len(train_idx),
            "validation_size": len(val_idx)
        })

    with open(f"{output_dir}/spatial_cv.json", "w") as f:

        json.dump(spatial_summary, f, indent=4)

    # Random cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    random_folds = list(kf.split(X_development))

    random_summary = []

    for i, (train_idx, val_idx) in enumerate(random_folds,start=1):

        random_summary.append({
            "fold": i,
            "train_size": len(train_idx),
            "validation_size": len(val_idx)
        })

    with open(f"{output_dir}/random_cv.json", "w") as f:

        json.dump(random_summary, f, indent=4)

    print("temporal_cv.json")
    print("spatial_cv.json")
    print("random_cv.json")

    return (
        temporal_folds,
        spatial_folds,
        random_folds
    )