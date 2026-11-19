# scripts/run_training.py

import sys
from pathlib import Path

# Add the project root so Python can find src
project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))

from src.config import target, predictors

from src.preprocessing import (
    load_and_clean_data,
    chronological_split,
    prepare_features,
    compare_outlier_treatment
)

from src.features import create_temporal_features

from src.splits import create_validation_splits


def main():

    print("PM2.5 HUBER REGRESSION EXPERIMENT")

    # LOAD DATA

    data_path = (
        project_dir /"data" /"raw" /"combined_gee_pm25.csv")

    print("\nLoading data")

    df = load_and_clean_data(data_path)

    print("Dataset shape:", df.shape)


    # CHRONOLOGICAL SPLIT
    print("\nCreating chronological split")

    train_df, validation_df, test_df = (
        chronological_split(df)
    )

    print("Train observations:", len(train_df))
    print("Validation observations:", len(validation_df))
    print("Test observations:", len(test_df))

    print("\nTrain period:", train_df["date"].min(), "to", train_df["date"].max())

    print("Validation period:", validation_df["date"].min(), "to", validation_df["date"].max())

    print("Test period:", test_df["date"].min(), "to", test_df["date"].max())


    # PREPROCESSING
    print("\nPreprocessing data")

    data = prepare_features(train_df, validation_df, test_df)

    print("\nSelected features:")
    print(data["selected_features"])


    # OUTLIER TREATMENT COMPARISON
    print("\nComparing outlier treatment")

    comparison = compare_outlier_treatment(
        data["X_train"],
        data["y_train"],
        data["X_test"],
        data["y_test"]
    )

    print("\nOutlier treatment comparison:")
    print(comparison.round(4).to_string(index=False))


    # TEMPORAL FEATURE ENGINEERING

    print("\nCreating temporal features")

    daily_df = create_temporal_features(df)

    print("Daily timeline shape:", daily_df.shape)


    # VALIDATION SCHEMAS
    print("\nCreating validation schemas")

    output_dir = project_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    create_validation_splits(
        train_df,
        validation_df,
        predictors,
        output_dir=output_dir
    )

    print("\nValidation files created:")
    print("- temporal_cv.json")
    print("- spatial_cv.json")
    print("- random_cv.json")


    # FINISHED
    print("\nEXPERIMENT FINISHED SUCCESSFULLY")

if __name__ == "__main__":
    main()
