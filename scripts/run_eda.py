# scripts/run_eda.py

import subprocess
import sys
from pathlib import Path


def main():

    # Project root folder
    project_dir = Path(__file__).resolve().parent.parent

    # Notebook paths
    eda_notebook = (
        project_dir /
        "notebooks" /
        "01_eda.ipynb"
    )

    timeseries_notebook = (
        project_dir /
        "notebooks" /
        "02_timeseries.ipynb"
    )

    print("Starting PM2.5 exploratory analysis...")

    # Run EDA notebook
    print("\nRunning 01_eda.ipynb...")

    subprocess.run([
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        str(eda_notebook)
    ], check=True)

    # Run time-series notebook
    print("\nRunning 02_timeseries.ipynb...")

    subprocess.run([
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        str(timeseries_notebook)
    ], check=True)

    print("\nEDA finished successfully.")

    print("\nThe project generated the exploratory results:")
    print("- Exploratory_Data.csv")
    print("- Exploratory_temporal.csv")


if __name__ == "__main__":
    main()