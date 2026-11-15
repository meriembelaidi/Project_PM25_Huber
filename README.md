# PM2.5 forecasting mini project

The goal is daily PM2.5 forecasting in the Mediterranean Basin. Later parts will compare predictive performance, robustness across validation choices, and the stability of explainability results. Part 1 is complete: the data have been audited, cleaned, explored, and prepared without fitting transformations on validation or test data.

## Dataset

The dataset is the Kaggle [Air Quality - Mediterranean Basin (2016-2026)](https://www.kaggle.com/datasets/sarahameziane/air-quality-mediterranean-basin20162026/). Put the downloaded CSV files in `data/raw/`. The license was not included in the local files, so it must be checked on the Kaggle page.

The merged input used in Part 1 has 41,718 rows and 15 columns. After the documented PM2.5 cleaning rules, 40,983 rows remain from 91 stations. The development set has 31,286 rows and the complete 2025 test year has 8,149 rows. Partial 2026 rows are kept outside both sets.

## Current status

Part 1 is done: exploration and preprocessing. Model training, statistical comparisons, robustness experiments, and explainability are coming later.

## Structure

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── figures/
│   └── slides/
├── folds/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_timeseries.ipynb
├── presentation/
│   └── Part1_EDA_Presentation.pptx
├── results/
│   ├── Exploratory_Data.csv
│   └── Exploratory_temporal.csv
├── scripts/
│   └── run_eda.py
├── src/
│   ├── config.py
│   ├── features.py
│   ├── preprocessing.py
│   └── splits.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

The project was developed with Python 3.10.5.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Reproduce Part 1

Run from the project root:

```bash
python scripts/run_eda.py
```

Then open and run `notebooks/01_eda.ipynb` followed by `notebooks/02_timeseries.ipynb` from a fresh kernel. The random seed is 42.

## Main outputs

- `results/Exploratory_Data.csv`: audit, missingness, cleaning counts, and numerical summaries.
- `results/Exploratory_temporal.csv`: timestamp, station-gap, autocorrelation, stationarity, drift, and lead-lag results.
- `data/processed/dev.csv`: development rows before 2025.
- `data/processed/test.csv`: the independent complete 2025 test period.
- `data/processed/features_all.csv`: engineered rows, including partial 2026 rows held outside development and test.
- `data/processed/feature_groups.json`: feature families for later work.
- `folds/`: temporal, spatial, and shuffled development folds.
- `figures/`: Part 1 diagnostic figures and presentation figures under `figures/slides/`.

## Team

- Name: BELAIDI Meriem
- Name: Ayache Farah
- Name: Lakel khadidja Lyne

