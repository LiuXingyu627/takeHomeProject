# ML Take-Home Project

Hi! This repository contains my take-home project submission.

I am a new graduate candidate, and this project shows how I approach a business problem end-to-end: data understanding, model building, evaluation, and business recommendations.

## What this project does

Using `census-bureau.data` and `census-bureau.columns`, I built:

1. A **classification model** to predict income class (`>50K` vs `<=50K`)
2. A **segmentation model** to group customers into meaningful personas for strategy

## Files in this repo

- `classification.ipynb` - supervised learning workflow (preprocessing, model comparison, tuning, thresholding, interpretation, and error analysis)
- `segmentation.ipynb` - unsupervised workflow (feature selection, cluster diagnostics, clustering, profiling, and business actions)
- `census-bureau.data` - input dataset
- `census-bureau.columns` - column names for the dataset
- `ML-TakehomeProject.pdf` - original project instructions

## Setup

### Option 1: Conda (recommended)

```bash
conda create -n ml-takehome python=3.12 -y
conda activate ml-takehome
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

### Option 2: Local Python

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

## How to run

1. Open terminal in this project folder
2. Start Jupyter:

```bash
jupyter notebook
```

3. Run notebooks in this order:
   - `classification.ipynb`
   - `segmentation.ipynb`

Note: please run from the repo root so relative paths to the dataset files work.

## What to expect from each notebook

### `classification.ipynb`

- Train/validation/test split
- Baseline model comparison
- Tuned XGBoost results
- Weighted ROC/threshold analysis
- Model interpretation (feature importance)
- False-negative analysis and model limitation notes

### `segmentation.ipynb`

- Feature selection and preprocessing
- K selection (elbow + quality metrics)
- K-Means segmentation
- Robustness comparison with hierarchical clustering and GMM
- Weighted segment profiling and prioritization
- Segment-level business recommendations

## How the two models work together

- **Segmentation** helps decide **what strategy/message** to use for each customer group.
- **Classification** helps decide **who to prioritize first** within each group.

In short: segment first for strategy, then score within segment for execution.

## Reproducibility notes

- Random seeds are used in key model steps (for example `random_state=42`).
- Small result differences can happen across machines due to package/version differences.

## References

- Scikit-learn docs: https://scikit-learn.org/stable/
- XGBoost docs: https://xgboost.readthedocs.io/
