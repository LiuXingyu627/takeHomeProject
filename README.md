# ML Take-Home Project

This repository contains my submission for the machine learning take-home project using the provided census dataset.

## Project Objective

Using the files `census-bureau.data` and `census-bureau.columns`, this project delivers:

1. A **classification model** to predict whether income is `>50K` or `<=50K`.
2. A **segmentation model** to identify distinct population groups and support marketing strategy.

## Repository Contents

- `classification.ipynb` - end-to-end supervised modeling workflow (preprocessing, model comparison, tuning, weighted evaluation, threshold strategy, interpretation, and error analysis).
- `segmentation.ipynb` - unsupervised segmentation workflow (feature selection, k selection diagnostics, clustering, interpretation, weighted profiling, and business recommendations).
- `census-bureau.data` - raw input dataset.
- `census-bureau.columns` - column names aligned to the dataset.
- `ML-TakehomeProject.pdf` - original project instructions.

## Environment Setup

### Option 1: Using Anaconda (recommended)

1. Open a terminal in the project folder.
2. Create and activate an environment:

```bash
conda create -n ml-takehome python=3.12 -y
conda activate ml-takehome
```

3. Install required packages:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

### Option 2: Using system/local Python

Install dependencies directly:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

## How to Run

1. Launch Jupyter from the repository root:

```bash
jupyter notebook
```

2. Open and run notebooks in this order:
   - `classification.ipynb`
   - `segmentation.ipynb`

> Note: Both notebooks assume they are run from the repository root so they can read `census-bureau.data` and `census-bureau.columns` with relative paths.

## Notebook Outputs

### 1) Classification (`classification.ipynb`)

Main outputs include:
- train/validation/test split summary
- baseline model comparison (Logistic Regression, Decision Tree, Random Forest, XGBoost)
- tuned XGBoost final performance on held-out test data
- weighted threshold operating table (precision/recall/lift)
- feature-importance interpretation (encoded and business-friendly aggregated views)
- false-negative profiling and governance notes

### 2) Segmentation (`segmentation.ipynb`)

Main outputs include:
- selected feature subset and preprocessing pipeline
- k-selection diagnostics (Elbow + quantitative quality metrics)
- final K-Means segment assignment
- top feature drivers of segmentation
- weighted segment profile table and segment prioritization ranking
- business actions and implementation notes

## Methodology Notes

- **Population weighting** (`weight` column) is incorporated where appropriate to make results population-representative.
- **Classification and segmentation are complementary**:
  - segmentation defines high-level strategy by customer group,
  - classification score supports within-group targeting priority.

## Reproducibility

- Random seeds are set in model steps where relevant (for example `random_state=42`).
- If results differ slightly across systems, this is expected due to environment/library variations.

## References

Key references used while building the solution:
- Scikit-learn documentation: https://scikit-learn.org/stable/
- XGBoost documentation: https://xgboost.readthedocs.io/
