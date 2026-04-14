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

## Notebook Outputs (exact)

### `classification.ipynb` outputs

When you run this notebook end-to-end, you will see:

1. **Data checks**
   - total row count
   - missing value count
   - weighted vs unweighted `>50K` rate
   - label count table
   - weight distribution summary

2. **Exploratory business tables**
   - weighted `P(>50K)` by:
     - education
     - weeks worked in year
     - marital status
     - sex

3. **Baseline model comparison (validation set)**
   - classification reports for:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - XGBoost
   - weighted ROC-AUC for each model

4. **XGBoost tuning results**
   - printed grid results across `max_depth` and `learning_rate`
   - best parameter pair and best validation ROC-AUC

5. **Final model evaluation (test set)**
   - final weighted classification report
   - final weighted test ROC-AUC

6. **Threshold and targeting outputs**
   - cutoff table with:
     - targeted population share
     - score threshold
     - weighted precision (hit rate)
     - weighted recall (coverage)
     - lift vs random
   - recommended operating threshold + expected precision/recall/lift

7. **Interpretability outputs**
   - top encoded features by importance
   - aggregated base-feature importance table for business readability

8. **Error analysis outputs**
   - weighted false negative rate at the recommended threshold
   - false-negative profile tables by subgroup

9. **Visualization**
   - ROC curve on held-out test set with AUC in legend

### `segmentation.ipynb` outputs

When you run this notebook end-to-end, you will see:

1. **Preprocessing confirmation**
   - selected segmentation feature subset
   - transformed matrix creation confirmation

2. **K-selection diagnostics**
   - Elbow plot (`k=2..8`)
   - cluster quality metrics table across candidate `k`:
     - silhouette
     - calinski-harabasz
     - davies-bouldin

3. **Final clustering output**
   - K-Means segment assignment with `k=5`
   - raw segment size counts

4. **Segment driver interpretation**
   - top feature importance chart from Random Forest explainer model

5. **Weighted business profiling**
   - overall weighted `>50K` rate
   - weighted segment profile table including:
     - population share
     - `>50K` earners rate
     - lift vs population average
     - opportunity index
     - top marital status / education / occupation
   - segment priority ranking table

6. **Business strategy output**
   - segment-by-segment recommended actions
   - implementation notes for monitoring and refresh cadence

7. **Robustness comparison**
   - comparison table: K-Means vs Hierarchical vs GMM
   - same evaluation metrics + runtime
   - printed recommended model from benchmark

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
