# Insurance Risk Classification

A machine learning pipeline that predicts insurance risk (binary classification) and compares three ensemble classifiers — **Random Forest**, **XGBoost**, and **Bagging** — using Pearson-correlation-based feature selection.

## Problem

Given a set of policyholder/risk-related features, the goal is to classify whether a case falls into a high-risk category. This kind of model helps insurers with underwriting decisions, premium pricing, and fraud/risk screening.

## Approach

1. **Feature selection** — Pearson correlation between each feature and the target is computed. Features with above-average positive correlation (or below-average negative correlation) are kept, reducing noise before training.
2. **Modeling** — Three classifiers are trained and compared on a held-out test set (80/20 split, stratified):
   - Random Forest (`class_weight="balanced"` to handle class imbalance)
   - XGBoost
   - Bagging (with a Decision Tree base estimator)
3. **Evaluation** —
   - *Cumulative metric curves*: predictions are sorted by predicted probability to show how Accuracy/Recall/Precision/F1 trend across the test set.
   - *Boxplots*: the test set is split into 5 chunks per model to visualise the distribution/stability of each metric.
   - A summary CSV of mean metrics per model.

## Tech stack

Python · scikit-learn · XGBoost · pandas · NumPy · matplotlib · seaborn

## Project structure

```
insurance-risk-project/
├── src/
│   └── main.py              # full pipeline: load → select features → train → evaluate → plot
├── data/
│   └── sample.csv             # (optional) small sample data for demo purposes. The full data is available in  (https://www.kaggle.com/datasets/kushshah95/the-insurance-company-tic-benchmark)
   └──Description.text  
├── outputs/                  # generated plots and CSV summaries (git-ignored)
├── requirements.txt
├── .gitignore
└── README.md
```

## How to run

```bash
git clone <repo-url>
cd insurance-risk-project
pip install -r requirements.txt

python src/main.py --data data/data.xlsx --desc data/dataDesc.txt --output-dir outputs
```

**Expected input:**
- `data.xlsx`: an Excel file where the last column is the binary target and all other columns are numeric features.
- `dataDesc.txt` *(optional)*: a text file mapping column indices to human-readable feature names (one per line, separated by `;`, tab, `,`, or `|`). If omitted, columns are auto-named `V1, V2, ...`.

**Outputs** (written to `outputs/`):
- `all_features_correlation.csv` — every feature ranked by absolute correlation with the target
- `lineplot_<metric>.png` — cumulative performance curves per model
- `boxplot_<metric>.png` — metric distribution comparison per model
- `mean_metrics.csv` — mean Accuracy/Recall/Precision/F1 per model

## Sample results

*(Add a plot or a screenshot of `outputs/boxplot_f1.png` here once you run the pipeline on your data — this is the single most convincing thing a reviewer will look at.)*

## Possible extensions

- Hyperparameter tuning (GridSearchCV / Optuna)
- SHAP values for model interpretability — important for insurance/underwriting use cases
- Cross-validation instead of a single train/test split
- Wrap as a small CLI or REST API (FastAPI) for serving predictions

## License

MIT
