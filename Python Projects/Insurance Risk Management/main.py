# -*- coding: utf-8 -*-
"""
Insurance Risk Classification Pipeline
========================================
Trains and compares three classifiers (Random Forest, XGBoost, Bagging)
on an insurance risk dataset, using Pearson-correlation-based feature
selection. Produces cumulative-metric line plots, boxplots, and a
summary table of mean evaluation metrics on the test set.

Usage
-----
    python src/main.py --data data/data.xlsx --desc data/dataDesc.txt

If --data / --desc are omitted, the script falls back to the defaults
defined in DEFAULT_DATA_FILE / DEFAULT_DESC_CANDIDATES below.
"""

import argparse
import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# ==========================================
# Configuration
# ==========================================
DEFAULT_DATA_FILE = "data/data.xlsx"
DEFAULT_DESC_CANDIDATES = [
    "data/dataDesc.txt",
    "data/desc.txt",
    "data/dateDesc.text",
]
DEFAULT_OUTPUT_DIR = "outputs"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_COLORS = {"RF": "#1f77b4", "XGB": "#d62728", "Bagging": "#2ca02c"}
METRIC_NAMES = ["Accuracy", "Recall", "Precision", "F1"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ==========================================
# Data loading
# ==========================================
def load_feature_names(desc_candidates: List[str]) -> Tuple[List[str], Optional[str]]:
    """Parse a feature-description file into an ordered list of feature names.

    Tries each candidate path in order and uses the first one that exists.
    Supports ';', tab, ',' or '|' as separators between an index/number
    column and the feature name.
    """
    desc_path = next((p for p in desc_candidates if os.path.exists(p)), None)
    if desc_path is None:
        logger.warning("No feature-description file found among: %s", desc_candidates)
        return [], None

    feature_names = []
    with open(desc_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = None
            for sep in [";", "\t", ",", "|"]:
                if sep in line:
                    parts = [x.strip() for x in line.split(sep) if x.strip()]
                    break
            if parts and len(parts) >= 2:
                feature_names.append(parts[1] if parts[0].replace(".", "", 1).isdigit() else parts[0])
            else:
                feature_names.append(line)

    logger.info("Loaded %d feature names from %s", len(feature_names), desc_path)
    return feature_names, desc_path


def load_data(data_file: str, feature_names: List[str]) -> pd.DataFrame:
    """Load the raw Excel dataset and assign column names (last column = target)."""
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Excel file not found: {data_file}")

    df = pd.read_excel(data_file, sheet_name=0, header=None, engine="openpyxl")
    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1).reset_index(drop=True)

    n_features = df.shape[1] - 1
    if len(feature_names) >= n_features:
        col_names = list(feature_names[:n_features]) + ["target"]
    else:
        col_names = [f"V{i + 1}" for i in range(n_features)] + ["target"]

    df.columns = col_names
    logger.info("Loaded data: %d rows, %d columns", *df.shape)
    return df


# ==========================================
# Feature selection (Pearson-correlation-based)
# ==========================================
def correlation_feature_selection(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Select features whose correlation with the target is above the mean
    correlation within their sign group (positive / negative)."""
    target_col = "target"
    df_numeric = df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    feature_cols = [c for c in df_numeric.columns if c != target_col]

    corrs = {c: df_numeric[c].corr(df_numeric[target_col], method="pearson") for c in feature_cols}
    corr_series = pd.Series(corrs).sort_values(ascending=False)

    pos_f = corr_series[corr_series > 0]
    neg_f = corr_series[corr_series < 0]
    useful_list = list(pos_f[pos_f > pos_f.mean()].index) + list(neg_f[neg_f < neg_f.mean()].index)

    return df_numeric, corr_series, useful_list


def save_correlation_table(corr_series: pd.Series, output_dir: str) -> None:
    """Save a ranked table of all feature correlations to CSV."""
    corr_table = pd.DataFrame({"Feature": corr_series.index, "Correlation": corr_series.values})
    corr_table = corr_table.reindex(
        corr_table["Correlation"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)
    corr_table.insert(0, "Rank", range(1, len(corr_table) + 1))

    out_path = os.path.join(output_dir, "all_features_correlation.csv")
    corr_table.to_csv(out_path, index=False, float_format="%.6f")
    logger.info("Saved feature-correlation table to %s", out_path)


# ==========================================
# Evaluation helpers
# ==========================================
def calculate_cumulative_metrics(
    y_true: pd.Series, y_probs: np.ndarray, y_preds: np.ndarray
) -> Dict[str, np.ndarray]:
    """Compute accuracy/recall/precision/F1 on growing prefixes of the test
    set, sorted by predicted probability, to visualise performance trends."""
    indices = np.argsort(y_probs)
    y_true_sorted = y_true.iloc[indices].values
    y_preds_sorted = y_preds[indices]

    acc, rec, prec, f1 = [], [], [], []
    for i in range(1, len(y_true_sorted) + 1):
        t, p = y_true_sorted[:i], y_preds_sorted[:i]
        acc.append(accuracy_score(t, p))
        rec.append(recall_score(t, p, zero_division=0))
        prec.append(precision_score(t, p, zero_division=0))
        f1.append(f1_score(t, p, zero_division=0))

    return {
        "Accuracy": np.sort(acc)[::-1],
        "Recall": np.sort(rec)[::-1],
        "Precision": np.sort(prec)[::-1],
        "F1": np.sort(f1)[::-1],
    }


def build_box_data(models: Dict, X_test, y_test) -> pd.DataFrame:
    """Split the test set into 5 chunks per model to build a metric
    distribution suitable for boxplots."""
    rows = []
    for name, model in models.items():
        y_preds = model.predict(X_test)
        chunks = np.array_split(np.arange(len(y_test)), 5)
        for idx in chunks:
            cur_y, cur_p = y_test.iloc[idx], y_preds[idx]
            rows.append(
                {
                    "Model": name,
                    "Accuracy": accuracy_score(cur_y, cur_p),
                    "Recall": recall_score(cur_y, cur_p, zero_division=0),
                    "Precision": precision_score(cur_y, cur_p, zero_division=0),
                    "F1": f1_score(cur_y, cur_p, zero_division=0),
                }
            )
    return pd.DataFrame(rows)


# ==========================================
# Plotting
# ==========================================
def plot_line_metrics(line_results: Dict, output_dir: str) -> None:
    for m_name in METRIC_NAMES:
        plt.figure(figsize=(10, 6))
        for model_name, values_by_metric in line_results.items():
            y_values = values_by_metric[m_name]
            x_values = np.arange(len(y_values))
            plt.plot(
                x_values, y_values, label=model_name,
                color=MODEL_COLORS[model_name], linewidth=2.5, alpha=0.8,
            )
            markers = np.arange(0, len(y_values), 100)
            plt.scatter(
                markers, y_values[markers], color=MODEL_COLORS[model_name],
                s=60, edgecolors="black", zorder=5,
            )

        plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, fontsize=12)
        plt.title(f"Cumulative {m_name} (Sorted Trend)", pad=40, fontweight="bold")
        plt.xlabel("Sorted Test Samples")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"lineplot_{m_name.lower()}.png"), dpi=300)
        plt.close()


def plot_boxplots(box_df: pd.DataFrame, output_dir: str) -> None:
    for m_name in METRIC_NAMES:
        plt.figure(figsize=(9, 6))
        sns.boxplot(x="Model", y=m_name, data=box_df, palette=list(MODEL_COLORS.values()), width=0.5)

        handles = [Line2D([0], [0], color=c, lw=4) for c in MODEL_COLORS.values()]
        plt.legend(
            handles, MODEL_COLORS.keys(), loc="lower center",
            bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, fontsize=12,
        )
        plt.title(f"{m_name} Distribution Comparison", pad=40, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"boxplot_{m_name.lower()}.png"), dpi=300)
        plt.close()


# ==========================================
# CLI / main
# ==========================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Insurance risk classification pipeline")
    parser.add_argument("--data", default=DEFAULT_DATA_FILE, help="Path to the input Excel file")
    parser.add_argument(
        "--desc", nargs="*", default=DEFAULT_DESC_CANDIDATES,
        help="Candidate paths for the feature-description file",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for saved outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "sans-serif"

    # 1) Load data
    feature_names, _ = load_feature_names(args.desc)
    df = load_data(args.data, feature_names)

    # 2) Feature selection
    df_clean, corr_series, selected_features = correlation_feature_selection(df)
    logger.info("Total features: %d | Selected features: %d", len(corr_series), len(selected_features))
    save_correlation_table(corr_series, args.output_dir)

    # 3) Train/test split
    X = df_clean[selected_features]
    y = df_clean["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 4) Train models
    models = {
        "RF": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE),
        "XGB": XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=RANDOM_STATE),
        "Bagging": BaggingClassifier(
            estimator=DecisionTreeClassifier(), n_estimators=100, random_state=RANDOM_STATE
        ),
    }

    line_results = {}
    for name, model in models.items():
        logger.info("Training %s...", name)
        model.fit(X_train, y_train)
        y_preds = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]
        line_results[name] = calculate_cumulative_metrics(y_test, y_probs, y_preds)

    box_df = build_box_data(models, X_test, y_test)

    # 5) Plots
    plot_line_metrics(line_results, args.output_dir)
    plot_boxplots(box_df, args.output_dir)

    # 6) Summary table
    summary_table = box_df.groupby("Model").mean(numeric_only=True).reset_index()
    logger.info("\n%s", summary_table.to_string(index=False))
    summary_table.to_csv(os.path.join(args.output_dir, "mean_metrics.csv"), index=False)


if __name__ == "__main__":
    main()
