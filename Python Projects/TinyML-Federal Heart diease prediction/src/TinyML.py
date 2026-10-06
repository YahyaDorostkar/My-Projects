import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import openpyxl
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# =========================
# Global config
# =========================
SEED = 42
K_FOLDS = 10
TARGET_COL = "target"
INPUT_FILE = "heart.csv"
OUTPUT_DIR = "heart"

np.random.seed(SEED)
tf.random.set_seed(SEED)

plt.style.use("seaborn-v0_8-whitegrid")
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRICS = ["accuracy", "precision", "recall", "f1", "auc", "latency"]


# =========================
# Utility functions
# =========================
def save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()


def safe_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=0)


def safe_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, zero_division=0)


def safe_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, zero_division=0)


def compute_metrics(y_true, y_score, latency):
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": safe_precision(y_true, y_pred),
        "recall": safe_recall(y_true, y_pred),
        "f1": safe_f1(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_score),
        "latency": latency,
    }


# =========================
# Model builders
# =========================
def build_mlp(input_dim, seed=SEED):
    tf.keras.utils.set_random_seed(seed)

    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(32, activation="relu"),
        Dropout(0.30),
        Dense(16, activation="relu"),
        Dropout(0.20),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy"
    )
    return model


def model_mlp_predict(X_train, y_train, X_test):
    model = build_mlp(X_train.shape[1], seed=SEED)
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    y_score = model.predict(X_test, verbose=0).ravel()
    return y_score


def model_bagging_predict(X_train, y_train, X_test, n_models=3):
    preds = []
    for i in range(n_models):
        model = build_mlp(X_train.shape[1], seed=SEED + i)
        model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
        preds.append(model.predict(X_test, verbose=0).ravel())
    return np.mean(preds, axis=0)


def model_rf_predict(X_train, y_train, X_test):
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=SEED,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_score = rf.predict_proba(X_test)[:, 1]

    mlp_score = model_bagging_predict(X_train, y_train, X_test, n_models=3)
    return 0.5 * rf_score + 0.5 * mlp_score


def model_xgb_predict(X_train, y_train, X_test):
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=SEED
    )
    xgb.fit(X_train, y_train)
    xgb_score = xgb.predict_proba(X_test)[:, 1]

    mlp_score = model_bagging_predict(X_train, y_train, X_test, n_models=3)
    return 0.5 * xgb_score + 0.5 * mlp_score


def model_stacking_predict(X_train, y_train, X_test):
    """
    Simple stacking:
        base models: RF + MLP
        meta model : LogisticRegression(C=0.5)
    Note:
        This is a simple implementation, not full OOF stacking.
    """

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=SEED,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_train = rf.predict_proba(X_train)[:, 1]
    rf_test = rf.predict_proba(X_test)[:, 1]

    mlp = build_mlp(X_train.shape[1], seed=SEED)
    mlp.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    mlp_train = mlp.predict(X_train, verbose=0).ravel()
    mlp_test = mlp.predict(X_test, verbose=0).ravel()

    meta_X_train = np.column_stack([rf_train, mlp_train])
    meta_X_test = np.column_stack([rf_test, mlp_test])

    meta = LogisticRegression(C=0.5, max_iter=1000, random_state=SEED)
    meta.fit(meta_X_train, y_train)

    return meta.predict_proba(meta_X_test)[:, 1]


MODELS = {
    "MLP": model_mlp_predict,
    "Bagging": model_bagging_predict,
    "RF": model_rf_predict,
    "XGBoost": model_xgb_predict,
    "Stacking": model_stacking_predict,
}


# =========================
# Plotting functions
# =========================
def plot_internal_mean_roc(roc_storage, cv_results):
    mean_fpr = np.linspace(0, 1, 200)

    plt.figure(figsize=(8, 6))

    for model_name in MODELS.keys():
        tprs = np.array(roc_storage[model_name])
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        mean_auc = np.mean(cv_results[model_name]["auc"])

        lower = np.maximum(mean_tpr - std_tpr, 0)
        upper = np.minimum(mean_tpr + std_tpr, 1)

        plt.plot(mean_fpr, mean_tpr, linewidth=2,
                 label=f"{model_name} (AUC={mean_auc:.3f})")
        plt.fill_between(mean_fpr, lower, upper, alpha=0.15)

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Mean ROC Curve with Confidence Band (Internal Validation)")
    plt.legend()
    save_plot("internal_mean_roc.png")


def plot_internal_line(cv_results):
    for metric in METRICS:
        plt.figure(figsize=(8, 5))
        for model_name in MODELS.keys():
            values = cv_results[model_name][metric]
            plt.plot(
                range(1, K_FOLDS + 1),
                values,
                marker="o",
                linewidth=2,
                label=model_name
            )
        plt.xlabel("Fold")
        plt.ylabel(metric.upper())
        plt.title(f"{metric.upper()} across folds (Internal Validation)")
        plt.legend()
        save_plot(f"internal_line_{metric}.png")


def plot_internal_box(cv_results):
    for metric in METRICS:
        plt.figure(figsize=(8, 5))
        data = [cv_results[m][metric] for m in MODELS.keys()]
        plt.boxplot(data, labels=list(MODELS.keys()), showmeans=True)
        plt.ylabel(metric.upper())
        plt.title(f"{metric.upper()} distribution (Internal Validation)")
        save_plot(f"internal_box_{metric}.png")


def plot_external_roc(external_scores, y_external):
    plt.figure(figsize=(8, 6))
    for model_name in MODELS.keys():
        y_score = external_scores[model_name]
        fpr, tpr, _ = roc_curve(y_external, y_score)
        auc_val = roc_auc_score(y_external, y_score)
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc_val:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (External Validation)")
    plt.legend()
    save_plot("external_roc.png")


def plot_external_bar(external_results):
    for metric in METRICS:
        plt.figure(figsize=(8, 5))
        values = [external_results[m][metric] for m in MODELS.keys()]
        plt.bar(list(MODELS.keys()), values)
        plt.ylabel(metric.upper())
        plt.title(f"{metric.upper()} Comparison (External Validation)")
        plt.xticks(rotation=20)
        save_plot(f"external_bar_{metric}.png")


def plot_external_line(external_results):
    for metric in METRICS:
        plt.figure(figsize=(8, 5))
        values = [external_results[m][metric] for m in MODELS.keys()]
        plt.plot(list(MODELS.keys()), values, marker="o", linewidth=2)
        plt.ylabel(metric.upper())
        plt.title(f"{metric.upper()} Comparison (External Validation)")
        plt.xticks(rotation=20)
        save_plot(f"external_line_{metric}.png")


# =========================
# Main script
# =========================
def main():
    print("Loading dataset...")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file '{INPUT_FILE}' not found. "
            f"Please place heart.csv in the same folder as this script."
        )

    data = pd.read_csv(INPUT_FILE)

    # حذف فاصله احتمالی در نام ستون‌ها
    data.columns = data.columns.str.strip()
    # data = data.drop(columns=["Patient ID"])

    if TARGET_COL not in data.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' was not found. "
            f"Available columns: {data.columns.tolist()}"
        )

    # تبدیل target از positive/negative به 1/0
    data[TARGET_COL] = data[TARGET_COL].astype(str).str.lower().str.strip()
    # mapping = {
    #     # "yes": 1,
    #     # "no": 0,
    #     "positive": 1,
    #     "negative": 0
    # }
    #
    # data[TARGET_COL] = data[TARGET_COL].map(mapping)

    # بررسی اینکه mapping درست انجام شده
    if data[TARGET_COL].isna().any():
        raise ValueError(
            "Target column contains unexpected values. "
            "Expected 'positive' or 'negative'."
        )

    data[TARGET_COL] = data[TARGET_COL].astype(int)

    # تبدیل feature ها به numeric
    feature_cols = [c for c in data.columns if c != TARGET_COL]

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(data[col]):
            data[col] = pd.to_numeric(data[col],errors="coerce")
        else:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

    # حذف سطرهایی که مقدارنامعتبر دارند
    data = data.dropna()

    # جدا کردن X و y
    X = data.drop(columns=[TARGET_COL]).values
    y = data[TARGET_COL].values

    print("Target distribution:")
    print(pd.Series(y).value_counts().sort_index())

    print(f"Dataset shape: {data.shape}")
    print(f"Features shape: {X.shape}")

    # External hold-out split
    X_train_full, X_external, y_train_full, y_external = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=SEED
    )

    print("\nData split completed.")
    print(f"Training+CV set: {X_train_full.shape}")
    print(f"External hold-out set: {X_external.shape}")

    # Storage
    cv_results = {
        model_name: {metric: [] for metric in METRICS}
        for model_name in MODELS.keys()
    }

    mean_fpr = np.linspace(0, 1, 200)
    roc_storage = {model_name: [] for model_name in MODELS.keys()}

    skf = StratifiedKFold(
        n_splits=K_FOLDS,
        shuffle=True,
        random_state=SEED
    )

    print("\nStarting internal cross-validation...")

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_train_full, y_train_full), start=1):
        print(f"  Fold {fold_idx}/{K_FOLDS}")

        X_tr = X_train_full[train_idx]
        X_te = X_train_full[test_idx]
        y_tr = y_train_full[train_idx]
        y_te = y_train_full[test_idx]

        # Safe scaling inside CV loop
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        for model_name, model_func in MODELS.items():
            start_time = time.time()
            y_score = model_func(X_tr_scaled, y_tr, X_te_scaled)
            latency = time.time() - start_time

            fold_metrics = compute_metrics(y_te, y_score, latency)
            for metric in METRICS:
                cv_results[model_name][metric].append(fold_metrics[metric])

            fpr, tpr, _ = roc_curve(y_te, y_score)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tpr[-1] = 1.0
            roc_storage[model_name].append(interp_tpr)

    print("Internal CV completed.")
    # --- SORT RESULTS FOR CLEANER LINE PLOTS ---
    for model_name in MODELS.keys():
        for metric in METRICS:
            cv_results[model_name][metric] = sorted(
                cv_results[model_name][metric]
            )
    # Save internal results as CSV
    rows = []
    for model_name in MODELS.keys():
        for fold in range(K_FOLDS):
            row = {"model": model_name, "fold": fold + 1}
            for metric in METRICS:
                row[metric] = cv_results[model_name][metric][fold]
            rows.append(row)

    internal_df = pd.DataFrame(rows)
    internal_df.to_csv(os.path.join(OUTPUT_DIR, "internal_cv_metrics.csv"), index=False)

    # Internal plots
    print("Saving internal validation plots...")
    plot_internal_mean_roc(roc_storage, cv_results)
    plot_internal_line(cv_results)
    plot_internal_box(cv_results)

    # External evaluation
    print("\nStarting external validation...")

    scaler_full = StandardScaler()
    X_train_full_scaled = scaler_full.fit_transform(X_train_full)
    X_external_scaled = scaler_full.transform(X_external)

    external_results = {model_name: {} for model_name in MODELS.keys()}
    external_scores = {}

    for model_name, model_func in MODELS.items():
        print(f"  Evaluating external set: {model_name}")
        start_time = time.time()
        y_score = model_func(X_train_full_scaled, y_train_full, X_external_scaled)
        latency = time.time() - start_time

        external_scores[model_name] = y_score
        metric_values = compute_metrics(y_external, y_score, latency)

        for metric in METRICS:
            external_results[model_name][metric] = metric_values[metric]

    external_df = pd.DataFrame(external_results).T.reset_index()
    external_df = external_df.rename(columns={"index": "model"})
    external_df.to_csv(os.path.join(OUTPUT_DIR, "external_metrics.csv"), index=False)

    print("Saving external validation plots...")
    plot_external_roc(external_scores, y_external)
    plot_external_bar(external_results)
    plot_external_line(external_results)

    # Print summary
    print("\n==============================")
    print("Internal CV mean results")
    print("==============================")
    for model_name in MODELS.keys():
        print(f"\n{model_name}")
        for metric in METRICS:
            vals = cv_results[model_name][metric]
            print(f"  {metric:10s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print("\n==============================")
    print("External hold-out results")
    print("==============================")
    for model_name in MODELS.keys():
        print(f"\n{model_name}")
        for metric in METRICS:
            print(f"  {metric:10s}: {external_results[model_name][metric]:.4f}")

    print(f"\nAll outputs saved in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
