# A Bayesian-Optimized Heterogeneous Deep Stacking Ensemble for Statistically Reliable Cardiovascular Disease Prediction

## Authors

Yahya Dorostkar Navaei, Yaser Ahangari Nanehkaran

## Description

This repository provides a Python implementation of a Bayesian-optimized heterogeneous deep stacking ensemble for cardiovascular disease prediction. The pipeline combines deep learning architectures with tuned classical machine learning models and a CatBoost meta-learner to construct the final stacked ensemble.

The implementation is designed as a reproducible experimental pipeline for tabular cardiovascular classification datasets. The dataset is supplied as a CSV file through the command line, while the target column can be selected by name or by using `-1` to indicate the last column.

The same code has also been added to the GitHub repository/account of **Yaser Ahangari Nanehkaran**.

## Main Features

- Dataset-agnostic CSV loading for tabular classification tasks
- Automatic handling of categorical and numerical predictors
- One-hot encoding of detected categorical variables
- Standardization with `StandardScaler`
- Bayesian hyperparameter optimization using Optuna and the TPE sampler
- Independent hyperparameter searches for MLP, 1D-CNN, and LSTM architectures
- Automatic selection among XGBoost, LightGBM, Random Forest, and SVM
- Autoencoder-based latent feature extraction for the stacking layer
- CatBoost-based final meta-learner with its own Optuna optimization stage
- Out-of-fold (OOF) meta-feature generation
- Repeated stratified train/test evaluation
- Ablation comparison between the stacked ensemble and individual base learners
- 95% confidence intervals for all reported metrics
- Paired Wilcoxon signed-rank tests
- Cohen's d effect sizes
- Pooled confusion matrix, ROC curves, and Precision-Recall curves
- Automatic export of numerical results, hyperparameters, and publication-oriented figures

## Dataset Information

The code is intentionally dataset-agnostic and accepts a generic tabular classification CSV file.

The default configuration in the script is:

```text
../heart_attack_prediction_dataset.csv
```

The default target is:

```text
-1
```

which means that the last column of the CSV file is used as the target variable.

The datasets analyzed for this study can be found in the Kaggle repository. Heart Disease Dataset, available at:

https://www.kaggle.com/datasets/sukhmandeepsinghbrar/heart-attack-dataset

https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset.


The loader supports:

- Numeric-only datasets
- String/object categorical variables
- Integer-coded categorical variables
- Mixed numerical and categorical datasets

By default, rows containing missing values are removed. Exact duplicate rows are also removed before splitting unless the `--keep_duplicates` option is used.

For integer columns, variables with at most 10 distinct values are automatically treated as categorical unless they are explicitly forced to remain numeric.

The dataset files are not embedded in this repository and must be provided by the user according to the corresponding study or benchmark protocol.

## Methodology

### 1. Data Preprocessing

The preprocessing stage performs the following operations:

- Read the CSV dataset
- Remove rows with missing values
- Remove exact duplicate rows by default
- Identify categorical and numerical features
- Apply one-hot encoding to categorical variables
- Encode class labels using `LabelEncoder`
- Standardize the resulting feature matrix using `StandardScaler`

### 2. Bayesian Hyperparameter Optimization

Optuna is used with the Tree-structured Parzen Estimator (TPE) sampler.

The classical model search considers:

- XGBoost
- LightGBM
- Random Forest
- Support Vector Machine (SVM)

The selected classical model is optimized using the macro F1-score as the objective.

Each deep architecture has an independent search space:

**MLP**
- First hidden-layer width
- Second hidden-layer width
- Dropout rate
- Learning rate
- Batch size

**1D-CNN**
- Number of convolutional filters
- Dense-layer width
- Dropout rate
- Learning rate
- Batch size

**LSTM**
- Number of LSTM units
- Dense-layer width
- Dropout rate
- Learning rate
- Batch size

Early stopping is used during deep-learning training.

### 3. Heterogeneous Stacking Architecture

The stacking pipeline combines:

- MLP probability outputs
- CNN probability outputs
- LSTM probability outputs
- Autoencoder latent representations
- Probability outputs from the best classical model selected by Optuna

The autoencoder uses a latent representation with a default dimension of 32.

These heterogeneous representations are concatenated to form the meta-feature matrix.

A CatBoost classifier is used as the final meta-learner. Its main hyperparameters are also optimized with Optuna.

### 4. Out-of-Fold Meta-Feature Generation

During the outer evaluation, the training portion of each repeated split is divided into stratified OOF folds.

The default number of OOF folds is:

```text
5
```

Predictions generated from these folds are used to construct the meta-feature matrix without directly using the corresponding validation fold for base-model fitting.

For the test set, predictions from the OOF-trained base models are averaged across folds before being passed to the meta-learner.

### 5. Repeated Experimental Evaluation

The default outer evaluation uses:

```text
30 repeated stratified train/test splits
```

with:

```text
80% training
20% testing
```

The random seed starts from 42 and is incremented for each outer run.

The default optimization settings are:

```text
Classical Optuna trials: 25
Deep-learning trials per architecture: 15
Meta-learner Optuna trials: 15
Deep-learning maximum epochs: 40
```

A reduced `--quick` mode is also available for smoke testing.

## Implemented Models

### Deep Learning Base Learners

- Multi-Layer Perceptron (MLP)
- One-Dimensional Convolutional Neural Network (1D-CNN)
- Long Short-Term Memory (LSTM)

### Classical Machine Learning Search Space

- XGBoost
- LightGBM
- Random Forest
- Support Vector Machine (SVM)

### Representation Learning

- Fully connected autoencoder for latent feature extraction

### Final Meta-Learner

- CatBoost Classifier

## Assessment Metrics

The pipeline computes the following metrics:

- **Accuracy:** overall classification correctness
- **Precision:** reliability of positive predictions using macro averaging
- **Recall:** sensitivity of the classifier using macro averaging
- **F1-score:** harmonic balance between precision and recall using macro averaging
- **AUC:** ROC-AUC based on the predicted class probabilities

For binary classification, the positive-class probability is used for ROC-AUC. For multi-class problems, macro one-vs-rest AUC is used.

## Statistical Reliability Analysis

To provide a more complete analysis than a single train/test result, the code calculates:

- Mean performance across repeated runs
- Standard deviation
- 95% confidence interval based on the t-distribution
- Paired Wilcoxon signed-rank test between the stacked ensemble and each base learner
- Cohen's d effect size for the ensemble versus each base learner

The statistical analysis is performed for:

- Accuracy
- Precision
- Recall
- F1
- AUC

## Ablation Analysis

The implementation reports the standalone performance of:

- MLP
- CNN
- LSTM
- The best tuned classical model

and compares them with:

- Stacked-Ensemble

This allows the contribution of the heterogeneous stacking stage to be examined quantitatively.

## Output Files

After execution, the selected output directory contains numerical summaries and figures.

### Main Tables and Data Files

```text
summary_table.csv
ensemble_per_run_raw.csv
hpo_trials_<MODEL>.csv
hpo_trials_MLP.csv
hpo_trials_CNN.csv
hpo_trials_LSTM.csv
hpo_trials_META_LEARNER.csv
best_hyperparameters.json
```

### Figures

All figures are saved in:

```text
<output_dir>/figures/
```

The pipeline generates PNG and PDF versions of the figures, including:

```text
hpo_trials_<MODEL>.png / .pdf
hpo_trials_MLP.png / .pdf
hpo_trials_CNN.png / .pdf
hpo_trials_LSTM.png / .pdf
hpo_trials_META_LEARNER.png / .pdf

accloss_MLP.png / .pdf
accloss_CNN.png / .pdf
accloss_LSTM.png / .pdf

metric_Accuracy_line.png / .pdf
metric_Accuracy_bar.png / .pdf
metric_Accuracy_box.png / .pdf

metric_Precision_line.png / .pdf
metric_Precision_bar.png / .pdf
metric_Precision_box.png / .pdf

metric_Recall_line.png / .pdf
metric_Recall_bar.png / .pdf
metric_Recall_box.png / .pdf

metric_F1_line.png / .pdf
metric_F1_bar.png / .pdf
metric_F1_box.png / .pdf

metric_AUC_line.png / .pdf
metric_AUC_bar.png / .pdf
metric_AUC_box.png / .pdf

confusion_matrix.png / .pdf
roc_curve.png / .pdf
pr_curve.png / .pdf
summary_table.png / .pdf
significance_table.png / .pdf
effect_size_forest_plot.png / .pdf
```

## Usage Instructions

### 1. Install Python

Python 3.11 is recommended for the environment used with this project.

### 2. Install the required libraries

```bash
pip install -r requirements.txt
```

### 3. Place the dataset

Place the required CSV dataset in the expected location or provide its path explicitly.

### 4. Run the full experiment

Using the actual script filename in this repository:

```bash
python "Last-BO-HDSE .py" --csv ../heart.csv --target -1 --n_runs 30 --n_trials_classical 25 --n_trials_dl 15 --output_dir ./results/heart
```

### 5. Run a quick smoke test

```bash
python "Last-BO-HDSE .py" --csv ../heart.csv --quick
```

The `--quick` option reduces the number of runs, Optuna trials, meta-tuning folds, and training epochs so that the pipeline can be checked before launching the full experiment.

### Command-Line Arguments

The main configuration parameters are:

```text
--csv
--target
--output_dir
--n_runs
--n_splits_oof
--n_trials_classical
--n_trials_dl
--n_trials_meta
--n_splits_meta_tuning
--dl_epochs
--keep_duplicates
--categorical_threshold
--categorical_cols
--force_numeric_cols
--quick
--seed
```

## Reproducibility

The implementation initializes deterministic seeds for the main stochastic components:

- Python `random`
- NumPy
- TensorFlow

The default master seed is:

```text
42
```

The code also records the selected hyperparameters, dataset information, runtime, per-run ensemble results, and HPO trial tables in the output directory.

## Computational Notes

The full experimental configuration is computationally intensive because it combines:

- 30 repeated outer evaluations
- 5-fold OOF stacking
- 25 classical-model optimization trials
- 15 trials for each deep architecture
- 15 meta-learner optimization trials
- Up to 40 training epochs per deep-learning call

For this reason, the `--quick` option is recommended for verifying installation, file paths, and basic execution before running the complete experiment.

## Methodological Note

The script creates a tuning subset before the outer repeated evaluation. However, the current implementation subsequently performs the 30 outer train/test splits on the full dataset rather than restricting them to the remaining subset (`X_rest_raw`).

Therefore, the current code should **not** be described as a completely isolated, leakage-free nested evaluation design without first modifying this part of the implementation. This note is included so that the repository documentation accurately reflects the behavior of the provided code.

## Limitations

- The implementation expects a tabular CSV classification dataset.
- Dataset-specific clinical interpretation is not hard-coded into the pipeline.
- The current implementation does not provide a dedicated external validation cohort.
- The current tuning subset is not completely isolated from the later outer evaluation because the outer splits are generated from the full dataset.
- Computational cost can be substantial for the full 30-run configuration.
- No formal open-source license file is included in the current code package.

## Citation

If this repository is used in academic research, please cite the associated article:

**A Bayesian-Optimized Heterogeneous Deep Stacking Ensemble for Statistically Reliable Cardiovascular Disease Prediction**

Authors:

**Yahya Dorostkar Navaei, Yaser Ahangari Nanehkaran**

Please also cite the original sources of any cardiovascular datasets used in your experiments.

## License and Academic Use

This repository is intended primarily for academic and research use. Please contact the authors regarding redistribution or commercial use until a formal open-source license is added.

## Contribution Guidelines

Contributions are welcome for:

- Additional cardiovascular datasets
- Alternative deep architectures
- Improved stacking strategies
- More rigorous nested validation
- External validation experiments
- Computational optimization
- Additional statistical analyses

Any contribution should include clear documentation and sufficient experimental details to support reproducibility.
