#!/usr/bin/env python3
"""
Train and evaluate a binary logistic regression model for entanglement classification
with two possible scenarios:

1) **Within-Species (Same Data)**:
   - Perform 5-fold cross validation on a single (features, targets) dataset.
   - Automatically splits data into train/test folds, computes performance metrics,
     and reports the mean ± std across folds.

2) **Cross-Species (Different Data)**:
   - Use 5-fold cross validation on the "train" dataset to get 5 separate models.
   - For each fold, fit on the training split of that fold, then apply to the entire "test" dataset.
   - Average the metrics (accuracy, balanced accuracy, precision, recall, f1,
     average precision, roc_auc) over the 5 models.

Supports using either the full feature set or a predefined "reduced" set of columns.

Usage Examples:
-----------------
1) Same data for training/testing (within-species):
   python entanglement_model.py \
       --train-features features.pkl \
       --train-targets targets.pkl \
       --test-features features.pkl \
       --test-targets targets.pkl \
       --feature-set all \
       --output-file results.txt

2) Different data for training vs. testing (cross-species):
   python entanglement_model.py \
       --train-features train_features.pkl \
       --train-targets train_targets.pkl \
       --test-features test_features.pkl \
       --test-targets test_targets.pkl \
       --feature-set reduced \
       --output-file cross_results.txt

Arguments
---------
--train-features : Path to a pickled Pandas DataFrame for training features.
--train-targets  : Path to a pickled Pandas Series/DataFrame for training targets.
--test-features  : Path to a pickled Pandas DataFrame for testing features.
--test-targets   : Path to a pickled Pandas Series/DataFrame for testing targets.
--feature-set    : "all" or "reduced". If "reduced", only a predefined subset of columns is used.
--output-file    : Path to which we append the performance results.

Behavior
--------
- Always uses 5-fold cross validation to create train/test splits from the training data.
- If test-features/test-targets are identical to the training data, does standard within-dataset 5-fold CV.
- Otherwise, still does 5-fold CV on the "train" data to get 5 estimators, then applies each fold's model to the entire "test" data.
- Averages the performance metrics (accuracy, balanced_accuracy, average_precision, f1, precision, recall, roc_auc)
  across the 5 folds in whichever scenario is used.
- The logistic regression model is:
    LogisticRegression(
        class_weight='balanced',
        penalty='none',
        max_iter=1000,
        verbose=0,
        n_jobs=1
    )
- Results are appended to --output-file, including mean ± std for each metric.
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, balanced_accuracy_score, average_precision_score,
    f1_score, precision_score, recall_score, roc_auc_score, make_scorer
)
from joblib import parallel_backend

##################
# Custom confusion-matrix based scorers
##################
def _true_negative(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[0][0]

def _true_positive(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[1][1]

def _false_negative(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[1][0]

def _false_positive(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[0][1]

##################
# Dictionary of scoring functions for cross_validate
##################
SCORING = {
    'accuracy':           'accuracy',
    'balanced_accuracy':  'balanced_accuracy',
    'average_precision':  'average_precision',
    'f1':                 'f1',
    'precision':          'precision',
    'recall':             'recall',
    'roc_auc':            'roc_auc',
    'true_positive':      make_scorer(_true_positive),
    'true_negative':      make_scorer(_true_negative),
    'false_positive':     make_scorer(_false_positive),
    'false_negative':     make_scorer(_false_negative)
}

##################
# Example "reduced" feature set
##################
REDUCED_FEATURES = [
    'D_0_pssm',
    'ACH3_pssm',
    'CN_exp',
    'Theta_exp',
    'Tau_exp',
    'psi',
    'rsaa',
    'SS7__Strand',
    'G_0_pssm'
]

def main():
    parser = argparse.ArgumentParser(
        description="Train & evaluate logistic regression for entanglement classification."
    )
    parser.add_argument("--train-features", required=True,
                        help="Pickled Pandas DataFrame for training features.")
    parser.add_argument("--train-targets", required=True,
                        help="Pickled Pandas Series/DataFrame for training targets.")
    parser.add_argument("--test-features", required=True,
                        help="Pickled Pandas DataFrame for testing features.")
    parser.add_argument("--test-targets", required=True,
                        help="Pickled Pandas Series/DataFrame for testing targets.")
    parser.add_argument("--feature-set", choices=["all","reduced"], default="all",
                        help="Use all features or only a predefined reduced subset.")
    parser.add_argument("--output-file", default="model_performance.txt",
                        help="File to append performance metrics.")
    args = parser.parse_args()

    ##################
    # Load data
    ##################
    train_features_df = pd.read_pickle(args.train_features)
    train_targets_df  = pd.read_pickle(args.train_targets)
    test_features_df  = pd.read_pickle(args.test_features)
    test_targets_df   = pd.read_pickle(args.test_targets)

    ##################
    # Subset columns if "reduced"
    ##################
    if args.feature_set == "reduced":
        keep_train = [c for c in REDUCED_FEATURES if c in train_features_df.columns]
        train_features_df = train_features_df[keep_train]

        keep_test = [c for c in REDUCED_FEATURES if c in test_features_df.columns]
        test_features_df = test_features_df[keep_test]

    ##################
    # Standard scale each dataset
    ##################
    scaler_train = StandardScaler()
    X_train = scaler_train.fit_transform(train_features_df)
    y_train = train_targets_df.values.ravel() if hasattr(train_targets_df, "values") else np.array(train_targets_df)

    scaler_test = StandardScaler()
    X_test = scaler_test.fit_transform(test_features_df)
    y_test = test_targets_df.values.ravel() if hasattr(test_targets_df, "values") else np.array(test_targets_df)

    ##################
    # Check if train and test are the same file -> within-species
    ##################
    same_data = (
        os.path.abspath(args.train_features) == os.path.abspath(args.test_features) and
        os.path.abspath(args.train_targets)  == os.path.abspath(args.test_targets)
    )

    ##################
    # Logistic Regression model
    ##################
    regressor = LogisticRegression(
        class_weight='balanced',
        penalty='none',
        max_iter=1000,
        verbose=0,
        n_jobs=1
    )

    with open(args.output_file, "a") as out_f:
        out_f.write("=======================================\n")
        out_f.write(f"Run started at: {time.ctime()}\n")
        out_f.write(f"Feature set: {args.feature_set}\n")
        out_f.write(f"Train features: {args.train_features}\n")
        out_f.write(f"Train targets:  {args.train_targets}\n")
        out_f.write(f"Test features:  {args.test_features}\n")
        out_f.write(f"Test targets:   {args.test_targets}\n\n")

        ##################
        # Within-species scenario
        ##################
        if same_data:
            out_f.write("Scenario: Single dataset => 5-fold cross validation.\n")
            start_t = time.time()

            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            with parallel_backend('multiprocessing', n_jobs=5):
                scores = cross_validate(regressor, X_train, y_train,
                                        cv=kf, scoring=SCORING, return_estimator=False)
            end_t = time.time()

            out_f.write(f"CV runtime: {end_t - start_t:.2f} sec\n")
            # Summarize means and std
            result_df = pd.DataFrame(scores)
            summary_df = result_df.describe().loc[['mean','std']].round(4)
            out_f.write("5-Fold CV Performance (mean ± std):\n")
            out_f.write(str(summary_df) + "\n\n")

        ##################
        # Cross-species scenario
        ##################
        else:
            out_f.write("Scenario: Cross-species => 5-fold CV on training, apply each fold's model to entire test set.\n")
            start_t = time.time()

            # We want each estimator, so use return_estimator=True
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            with parallel_backend('multiprocessing', n_jobs=5):
                scores = cross_validate(regressor, X_train, y_train,
                                        cv=kf, scoring=SCORING, return_estimator=True)
            end_t = time.time()
            out_f.write(f"Train CV runtime: {end_t - start_t:.2f} sec\n")

            # Evaluate each fold's model on X_test
            metrics_list = []
            for model in scores['estimator']:
                y_pred  = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

                row_dict = {}
                row_dict['accuracy']           = accuracy_score(y_test, y_pred)
                row_dict['balanced_accuracy']  = balanced_accuracy_score(y_test, y_pred)
                row_dict['average_precision']  = average_precision_score(y_test, y_proba)
                row_dict['f1']                = f1_score(y_test, y_pred)
                row_dict['precision']         = precision_score(y_test, y_pred)
                row_dict['recall']            = recall_score(y_test, y_pred)
                row_dict['roc_auc']           = roc_auc_score(y_test, y_proba)
                metrics_list.append(row_dict)

            cross_df = pd.DataFrame(metrics_list)
            summary_df = cross_df.describe().loc[['mean','std']].round(4)
            out_f.write("Test performance over 5 folds (mean ± std):\n")
            out_f.write(str(summary_df) + "\n\n")

        out_f.write("Done.\n\n")


if __name__ == "__main__":
    main()

