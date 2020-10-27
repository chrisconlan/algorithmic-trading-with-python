# See pypm.ml_model.model
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.base import clone

from joblib import Parallel, delayed

# Number of jobs to run in parallel
# Set to number of computer cores to use
N_JOBS = 10
N_SPLITS = 5
N_REPEATS = 4


def _fit_and_score(classifier, X, y, w, train_index, test_index, i) -> float:
    """
    The function used by joblib to split, train, and score cross validations
    """
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    w_train = w.iloc[train_index]
    w_test = w.iloc[test_index]

    classifier.fit(X_train, y_train, w_train)
    score = classifier.score(X_test, y_test, w_test)

    print(f"Finished {i} ({100*score:.1f}%)")

    return score


def repeated_k_fold(classifier, X, y, w) -> np.ndarray:
    """
    Perform repeated k-fold cross validation on a classifier. Spread fitting
    job over multiple computer cores.
    """
    n_jobs = N_JOBS

    n_splits = N_SPLITS
    n_repeats = N_REPEATS

    total_fits = n_splits * n_repeats

    _k_fold = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)

    print(f"Fitting {total_fits} models {n_jobs} at a time ...")
    print()

    parallel = Parallel(n_jobs=n_jobs)
    scores = parallel(
        delayed(_fit_and_score)(clone(classifier), X, y, w, train_index,
                                test_index, i)
        for i, (train_index, test_index) in enumerate(_k_fold.split(X)))

    return np.array(scores)


def calculate_model(df: pd.DataFrame) -> RandomForestClassifier:
    """
    Given a dataframe with a y column, weights column, and predictor columns
    with arbitrary names, cross-validated and fit a classifier. Print
    diagnostics.
    """
    classifier = RandomForestClassifier(n_estimators=100)

    # Separate data
    predictor_columns = [
        c for c in df.columns.values if c not in ("y", "weights")
    ]
    X = df[predictor_columns]
    y = df["y"]
    w = df["weights"]

    # Fit cross validation
    scores = repeated_k_fold(classifier, X, y, w)

    # Get a full dataset fit for importance scores
    classifier.fit(X, y, w)

    # Compute diagnostics
    _imp = classifier.feature_importances_
    importance_series = pd.Series(_imp, index=predictor_columns)
    importance_series = importance_series.sort_values(ascending=False)

    # baseline accuracy is the best value achievable with a constant guess
    baseline = np.max(y.value_counts() / y.shape[0])

    # Compute a rough confidence interval for the improvement
    mean_score = scores.mean()
    std_score = scores.std()

    upper_bound = mean_score + 2 * std_score
    lower_bound = mean_score - 2 * std_score
    ibounds = (lower_bound - baseline, upper_bound - baseline)

    print("Feature importances")
    for col, imp in list(importance_series.items()):
        print(f"{col:24} {imp:>.3f}")
    print()

    print("Cross validation scores")
    print((np.round(100 * scores, 1)))
    print()

    print(f"Baseline accuracy {100*baseline:.1f}%")
    print(f"OOS accuracy {100*mean_score:.1f}% +/- {200 * scores.std():.1f}%")
    print(f"Improvement {100*(ibounds[0]):.1f} to {100*(ibounds[1]):.1f}%")
    print()

    return classifier
