"""Train-only projections and held-out pairwise diagram-state decoding."""
from itertools import combinations

import numpy as np
from sklearn.decomposition import FastICA, FactorAnalysis, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


class InsufficientSamplesError(ValueError):
    """The requested projection or state-pair comparison cannot be estimated."""


def balanced_indices(labels, rng):
    """Downsample each observed class to the rarest class."""
    classes, counts = np.unique(labels, return_counts=True)
    if not len(classes):
        raise InsufficientSamplesError('No samples remain.')
    return np.concatenate([
        rng.choice(np.flatnonzero(labels == label), size=int(counts.min()),
                   replace=False)
        for label in classes
    ])


def fit_projection(method, predictors, labels, n_components=2):
    """Fit a projection (including any preprocessing) using training data only."""
    if method == 'LDA':
        classes, counts = np.unique(labels, return_counts=True)
        if len(classes) < 2 or counts.min() < 2:
            raise InsufficientSamplesError(
                'LDA requires at least two classes and two training samples '
                f'per class after balancing; found {len(classes)} classes '
                f'and minimum count {counts.min() if len(counts) else 0}.')
        dimensions = min(n_components, len(classes) - 1, predictors.shape[1])
        projection = LinearDiscriminantAnalysis(n_components=dimensions, solver='svd')
    elif method == 'PCA':
        projection = PCA(n_components=n_components, svd_solver='full')
    elif method == 'ICA':
        projection = make_pipeline(
            StandardScaler(),
            FastICA(n_components=n_components, random_state=0, max_iter=2000))
    elif method == 'FA':
        projection = make_pipeline(
            StandardScaler(), FactorAnalysis(n_components=n_components, random_state=0))
    else:
        raise ValueError(f'Unknown projection method {method!r}.')
    projection.fit(predictors, labels)
    return projection


def state_samples(X_train, y_train, X_test, y_test, seed=0,
                  expected_states=None, keep_states=None, context=''):
    """Balance each split; test availability must not change training samples."""
    train_states, test_states = np.unique(y_train), np.unique(y_test)
    states = np.intersect1d(train_states, test_states)
    expected = (np.union1d(train_states, test_states) if expected_states is None
                else np.asarray(expected_states))
    if keep_states is not None:
        train_states = np.intersect1d(train_states, list(keep_states))
        states = np.intersect1d(states, list(keep_states))
        expected = np.intersect1d(expected, list(keep_states))
    excluded = np.setdiff1d(expected, states)
    if len(excluded):
        print(f'{context} States absent from train or test samples; '
              f'excluding IDs {excluded.tolist()} from pairwise decoding.')
    if len(states) < 2:
        raise InsufficientSamplesError(
            f'{context} Need at least two diagram states represented in both splits.')
    rng = np.random.default_rng(seed)
    result = []
    for X, y, included in ((X_train, y_train, train_states), (X_test, y_test, states)):
        mask = np.isin(y, included)
        X, y = X[mask], y[mask]
        indices = balanced_indices(y, rng)
        result.extend((X[indices], y[indices]))
    return tuple(result)


def pairwise_test_accuracy(X_train, y_train, X_test, y_test, C=1e4, tol=1e-3):
    """Mean held-out accuracy over all unordered pairs of diagram-state labels."""
    train_classes, test_classes = np.unique(y_train), np.unique(y_test)
    if len(test_classes) < 2 or not np.isin(test_classes, train_classes).all():
        raise InsufficientSamplesError(
            'Pairwise decoding needs two or more test states, all represented in training.')
    scaler = StandardScaler().fit(X_train)
    train, test = scaler.transform(X_train), scaler.transform(X_test)
    accuracies = []
    for first, second in combinations(test_classes, 2):
        train_mask = (y_train == first) | (y_train == second)
        test_mask = (y_test == first) | (y_test == second)
        classifier = LinearSVC(C=C, tol=tol, max_iter=5000, dual=False)
        classifier.fit(train[train_mask], y_train[train_mask])
        accuracies.append(classifier.score(test[test_mask], y_test[test_mask]))
    return float(np.mean(accuracies))


def state_projection_analysis(X_train, y_train, X_test, y_test, seed=0,
                              keep_states=None, expected_states=None,
                              score=True, shuffled=False):
    """Fit each method on train states; transform and optionally score test states."""
    X_train, y_train, X_test, y_test = state_samples(
        X_train, y_train, X_test, y_test, seed=seed, keep_states=keep_states,
        expected_states=expected_states)
    if shuffled:
        rng = np.random.default_rng(seed)
        y_train, y_test = rng.permutation(y_train), rng.permutation(y_test)
    projections, accuracies = {}, {} if score else None
    for method in (('LDA',) if shuffled else ('LDA', 'PCA', 'ICA', 'FA')):
        projection = fit_projection(method, X_train, y_train)
        train_projection = projection.transform(X_train)
        test_projection = projection.transform(X_test)
        projections[method] = test_projection
        if score:
            accuracies[method] = pairwise_test_accuracy(
                train_projection, y_train, test_projection, y_test)
    return projections, accuracies, y_test
