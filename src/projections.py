"""Dimensionality-reduction projections and decoding analyses of hidden states.

Moved verbatim from src/models.py (which re-exports these names for backward
compatibility). Computation only -- no plotting here.
"""
from itertools import product

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import FastICA, FactorAnalysis
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def train_lda_from_tensors(X_tensor: torch.Tensor, Y_labels: torch.Tensor,
                           return_preds=False):
    """Fit an LDA classifier on (N, D) activations; returns overall accuracy."""
    X = X_tensor.detach().cpu().numpy()
    if isinstance(Y_labels, torch.Tensor):
        Y = Y_labels.detach().cpu().numpy()
    else:
        Y = Y_labels
    if len(Y.shape) > 1:
        raise NotImplementedError("multi-dimensional targets are not supported")
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(X, Y)
    Y_pred = lda.predict(X)
    report = classification_report(Y, Y_pred, output_dict=True, zero_division=0)
    if return_preds:
        return report['accuracy'], None, None, (Y_pred, Y)
    return report['accuracy'], None, None


def get_pairwise_decodings(predictor, target):
    """LDA decoding accuracy for every ordered pair of target states.

    Returns a (n_states, n_states) matrix with 1.0 on the diagonal and the
    pairwise LDA training accuracy off-diagonal.
    """
    unique_targets = target.unique().tolist()
    all_pairs = list(product(unique_targets, repeat=2))
    pairwise_accs = np.ones((len(unique_targets), len(unique_targets)))
    for pair in all_pairs:
        if pair[0] == pair[1]:
            continue
        indices_0 = [idx for idx, x in enumerate(target) if x == pair[0]]
        indices_1 = [idx for idx, x in enumerate(target) if x == pair[1]]
        pred = torch.concat([predictor[indices_0, :], predictor[indices_1, :]], dim=0)
        labels = torch.tensor([pair[0]] * len(indices_0) + [pair[1]] * len(indices_1),
                              requires_grad=False)
        acc, _, _ = train_lda_from_tensors(pred, labels)
        pairwise_accs[pair[0], pair[1]] = acc
    return pairwise_accs


def get_lda_projection(X, labels, d=2):
    """Supervised LDA projection of (N, D) activations to d dimensions."""
    X = X.detach().cpu().numpy()
    lda = LinearDiscriminantAnalysis(n_components=d, solver='svd')
    return lda.fit_transform(X, labels)


def get_pca_projection(X):
    """Unsupervised 2D PCA projection (torch implementation)."""
    N, D = X.shape
    X_centered = X - X.mean(0, keepdim=True)
    cov = X_centered.T @ X_centered / (N - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    idx = torch.argsort(eigvals, descending=True)[:2]
    W = eigvecs[:, idx]
    return X_centered @ W


def get_ica_projection(X):
    Xz = StandardScaler().fit_transform(X)
    return FastICA(n_components=2, random_state=0, max_iter=2000).fit_transform(Xz)


def get_fa_projection(X):
    Xz = StandardScaler().fit_transform(X)
    return FactorAnalysis(n_components=2, random_state=0).fit_transform(Xz)


def dimensionality_test(X_raw, y_cat, C=1e4, tol=1e-3):
    """Mean pairwise linear separability of classes in a 2D embedding.

    For every unordered class pair, fits a fast Perceptron check and then a
    LinearSVC, and reports the mean pairwise accuracy.
    """
    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw)

    classes = np.unique(y_cat)
    pairs = [(classes[i], classes[j])
             for i in range(len(classes)) for j in range(i + 1, len(classes))]

    out = {}
    for p0, p1 in pairs:
        mask = (y_cat == p0) | (y_cat == p1)
        Xa = X[mask]
        ya = np.where(y_cat[mask] == p0, 1, -1)

        # fast feasibility check: a converged perceptron means separable
        p = Perceptron(max_iter=200, tol=1e-3, fit_intercept=True)
        p.fit(Xa, ya)
        if (p.predict(Xa) == ya).mean() == 1.0:
            clf = LinearSVC(C=C, tol=tol, max_iter=2000, dual=False)
            clf.fit(Xa, ya)
            w = clf.coef_[0]; b = clf.intercept_[0]
            margin = 1.0 / np.linalg.norm(w)
            signed = ya * (Xa @ w + b) / np.linalg.norm(w)
            n_sv_approx = int((np.abs(signed - 1.0) < 1e-2).sum())
            out[(p0, p1)] = dict(separable=True, margin=float(margin),
                                 accuracy=1.0, n_support_approx=n_sv_approx)
            continue

        clf = LinearSVC(C=C, tol=tol, max_iter=5000, dual=False)
        clf.fit(Xa, ya)
        w = clf.coef_[0]; b = clf.intercept_[0]
        normw = np.linalg.norm(w)
        pred = np.sign(Xa @ w + b)
        acc = (pred == ya).mean()
        signed = ya * (Xa @ w + b) / (normw + 1e-12)
        out[(p0, p1)] = dict(separable=acc == 1.0, margin=float(1.0 / (normw + 1e-12)),
                             accuracy=float(acc), min_signed_distance=float(signed.min()))

    return np.mean([out[pair]['accuracy'] for pair in pairs])
