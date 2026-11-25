"""History-supervised LDA, evaluated on held-out cognitive-diagram states."""
import warnings

import numpy as np

from src.decoding import (InsufficientSamplesError, balanced_indices,
                          fit_projection, pairwise_test_accuracy, state_samples)


def history_samples(hidden: np.ndarray, inputs: np.ndarray,
                    k: int) -> tuple[np.ndarray, np.ndarray]:
    """Pair post-input h_t with complete ordered windows ending at x_t."""
    if hidden.ndim != 3 or inputs.ndim != 2 or hidden.shape[:2] != inputs.shape:
        raise ValueError('Expected aligned hidden (N, T, D) and input IDs (N, T).')
    if not 1 <= k <= inputs.shape[1]:
        raise ValueError(f'k={k} requires at least {k} retained timesteps.')
    windows = np.lib.stride_tricks.sliding_window_view(inputs, k, axis=1)
    _, labels = np.unique(windows.reshape(-1, k), axis=0, return_inverse=True)
    return hidden[:, k - 1:, :].reshape(-1, hidden.shape[-1]), labels


def history_decoding_scores(hidden_train: np.ndarray, inputs_train: np.ndarray,
                            truths_train: np.ndarray, hidden_test: np.ndarray,
                            inputs_test: np.ndarray, truths_test: np.ndarray,
                            seed: int, max_k: int = 3
                            ) -> list[dict[str, int | float | str]]:
    """Fit LDA on training histories; decode original states on held-out sequences.

    Truths are aligned integer diagram-state IDs, not history labels. History
    balancing is used only to fit the projection. State balancing is performed
    separately for the pairwise classifiers, after transforming both splits.
    """
    if not 1 <= max_k <= 3:
        raise ValueError('max_k must be between 1 and 3.')
    for hidden, truths in ((hidden_train, truths_train), (hidden_test, truths_test)):
        if truths.shape != hidden.shape[:2]:
            raise ValueError('Expected diagram-state IDs aligned with hidden states.')
    expected_states = np.union1d(truths_train, truths_test)
    rows = []
    for k in range(1, max_k + 1):
        X_train, history_train = history_samples(hidden_train, inputs_train, k)
        X_test, _ = history_samples(hidden_test, inputs_test, k)
        y_train = truths_train[:, k - 1:].reshape(-1)
        y_test = truths_test[:, k - 1:].reshape(-1)
        classes, counts = np.unique(history_train, return_counts=True)
        min_count = int(counts.min()) if len(counts) else 0
        row = dict(k=k, n_history_classes=len(classes),
                   history_train_samples_per_class=min_count,
                   n_projection_train_samples=len(classes) * min_count,
                   n_components=min(2, max(0, len(classes) - 1), X_train.shape[1]))
        try:
            fit_indices = balanced_indices(history_train, np.random.default_rng(seed))
            projection = fit_projection(
                'LDA', X_train[fit_indices], history_train[fit_indices])
            train, y_train, test, y_test = state_samples(
                projection.transform(X_train), y_train,
                projection.transform(X_test), y_test, seed=seed,
                expected_states=expected_states, context=f'k={k}:')
        except InsufficientSamplesError as error:
            warnings.warn(f'k={k}: state decoding unavailable: {error}',
                          RuntimeWarning, stacklevel=2)
            rows.append(dict(**row, accuracy=float('nan'),
                             n_state_classes=0, n_train_samples=0, n_test_samples=0,
                             status='insufficient_samples', reason=str(error)))
            continue
        accuracy = pairwise_test_accuracy(train, y_train, test, y_test)
        row['n_components'] = train.shape[1]
        rows.append(dict(**row, accuracy=accuracy,
                         n_state_classes=len(np.unique(y_test)),
                         n_train_samples=len(y_train), n_test_samples=len(y_test),
                         status='ok', reason=''))
    return rows
