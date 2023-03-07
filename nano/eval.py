"""
Evaluation functions.

Derek van Tilborg | 06-03-2023 | Eindhoven University of Technology

"""

import numpy as np
from nano.models import XGBoostEnsemble
from nano.utils import augment_data


def k_fold_cross_validation(x: np.ndarray, y: np.ndarray, n_folds: int = 10, ensemble_size: int = 10, seed: int = 42,
                            augment: int = False, **kwargs) -> (np.ndarray, np.ndarray):
    assert len(x) == len(y), f"x and y should contain the same number of samples x:{len(x)}, y:{len(y)}"

    y_hats, y_hats_uncertainty = np.zeros(y.shape), np.zeros(y.shape)

    rng = np.random.default_rng(seed)
    folds = rng.integers(low=0, high=n_folds, size=len(x))

    for i in range(n_folds):
        x_train, y_train = x[folds != i], y[folds != i]

        if augment:
            x_train, y_train = augment_data(x_train, y_train, n_times=augment, seed=seed)

        x_val, y_val = x[folds == i], y[folds == i]

        ensmbl = XGBoostEnsemble(ensemble_size=ensemble_size, **kwargs)
        ensmbl.train(x_train, y_train)

        y_hat, y_hat_mean, y_hat_uncertainty = ensmbl.predict(x_val)

        y_hats[folds == i] = y_hat_mean
        y_hats_uncertainty[folds == i] = y_hat_uncertainty

    return y_hats, y_hats_uncertainty


def calc_rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """ Calculates the Root Mean Square Error """
    assert type(y) is np.ndarray and type(y_hat) is np.ndarray, 'y and y_hat should be Numpy Arrays'
    assert len(y) == len(y_hat), f"y and y_hat should contain the same number of samples y:{len(y)}, y_hat:{len(y_hat)}"

    return np.sqrt(np.mean(np.square(y - y_hat)))
