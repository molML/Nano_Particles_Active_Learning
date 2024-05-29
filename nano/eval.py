"""
Evaluation functions.

Derek van Tilborg | 06-03-2023 | Eindhoven University of Technology

"""

import torch
from torch import Tensor
from typing import Union
from nano.models import XGBoostEnsemble, BayesianNN, RFEnsemble, GP
from nano.utils import augment_data
import numpy as np
from math import ceil
import pandas as pd


def evaluate_model(x: np.ndarray, y: np.ndarray, std: np.ndarray, id: np.ndarray, filename: str, hyperparameters: dict,
                   bootstrap: int = 10, n_folds: int = 5, ensemble_size=1, augment=5, model: str = 'bnn'):
    """ Function to evaluate model performance through bootstrapped k-fold cross-validation"""

    # estimate mean model performance over b bootstraps
    y_hats, bottom_5, upper_95, y_hats_uncertainty = [], [], [], []
    for b in range(bootstrap):
        # every "bootstrap" we have different CV splits
        y_hat, y_hat_mean, y_hat_uncertainty = k_fold_cross_validation(x, y, std, seed=b, n_folds=n_folds,
                                                                       ensemble_size=ensemble_size,
                                                                       augment=augment, model=model,
                                                                       **hyperparameters)

        # Calculate the 90% interval. We use 'ceil' to prevent errors with smaller ensemble sizes. Note that for
        # small ensemble sizes/ sampling frequencies, this 90% CI becomes meaningless
        y_hat_sorted = torch.sort(torch.tensor(y_hat), dim=-1)[0]
        bottom = y_hat_sorted.kthvalue(ceil(y_hat_sorted.shape[1] * 0.05), dim=1)[0]
        upper = y_hat_sorted.kthvalue(ceil(y_hat_sorted.shape[1] * 0.95), dim=1)[0]

        bottom_5.append(bottom)
        upper_95.append(upper)
        y_hats.append(y_hat_mean)
        y_hats_uncertainty.append(y_hat_uncertainty)

    # Take the mean over the bootstraps
    mean_y_hat = np.mean(y_hats, axis=0)
    mean_y_uncertainty = np.mean(y_hats_uncertainty, axis=0)

    mean_bottom_5 = torch.mean(torch.stack(bottom_5), 0)
    mean_upper_95 = torch.mean(torch.stack(upper_95), 0)

    # Put everything in a dataframe and save it somewhere
    df = pd.DataFrame({'ID': id,
                       'PLGA': x[:, 0],
                       'PP-L': x[:, 1],
                       'PP-COOH': x[:, 2],
                       'PP-NH2': x[:, 3],
                       'S/AS': x[:, 4],
                       'y': y,
                       'y_hat': mean_y_hat,
                       'CI_5%': mean_bottom_5,
                       'CI_95%': mean_upper_95,
                       'y_uncertainty': mean_y_uncertainty})
    df.to_csv(filename)

    # calculate the RMSE
    rmse = calc_rmse(y, mean_y_hat)

    return df, rmse


def k_fold_cross_validation(x: np.ndarray, y: np.ndarray, std: np.array, n_folds: int = 5,
                            ensemble_size: int = 1, seed: int = 42, augment: int = False, model: str = 'bnn',
                            sampling_freq: int = 500, **kwargs) -> (np.ndarray, np.ndarray, np.ndarray):
    assert len(x) == len(y), f"x and y should contain the same number of samples x:{len(x)}, y:{len(y)}"

    if model == 'gp':
        ensemble_size = 1
    # Define some variables
    y_hats = np.zeros((y.shape[0], ensemble_size if model != 'bnn' else sampling_freq))
    y_hats_mean, y_hats_uncertainty = np.zeros(y.shape), np.zeros(y.shape)
    # Set random state and create folds
    rng = np.random.default_rng(seed)
    folds = rng.integers(low=0, high=n_folds, size=len(x))

    for i in range(n_folds):
        # Subset train/test folds
        x_train, y_train, y_train_std = x[folds != i], y[folds != i], std[folds != i]
        x_test, y_test, y_test_std = x[folds == i], y[folds == i], std[folds == i]
        # Augment train data
        if augment:
            x_train, y_train = augment_data(x_train, y_train, y_train_std, n_times=augment, seed=seed)

        if model == 'xgb':
            m = XGBoostEnsemble(ensemble_size=ensemble_size, **kwargs)

            # Train and predict on the test split
            m.train(x_train, y_train)
            y_hat, y_hat_mean, y_hat_uncertainty = m.predict(x_test)
            y_hat = y_hat.T

        elif model == 'rf':
            m = RFEnsemble(ensemble_size=ensemble_size, **kwargs)

            # Train and predict on the test split
            m.train(x_train, y_train)
            y_hat, y_hat_mean, y_hat_uncertainty = m.predict(x_test)
            y_hat = y_hat.T

        elif model == 'gp':
            m = GP(**kwargs)
            # m = GP(**hypers)
            # Train and predict on the test split
            m.train(x_train, y_train)
            y_hat, y_hat_mean, y_hat_uncertainty = m.predict(x_test)
            y_hat = y_hat.reshape((y_hat.shape[0], 1))

        elif model == 'bnn':
            m = BayesianNN(**kwargs)

            # Train and predict on the test split
            m.train(x_train, y_train)
            y_hat, y_hat_mean, y_hat_uncertainty = m.predict(x_test, num_samples=sampling_freq)

            # Delete the NN to free memory
            try:
                del m.model
                del m.guide
                del m.svi
                del m
                torch.cuda.empty_cache()
            except:
                pass

        # Add the predicted test values to the y_hats tensor in the correct spot along with the uncertainty
        y_hats[folds == i] = y_hat
        y_hats_mean[folds == i] = y_hat_mean
        y_hats_uncertainty[folds == i] = y_hat_uncertainty

    return y_hats, y_hats_mean, y_hats_uncertainty


def calc_rmse(y: Union[np.ndarray, Tensor], y_hat: Union[np.ndarray, Tensor]) -> float:
    """ Calculates the Root Mean Square Error """
    if type(y) is Tensor:
        y = np.array(y)
    if type(y_hat) is Tensor:
        y_hat = np.array(y_hat)

    assert len(y) == len(y_hat), f"y and y_hat should contain the same number of samples y:{len(y)}, y_hat:{len(y_hat)}"

    return np.sqrt(np.mean(np.square(y - y_hat)))
