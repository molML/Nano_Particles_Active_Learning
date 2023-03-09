"""

A collection of support functions

Derek van Tilborg | 06-03-2023 | Eindhoven University of Technology

"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor
from sklearn.utils import shuffle as sklearn_shuffle


def load_data(cycle: int = 0, seed: int = 42, shuffle: bool = True) -> (np.ndarray, np.ndarray, np.ndarray,
                                                                       np.ndarray, np.ndarray, np.ndarray):
    """ Loads the data from our pre-processed files, with the option to shuffle the samples """

    # Load the uptake data
    data = pd.read_csv(f'data/cycle_{cycle}/Uptake_data_cycle_{cycle}.csv')
    if shuffle:
        data = sklearn_shuffle(data, random_state=seed)

    id = np.array(data['ID'])
    x = np.array([data['PLGA'], data['PP-L'], data['PP-COOH'], data['PP-NH2'], data['S/AS']]).T
    uptake_y = np.array(data['Uptake'])
    pdi_y = np.array(data['PDI'])

    # Load the screen data
    screen = pd.read_csv('data/screen_library.csv')
    if shuffle:
        screen = sklearn_shuffle(screen, random_state=seed)

    screen_id = np.array(screen['ID'])
    screen_x = np.array([screen['PLGA'], screen['PP-L'], screen['PP-COOH'], screen['PP-NH2'], screen['S/AS']]).T

    return x, uptake_y, pdi_y, id, screen_x, screen_id


def augment_data(x: np.ndarray, y: np.ndarray, n_times: int = 5, shuffle: bool = True, seed: int = 42,
                 verbose: bool = False) -> (np.ndarray, np.ndarray):

    experimental_error = {'PLGA': 1.2490, 'PP-L': 1.2121, 'PP-COOH': 1.2359,  'PP-NH2': 1.2398,  'S/AS': 0}

    rng = np.random.default_rng(seed)
    n = len(x)

    assert len(x) == len(y), f"x and y should contain the same number of samples x:{len(x)}, y:{len(y)}"
    if verbose:
        print(f"Augmenting {n_times} times; {n} original values + {n*(n_times-1)} ({n}x{(n_times-1)}) augmented values")

    # TODO I currently do not augment the labels.
    x_prime, y_prime = x, y
    for i in range(n_times-1):
        augmentation = np.vstack([rng.normal(100, experimental_error['PLGA'], n) / 100,
                                  rng.normal(100, experimental_error['PP-L'], n) / 100,
                                  rng.normal(100, experimental_error['PP-COOH'], n) / 100,
                                  rng.normal(100, experimental_error['PP-NH2'], n) / 100,
                                  rng.normal(100, experimental_error['S/AS'], n) / 100]).T

        # multiply the data with the augmentation matrix
        x_prime = np.vstack((x_prime, x * augmentation))
        y_prime = np.append(y_prime, y)

    if shuffle:
        shuffling = rng.permutation(len(x_prime))
        x_prime = x_prime[shuffling]
        y_prime = y_prime[shuffling]

    return x_prime, y_prime


def screen_predict(screen_x, screen_id, uptake_model, pdi_model, filename: str):
    # Perform predictions
    y_hat_uptake, y_hat_mean_uptake, y_hat_uncertainty_uptake = uptake_model.predict(screen_x)
    y_hat_pdi, y_hat_mean_pdi, y_hat_uncertainty_pdi = pdi_model.predict(screen_x)

    screen_df = pd.DataFrame({'ID': screen_id,
                              'y_hat_uptake': y_hat_mean_uptake,
                              'y_uncertainty_uptake': y_hat_uncertainty_uptake,
                              'y_hat_pdi': y_hat_mean_pdi,
                              'y_uncertainty_pdi': y_hat_uncertainty_pdi,
                              'x_PLGA': screen_x[:, 0],
                              'x_PP-L': screen_x[:, 1],
                              'x_PP-COOH': screen_x[:, 2],
                              'x_PP-NH2': screen_x[:, 3],
                              'x_S/AS': screen_x[:, 4]})

    screen_df.to_csv(filename, index=False)

    return screen_df


def numpy_to_dataloader(x: np.ndarray, y: np.ndarray = None, **kwargs) -> DataLoader:

    if y is None:
        return DataLoader(TensorDataset(Tensor(x)),  **kwargs)
    else:
        return DataLoader(TensorDataset(Tensor(x), Tensor(y)),  **kwargs)
