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
    uptake_data = pd.read_csv(f'data/cycle_{cycle}/Uptake_data_cycle_{cycle}.csv')
    if shuffle:
        uptake_data = sklearn_shuffle(uptake_data, random_state=seed)

    uptake_id = np.array(uptake_data['ID'])
    uptake_x = np.array([uptake_data['PLGA'],
                         uptake_data['PP-L'],
                         uptake_data['PP-COOH'],
                         uptake_data['PP-NH2'],
                         uptake_data['S/AS']]).T
    uptake_y = np.array(uptake_data['Uptake'])

    # Load the PDI data
    pdi_data = pd.read_csv(f'data/cycle_{cycle}/PDI_data_cycle_{cycle}.csv')
    if shuffle:
        pdi_data = sklearn_shuffle(pdi_data, random_state=seed)

    pdi_id = np.array(pdi_data['ID'])
    pdi_x = np.array([pdi_data['PLGA'],
                      pdi_data['PP-L'],
                      pdi_data['PP-COOH'],
                      pdi_data['PP-NH2'],
                      pdi_data['S/AS']]).T
    pdi_y = np.array(pdi_data['PDI'])

    # Load the screen data
    screen_data = pd.read_csv('data/screen_library.csv')
    if shuffle:
        screen_data = sklearn_shuffle(screen_data, random_state=seed)

    screen_id = np.array(screen_data['ID'])
    screen_x = np.array([screen_data['PLGA'],
                         screen_data['PP-L'],
                         screen_data['PP-COOH'],
                         screen_data['PP-NH2'],
                         screen_data['S/AS']]).T

    return uptake_x, uptake_y, uptake_id, pdi_x, pdi_y, pdi_id, screen_x, screen_id


def augment_data(x: np.ndarray, y: np.ndarray, n_times: int = 10, shuffle: bool = True, seed: int = 42,
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