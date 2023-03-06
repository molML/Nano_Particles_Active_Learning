"""

A collection of support functions

Derek van Tilborg | 06-03-2023 | Eindhoven University of Technology

"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np


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
