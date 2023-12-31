"""

A collection of support functions.
    - load_data()
    - augment_data()
    - screen_predict()
    - numpy_to_dataloader()

Derek van Tilborg | 06-03-2023 | Eindhoven University of Technology

"""

import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor
from sklearn.utils import shuffle as sklearn_shuffle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_data(cycle: int = 0, set: str = 'uptake', seed: int = 42, shuffle: bool = True, omit_unstable: bool = False,
              suffix: str = '') -> \
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """ Loads the data from our pre-processed files, with the option to shuffle the samples """

    assert set in ['uptake', 'pdi', 'size', 'screen'], "'set' should be 'uptake', 'pdi', 'size', or 'screen'"

    # Load the uptake data
    data = pd.read_csv(f'data/cycle_{cycle}/Uptake_data_cycle_{cycle}{suffix}.csv')
    if omit_unstable:
        data = data.drop(data[data['PDI'] > 0.2].index)
    if shuffle:
        data = sklearn_shuffle(data, random_state=seed)

    id = np.array(data['ID'])
    x = np.array([data['PLGA'], data['PP-L'], data['PP-COOH'], data['PP-NH2'], data['S/AS']]).T

    if set == 'uptake':
        uptake_y = np.array(data['Uptake'])
        uptake_std = np.array(data['Uptake_stdev'])
        nan_filter = np.invert(np.isnan(uptake_y))

        return x[nan_filter], uptake_y[nan_filter], uptake_std[nan_filter], id[nan_filter]

    elif set == 'pdi':
        pdi_y = np.array(data['PDI'])
        pdi_std = np.array(data['PDI_stdev'])
        nan_filter = np.invert(np.isnan(pdi_y))

        return x[nan_filter], pdi_y[nan_filter], pdi_std[nan_filter], id[nan_filter]

    elif set == 'size':
        size_y = np.array(data['Z_ave'])
        size_std = np.array(data['Z_ave_stdev'])
        nan_filter = np.invert(np.isnan(size_y))

        return x[nan_filter], size_y[nan_filter], size_std[nan_filter], id[nan_filter]

    elif set == 'screen':
        # Load the screen data
        screen = pd.read_csv('data/screen_library.csv')
        if shuffle:
            screen = sklearn_shuffle(screen, random_state=seed)

        screen_id = np.array(screen['ID'])
        screen_x = np.array([screen['PLGA'], screen['PP-L'], screen['PP-COOH'], screen['PP-NH2'], screen['S/AS']]).T

        return screen_x, screen_id


def augment_data(x: np.ndarray, y: np.ndarray, std: np.array, n_times: int = 5, shuffle: bool = True,
                 seed: int = 42, verbose: bool = False) -> (np.ndarray, np.ndarray):

    experimental_error = {'PLGA': 1.2490, 'PP-L': 1.2121, 'PP-COOH': 1.2359,  'PP-NH2': 1.2398,  'S/AS': 0}

    rng = np.random.default_rng(seed)
    n = len(x)

    assert len(x) == len(y), f"x and y should contain the same number of samples x:{len(x)}, y:{len(y)}"
    if verbose:
        print(f"Augmenting {n_times} times; {n} original values + {n*(n_times-1)} ({n}x{(n_times-1)}) augmented values")

    x_prime, y_prime = x, y
    for i in range(n_times-1):
        x_augmentation = np.vstack([rng.normal(100, experimental_error['PLGA'], n) / 100,
                                    rng.normal(100, experimental_error['PP-L'], n) / 100,
                                    rng.normal(100, experimental_error['PP-COOH'], n) / 100,
                                    rng.normal(100, experimental_error['PP-NH2'], n) / 100,
                                    rng.normal(100, experimental_error['S/AS'], n) / 100]).T

        y_augmentation = rng.normal(100, std) / 100

        # multiply the data with the augmentation matrix
        x_prime = np.vstack((x_prime, x * x_augmentation))
        y_prime = np.append(y_prime, y * y_augmentation)

    if shuffle:
        shuffling = rng.permutation(len(x_prime))
        x_prime = x_prime[shuffling]
        y_prime = y_prime[shuffling]

    return x_prime, y_prime


def screen_predict(screen_x, screen_id, uptake_model, pdi_model, size_model, filename: str):
    """ Helper function to perform predictions and store them in the correct format for screening"""
    # Perform predictions
    y_hat_uptake, y_hat_mean_uptake, y_hat_uncertainty_uptake = uptake_model.predict(screen_x)
    y_hat_pdi, y_hat_mean_pdi, y_hat_uncertainty_pdi = pdi_model.predict(screen_x)
    y_hat_size, y_hat_mean_size, y_hat_uncertainty_size = size_model.predict(screen_x)

    # get the 90% CI
    y_hat_uptake_05 = y_hat_uptake.kthvalue(int(y_hat_uptake.shape[1] * 0.05), dim=1)[0]
    y_hat_uptake_95 = y_hat_uptake.kthvalue(int(y_hat_uptake.shape[1] * 0.95), dim=1)[0]

    screen_df = pd.DataFrame({'ID': screen_id,
                              'y_hat_uptake': y_hat_mean_uptake,
                              'y_uncertainty_uptake': y_hat_uncertainty_uptake,
                              'y_hat_uptake_CI05%': y_hat_uptake_05,
                              'y_hat_uptake_CI95%': y_hat_uptake_95,
                              'y_hat_pdi': y_hat_mean_pdi,
                              'y_uncertainty_pdi': y_hat_uncertainty_pdi,
                              'y_hat_size': y_hat_mean_size,
                              'y_uncertainty_size': y_hat_uncertainty_size,
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
