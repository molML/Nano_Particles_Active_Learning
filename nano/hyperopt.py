"""

Code for Bayesian hyperparamter optimization using bootstrapped n-fold cross-validation.
    - optimize_hyperparameters()
    - BayesianOptimization
    - dict_to_search_space()
    - convert_types()
    - get_best_hyperparameters()

Derek van Tilborg | 06-03-2023 | Eindhoven University of Technology

"""
import warnings

import numpy as np
from typing import Union
import itertools

import pandas as pd
from skopt import gp_minimize
from skopt.space.space import Categorical, Real, Integer
from skopt.utils import use_named_args
from nano.eval import k_fold_cross_validation, calc_rmse
from nano.hyperparameters import XGBoost_hypers, BNN_hypers


def optimize_hyperparameters(x: np.ndarray, y: np.ndarray, std: np.array, log_file: str = 'hypers_log.csv',
                             n_calls: int = 50, min_init_points: int = 10, bootstrap: int = 5, n_folds: int = 5,
                             ensemble_size: int = 1, augment: int = False, model="bnn",
                             method: str = 'grid_search') -> dict:
    """ Wrapper function to optimize hyperparameters on a dataset using bootstrapped k-fold cross-validation """

    assert model in ['bnn', 'xgb'], f"'model' must be 'bnn', or 'xgb'"

    if model == 'xgb':
        hypers = XGBoost_hypers
    if model == 'bnn':
        hypers = BNN_hypers

    if method == 'bayesian':
        # Optimize hyperparameters
        opt = BayesianOptimization()
        opt.optimize(x, y, std, dimensions=hypers, n_calls=n_calls, min_init_points=min_init_points, log_file=log_file,
                     bootstrap=bootstrap, n_folds=n_folds, ensemble_size=ensemble_size, augment=augment, model=model)
    elif method == 'grid_search':
        grid_search(x, y, std, dimensions=hypers, log_file=log_file, bootstrap=bootstrap, n_folds=n_folds,
                    ensemble_size=ensemble_size, augment=augment, model=model)

    best_hypers = get_best_hyperparameters(log_file)

    return best_hypers


def grid_search(x, y, std, dimensions, log_file: str, bootstrap: int = 5, n_folds: int = 5, ensemble_size: int = 10,
                augment: int = 5, model: str = 'bnn'):

    all_hypers = [dict(zip(dimensions.keys(), v)) for v in itertools.product(*dimensions.values())]
    for hypers in all_hypers:

        print(f"Current hyperparameters: {hypers}")
        with open(log_file) as f:
            previous_hypers = [eval(','.join(l.strip().split(',')[1:])) for l in f.readlines()]
        if hypers not in previous_hypers:
            scores = []
            for i in range(bootstrap):
                try:
                    y_hats, y_hats_mean, y_hats_uncertainty = k_fold_cross_validation(x, y, std, n_folds=n_folds, seed=i,
                                                                                      ensemble_size=ensemble_size,
                                                                                      augment=augment, model=model, **hypers)
                    scores.append(calc_rmse(y, y_hats_mean))
                except:
                    warnings.warn(f'Failed run {i} for {hypers}')

            if len(scores) == 0:
                score = 'error'
            else:
                score = sum(scores) / len(scores)

            with open(log_file, 'a') as f:
                f.write(f"{score},{hypers}\n")


class BayesianOptimization:
    def __init__(self):
        """ Init the class with a trainable model. The model class should contain a train() and predict() function
        and be initialized with all of its hyperparameters """
        self.best_score = 1000  # Arbitrary high starting score
        self.history = []
        self.results = None

    # dimensions= XGBoost_hypers
    def optimize(self, x: np.ndarray, y: np.ndarray, std: np.array, dimensions: dict[str, list[Union[float, str, int]]],
                 n_calls: int = 100, min_init_points: int = 10, log_file: str = None, n_folds: int = 5,
                 bootstrap: int = 5, ensemble_size: int = 1, augment: int = False, model: str = 'bnn'):

        # Convert dict of hypers to skopt search_space
        dimensions = {k: [v] if type(v) is not list else v for k, v in dimensions.items()}
        dimensions = dict_to_search_space(dimensions)

        # touch hypers log file
        with open(log_file, 'w') as f:
            f.write(f"score,hypers\n")

        # Objective function for Bayesian optimization
        @use_named_args(dimensions=dimensions)
        def objective(**hyperparameters) -> float:
            # If the same set of hypers gets selected twice (which can happen in the first few runs), skip it
            if hyperparameters in [j for i, j in self.history]:
                score = [i for i, j in self.history if j == hyperparameters][0]
                print(f"skipping - already ran this set of hyperparameters: {hyperparameters}")
            else:
                try:
                    hyperparameters = convert_types(hyperparameters)
                    print(f"Current hyperparameters: {hyperparameters}")

                    scores = []
                    for i in range(bootstrap):
                        # break
                        y_hats, y_hats_mu, y_hats_sigma = k_fold_cross_validation(x, y, std, n_folds=n_folds,  seed=i,
                                                                                  ensemble_size=ensemble_size,
                                                                                  augment=augment, model=model,
                                                                                  **hyperparameters)
                        scores.append(calc_rmse(y, y_hats_mu))

                    score = sum(scores)/len(scores)

                    with open(log_file, 'a') as f:
                        f.write(f"{score},{hyperparameters}\n")

                # If this combination of hyperparameters fails, we use a dummy score that is worse than the best
                except:
                    print(">>  Failed")
                    score = self.best_score + 1
            if score > 100000:
                score = self.best_score + 1

            # append to history and update best score if needed
            self.history.append((score, hyperparameters))
            if score < self.best_score:
                self.best_score = score

            return score

        # Perform Bayesian hyperparameter optimization with n-fold cross-validation
        self.results = gp_minimize(func=objective,
                                   dimensions=dimensions,
                                   acq_func='EI',  # Expected Improvement
                                   n_initial_points=min_init_points,    # Run for this many cycles randomly before BO
                                   n_calls=n_calls,  # Total calls
                                   verbose=True)


def dict_to_search_space(hyperparams: dict[str, list[Union[float, str, int]]]) -> list:
    """ Takes a dict of hyperparameters and converts to skopt search_space"""

    search_space = []

    for k, v in hyperparams.items():
        if type(v[0]) is float and len(v) == 2:
            if k == 'lr' or k == 'learning_rate':
                search_space.append(Real(low=min(v), high=max(v), prior='log-uniform', name=k))
            else:
                search_space.append(Real(low=min(v), high=max(v), name=k))
        elif type(v[0]) is int and len(v) == 2:
            search_space.append(Integer(low=min(v), high=max(v), name=k))
        else:
            search_space.append(Categorical(categories=list(v), name=k))

    return search_space


def convert_types(params: dict) -> dict:
    """ Convert to proper typing. For some reason skopt will mess with float and int typing"""
    new_dict = {}
    for k, v in params.items():
        if isinstance(v, np.generic):
            new_dict[k] = v.item()
        else:
            new_dict[k] = v
    return new_dict


def get_best_hyperparameters(filename: str) -> dict:
    """ Get the best hyperparameters from the log file for an experiment """
    with open(filename) as f:
        best_score = 1000000
        for line in f.readlines()[1:]:
            linesplit = line.split(',')
            if float(linesplit[0]) < best_score:
                hypers_str = ','.join(linesplit[1:]).strip()
                best_score = float(linesplit[0])

    return eval(hypers_str)
