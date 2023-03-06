"""

Code for Bayesian hyperparamter optimization using n-fold cross-validation.

Derek van Tilborg | 06-03-2023 | Eindhoven University of Technology

"""

from skopt import gp_minimize
from skopt.space.space import Categorical, Real, Integer
from nano.eval import k_fold_cross_validation, calc_rmse
from skopt.utils import use_named_args
import os
import numpy as np
from typing import Union


def optimize_hyperparameters(x: np.ndarray, y: np.ndarray, log_file: str, n_calls: int = 50, min_init_points: int = 10,
                             bootstrap: int = 10, n_folds: int = 10, ensemble_size: int = 10, augment: int = False):

    # TODO expand this list
    hypers = {
        # Parameters that we are going to tune.
        'max_depth': [2, 20],
        'min_child_weight': [1, 10],
        'gamma': [0.0, 10.0],
        'learning_rate': [0.01, 1],
        'n_estimators': [50, 200],
        'eta': [0.1, 0.5],
        'subsample': [0.5, 1.0],
        'colsample_bytree': [0.1, 1.0],
        'reg_alpha': [0.0, 10.0],
        'reg_lambda': [0.0, 10.0],
        # Other parameters
        'objective': ['reg:squarederror'],
        'eval_metric': ["rmse"]
    }

    # Optimize hyperparameters
    opt = BayesianOptimization()
    opt.optimize(x, y, dimensions=hypers, n_calls=n_calls, min_init_points=min_init_points, log_file=log_file,
                 bootstrap=bootstrap, n_folds=n_folds, ensemble_size=ensemble_size, augment=augment)

    best_hypers = get_best_hyperparameters(log_file)

    return best_hypers


class BayesianOptimization:
    def __init__(self):
        """ Init the class with a trainable model. The model class should contain a train() and test() function
        and be initialized with its hyperparameters """
        self.best_score = 100
        self.history = []
        self.results = None

    def optimize(self, x: np.ndarray, y: np.ndarray, dimensions: dict[str, list[Union[float, str, int]]],
                 n_calls: int = 50, min_init_points: int = 10, log_file: str = None, n_folds: int = 10,
                 bootstrap: int = 10, ensemble_size: int = 10, augment: int = False):

        # Prevent too mant calls if there aren't as many possible hyperparameter combi's as calls (10 in the min calls)
        dimensions = {k: [v] if type(v) is not list else v for k, v in dimensions.items()}
        dimensions = dict_to_search_space(dimensions)

        # Objective function for Bayesian optimization
        @use_named_args(dimensions=dimensions)
        def objective(**hyperparameters) -> float:

            already_done = False
            if hyperparameters in [j for i, j in self.history]:
                already_done = True
                score = [i for i, j in self.history if j == hyperparameters][0]
                print("skipping - already ran this set of hyperparameters")

            if not already_done:
                try:
                    hyperparameters = convert_types(hyperparameters)
                    print(f"Current hyperparameters: {hyperparameters}")

                    scores = []
                    for i in range(bootstrap):
                        y_hat, _ = k_fold_cross_validation(x, y, n_folds=n_folds, seed=i, ensemble_size=ensemble_size,
                                                           augment=augment, **hyperparameters)
                        scores.append(calc_rmse(y, y_hat))

                    score = sum(scores)/len(scores)

                    if log_file is not None:
                        log_hyperparameters(log_file, score=score, hypers=hyperparameters)

                # If this combination of hyperparameters fails, we use a dummy score that is worse than the best
                except:
                    print(">>  Failed")
                    score = self.best_score + 1

            self.history.append((score, hyperparameters))

            if score < self.best_score:
                self.best_score = score

            return score

        # Perform Bayesian hyperparameter optimization with 5-fold cross-validation
        self.results = gp_minimize(func=objective,
                                   dimensions=dimensions,
                                   acq_func='EI',  # expected improvement
                                   n_initial_points=min_init_points,
                                   n_calls=n_calls,
                                   verbose=True)


def dict_to_search_space(hyperparams: dict[str, list[Union[float, str, int]]]):

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
    new_dict = {}
    for k, v in params.items():
        if isinstance(v, np.generic):
            new_dict[k] = v.item()
        else:
            new_dict[k] = v
    return new_dict


def log_hyperparameters(filename: str, score: float, hypers: dict):
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write(f"score,hypers\n")
    with open(filename, 'a') as f:
        f.write(f"{score},{hypers}\n")


def get_best_hyperparameters(filename: str) -> dict:
    """ Get the best hyperparameters from the log file for an experiment + dataset combi """
    with open(filename) as f:
        best_score = 1000000
        # f=f
        for line in f.readlines()[1:]:
            linesplit = line.split(',')
            if float(linesplit[0]) < best_score:
                hypers_str = ','.join(linesplit[1:]).strip()
                best_score = float(linesplit[0])

    return eval(hypers_str)
