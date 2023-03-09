
"""

"""

import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
from nano.hyperopt import optimize_hyperparameters
from nano.eval import calc_rmse, evaluate_model
from nano.acquisition import dbal
from nano.utils import load_data, augment_data, screen_predict
from nano.vis import scatter
from nano.models import XGBoostEnsemble, RFEnsemble
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


""" Experimental settings ------------------------------------------------ """

# Cycle
CYCLE = 0
# Number of tries during Bayesian hyperparameter optimization.
HYPEROPT_CALLS = 500
# Number of times to bootstrap all k-fold cross-validation runs. Creates n differently split k-folds .
BOOTSTRAP = 10
# Number of folds in our k-fold cross-validation.
N_FOLDS = 5
# The size of our ensemble.
ENSEMBLE_SIZE = 10
# Number of times to augment the data. 5x means 1x the original data + 4x an augmented copy
AUGMENT = 5


""" Load data ------------------------------------------------ """

x, uptake_y, pdi_y, id, screen_x, screen_id = load_data(cycle=CYCLE, shuffle=True)


""" Uptake model ------------------------------------------------ """

# Hyperparameter optimization
best_hypers_uptake = optimize_hyperparameters(x=x, y=uptake_y,
                                              log_file=f'results/uptake_model_{CYCLE}_hypers.csv',
                                              n_calls=HYPEROPT_CALLS,
                                              bootstrap=BOOTSTRAP,
                                              n_folds=N_FOLDS,
                                              ensemble_size=ENSEMBLE_SIZE,
                                              augment=AUGMENT)

# Evaluate model performance with bootstrapped k-fold cross-validation
uptake_eval_results, uptake_rmse = evaluate_model(x=x, y=uptake_y, id=id,
                                                  filename=f'results/uptake_model_{CYCLE}_eval.csv',
                                                  hyperparameters=best_hypers_uptake,
                                                  bootstrap=BOOTSTRAP,
                                                  n_folds=N_FOLDS,
                                                  ensemble_size=ENSEMBLE_SIZE,
                                                  augment=AUGMENT)

# Quickly plot predicted vs true to make sure our model makes sense
scatter(y=uptake_y, y_hat=uptake_eval_results['y_hat'], uncertainty=uptake_eval_results['y_uncertainty'], labels=id)

# Train final model with augmented train data
uptake_x_augmented, uptake_y_augmented = augment_data(x, uptake_y, n_times=AUGMENT)
uptake_model = XGBoostEnsemble(ensemble_size=ENSEMBLE_SIZE, **best_hypers_uptake)
uptake_model.train(uptake_x_augmented, uptake_y_augmented)
torch.save(uptake_model, f'models/uptake_model_{CYCLE}.pt')


""" PdI model ------------------------------------------------ """


# hyperparameter optimization
best_hypers_pdi = optimize_hyperparameters(x=x, y=pdi_y,
                                           log_file=f'results/pdi_model_{CYCLE}_hypers.csv',
                                           n_calls=HYPEROPT_CALLS,
                                           bootstrap=BOOTSTRAP,
                                           n_folds=N_FOLDS,
                                           ensemble_size=ENSEMBLE_SIZE,
                                           augment=AUGMENT)

# Evaluate model performance with bootstrapped k-fold cross-validation
pdi_eval_results, pdi_rmse = evaluate_model(x=x, y=pdi_y, id=id,
                                            filename=f'results/pdi_model_{CYCLE}_eval.csv',
                                            hyperparameters=best_hypers_pdi,
                                            bootstrap=BOOTSTRAP,
                                            n_folds=N_FOLDS,
                                            ensemble_size=ENSEMBLE_SIZE,
                                            augment=AUGMENT)

# Quickly plot predicted vs true
scatter(y=pdi_y, y_hat=pdi_eval_results['y_hat'], uncertainty=pdi_eval_results['y_uncertainty'], labels=id)

# Train final model with augmented train data
pdi_x_augmented, pdi_y_augmented = augment_data(x, pdi_y, n_times=AUGMENT)
pdi_model_0 = XGBoostEnsemble(ensemble_size=ENSEMBLE_SIZE, **best_hypers_pdi)
pdi_model_0.train(pdi_x_augmented, pdi_y_augmented)
torch.save(pdi_model_0, f'models/pdi_model_{CYCLE}.pt')
