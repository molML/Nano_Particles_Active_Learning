
"""

Script to train models for a given cycle, evaluate models, perform predictions on the screen data, and pick the
formulations that need to be selected next iteration

Derek van Tilborg | 09-03-2023 | Eindhoven University of Technology

"""

import pandas as pd
import numpy as np
from nano.hyperopt import optimize_hyperparameters
from nano.eval import calc_rmse, evaluate_model
from nano.acquisition import acquisition_function
from nano.utils import load_data, augment_data, screen_predict
from nano.vis import scatter, picks_pca
from nano.models import XGBoostEnsemble, RFEnsemble, BayesianNN
import torch

""" Experimental settings ------------------------------------------------ """

# Cycle
CYCLE = 0
# Number of tries during Bayesian hyperparameter optimization.
HYPEROPT_CALLS = 100
# Number of times to bootstrap all k-fold cross-validation runs. Creates b differently split k-folds .
BOOTSTRAP = 5
# Number of folds in our k-fold cross-validation.
N_FOLDS = 5
# The size of our ensemble.
ENSEMBLE_SIZE = 10
# Number of times to augment the data. 5x means 1x the original data + 4x an augmented copy
AUGMENT = 5

# m fraction, see nano.acquisition
M_ACQUISITION_FRACTION = 0.10
# k, or acquisition batch size, this is the number of formulations we sample from our screen set
K_ACQUISITION_BATCHSIZE = 10
# Acquisition mode. Can be either 'explorative' or 'exploitative', see nano.acquisition
ACQUISITION_MODE = 'explorative'


""" Load data ------------------------------------------------ """

x, uptake_y, pdi_y, id, screen_x, screen_id = load_data(cycle=CYCLE, shuffle=True)


""" Uptake model ------------------------------------------------ """

# Hyperparameter optimization
best_hypers_uptake = optimize_hyperparameters(x=x, y=uptake_y,
                                              log_file=f'results/uptake_model_{CYCLE}_hypers_bnn.csv',
                                              n_calls=HYPEROPT_CALLS,
                                              bootstrap=BOOTSTRAP,
                                              n_folds=N_FOLDS,
                                              ensemble_size=ENSEMBLE_SIZE,
                                              augment=AUGMENT,
                                              model='bnn')

# best_hypers_uptake = {'lr': 0.001, 'hidden_size': 32, 'epochs': 10000, 'n_layers': 2}
# Evaluate model performance with bootstrapped k-fold cross-validation
uptake_eval_results, uptake_rmse = evaluate_model(x=x, y=uptake_y, id=id,
                                                  filename=f'results/uptake_model_{CYCLE}_eval_bnn.csv',
                                                  hyperparameters=best_hypers_uptake,
                                                  bootstrap=BOOTSTRAP,
                                                  n_folds=N_FOLDS,
                                                  ensemble_size=ENSEMBLE_SIZE,
                                                  augment=AUGMENT,
                                                  model='bnn')

# Quickly plot predicted vs true to make sure our model makes sense
scatter(y=uptake_y, y_hat=uptake_eval_results['y_hat'], uncertainty=uptake_eval_results['y_uncertainty'], labels=id)

# Train final model with augmented train data
uptake_x_augmented, uptake_y_augmented = augment_data(x, uptake_y, n_times=AUGMENT)
uptake_model = BayesianNN(**best_hypers_uptake)
uptake_model.train(uptake_x_augmented, uptake_y_augmented)
torch.save(uptake_model, f'models/uptake_model_{CYCLE}_bnn.pt')


""" PdI model ------------------------------------------------ """

# hyperparameter optimization
best_hypers_pdi = optimize_hyperparameters(x=x, y=pdi_y,
                                           log_file=f'results/pdi_model_{CYCLE}_hypers_bnn.csv',
                                           n_calls=HYPEROPT_CALLS,
                                           bootstrap=BOOTSTRAP,
                                           n_folds=N_FOLDS,
                                           ensemble_size=1,
                                           augment=AUGMENT,
                                           model='bnn')

# Evaluate model performance with bootstrapped k-fold cross-validation
pdi_eval_results, pdi_rmse = evaluate_model(x=x, y=pdi_y, id=id,
                                            filename=f'results/pdi_model_{CYCLE}_eval_bnn.csv',
                                            hyperparameters=best_hypers_pdi,
                                            bootstrap=BOOTSTRAP,
                                            n_folds=N_FOLDS,
                                            ensemble_size=ENSEMBLE_SIZE,
                                            augment=AUGMENT,
                                            model='bnn')

# Quickly plot predicted vs true
scatter(y=pdi_y, y_hat=pdi_eval_results['y_hat'], uncertainty=pdi_eval_results['y_uncertainty'], labels=id)

# Train final model with augmented train data
pdi_x_augmented, pdi_y_augmented = augment_data(x, pdi_y, n_times=AUGMENT)
# pdi_model = XGBoostEnsemble(ensemble_size=ENSEMBLE_SIZE, **best_hypers_pdi)
pdi_model = BayesianNN(**best_hypers_pdi)
pdi_model.train(pdi_x_augmented, pdi_y_augmented)
torch.save(pdi_model, f'models/pdi_model_{CYCLE}_bnn.pt')


""" Screen ------------------------------------------------ """

uptake_model = torch.load(f'models/uptake_model_{CYCLE}_bnn.pt')
pdi_model = torch.load(f'models/pdi_model_{CYCLE}_bnn.pt')

screen_df = screen_predict(screen_x, screen_id, uptake_model, pdi_model, f'results/screen_predictions_{CYCLE}_bnn.csv')


""" Sample acquisition ------------------------------------ """

previously_picked = ...  # TODO find previously picked samples

picks = acquisition_function(screen_df,
                             previously_picked=previously_picked,
                             m=int(len(screen_df)*M_ACQUISITION_FRACTION),
                             k=K_ACQUISITION_BATCHSIZE,
                             mode=ACQUISITION_MODE,
                             pdi_cutoff=0.2)

# Quick PCA to visualise screening picks sampling
picks_pca(screen_df, screen_x, picks)

# Write picks to dataframe
picks_df = screen_df.loc[screen_df['ID'].isin(picks)]
picks_df.to_csv(f'results/picks_{CYCLE}.csv')




