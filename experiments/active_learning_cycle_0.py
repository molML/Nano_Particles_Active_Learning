
"""

Script to train models for a given cycle, evaluate models, perform predictions on the screen data, and pick the
formulations that need to be selected next iteration

Derek van Tilborg | 09-03-2023 | Eindhoven University of Technology

"""

import pandas as pd
from nano.hyperopt import optimize_hyperparameters
from nano.eval import calc_rmse, evaluate_model
from nano.acquisition import acquisition_function
from nano.utils import load_data, augment_data, screen_predict
from nano.vis import scatter, picks_pca
from nano.models import XGBoostEnsemble, BayesianNN
import torch
pd.set_option('display.max_columns', None)


""" Experimental settings ------------------------------------------------ """

# Cycle
CYCLE = 0
DATE = '5Apr'
# Number of tries during Bayesian hyperparameter optimization.
HYPEROPT_CALLS_XGB = 500
# Number of times to bootstrap all k-fold cross-validation runs. Creates b differently split k-folds .
BOOTSTRAP = 3
# Number of folds in our k-fold cross-validation.
N_FOLDS = 5
# Number of times to augment the data. 5x means 1x the original data + 4x an augmented copy
AUGMENT = 5

# m fraction, see nano.acquisition
M_ACQUISITION_FRACTION = 0.05
# k, or acquisition batch size, this is the number of formulations we sample from our screen set
K_ACQUISITION_BATCHSIZE = 10
# Acquisition mode. Can be either 'explorative' or 'exploitative', see nano.acquisition
ACQUISITION_MODE = 'explorative'

if __name__ == '__main__':

    """ Load data ------------------------------------------------ """

    uptake_x, uptake_y, uptake_std, uptake_id = load_data(cycle=CYCLE, set='uptake', shuffle=True, omit_unstable=True)

    """ Uptake model ------------------------------------------------ """

    # Hyperparameter optimization
    best_hypers_uptake = optimize_hyperparameters(x=uptake_x, y=uptake_y, std=uptake_std,
                                                  log_file=f'results/uptake_model_{CYCLE}_hypers_bnn_{DATE}.csv',
                                                  bootstrap=BOOTSTRAP,
                                                  n_folds=N_FOLDS,
                                                  augment=AUGMENT,
                                                  model='bnn',
                                                  method='grid_search')

    # Evaluate model performance with bootstrapped k-fold cross-validation
    uptake_eval_results, uptake_rmse = evaluate_model(x=uptake_x, y=uptake_y, std=uptake_std, id=uptake_id,
                                                      filename=f'results/uptake_model_{CYCLE}_eval_bnn_{DATE}.csv',
                                                      hyperparameters=best_hypers_uptake,
                                                      bootstrap=BOOTSTRAP,
                                                      n_folds=N_FOLDS,
                                                      augment=AUGMENT,
                                                      model='bnn')

    # Quickly plot predicted vs true to make sure our model makes sense
    scatter(y=uptake_y, y_hat=uptake_eval_results['y_hat'], uncertainty=uptake_eval_results['y_uncertainty'],
            labels=uptake_id)

    # Train final model with augmented train data
    uptake_x_augmented, uptake_y_augmented = augment_data(uptake_x, uptake_y, uptake_std, n_times=AUGMENT)
    uptake_model = BayesianNN(**best_hypers_uptake)
    uptake_model.train(uptake_x_augmented, uptake_y_augmented)
    torch.save(uptake_model, f'models/uptake_model_{CYCLE}_bnn_{DATE}.pt')

    """ PdI model ------------------------------------------------ """

    # Load data, we now include unstable particles to learn from them
    pdi_x, pdi_y, pdi_std, pdi_id = load_data(cycle=CYCLE, set='pdi', shuffle=True, omit_unstable=False)

    # hyperparameter optimization
    best_hypers_pdi = optimize_hyperparameters(x=pdi_x, y=pdi_y, std=pdi_std,
                                               log_file=f'results/pdi_model_{CYCLE}_hypers_xgb_{DATE}.csv',
                                               n_calls=HYPEROPT_CALLS_XGB,
                                               bootstrap=BOOTSTRAP,
                                               n_folds=N_FOLDS,
                                               augment=AUGMENT,
                                               model='xgb',
                                               method='bayesian')

    # Evaluate model performance with bootstrapped k-fold cross-validation
    pdi_eval_results, pdi_rmse = evaluate_model(x=pdi_x, y=pdi_y, std=pdi_std, id=pdi_id,
                                                filename=f'results/pdi_model_{CYCLE}_eval_xgb_{DATE}.csv',
                                                hyperparameters=best_hypers_pdi,
                                                bootstrap=BOOTSTRAP,
                                                n_folds=N_FOLDS,
                                                augment=AUGMENT,
                                                model='xgb')

    # Quickly plot predicted vs true
    scatter(y=pdi_y, y_hat=pdi_eval_results['y_hat'], uncertainty=pdi_eval_results['y_uncertainty'], labels=pdi_id)

    # Train final model with augmented train data
    pdi_x_augmented, pdi_y_augmented = augment_data(pdi_x, pdi_y, pdi_std, n_times=AUGMENT)
    pdi_model = XGBoostEnsemble(ensemble_size=1, **best_hypers_pdi)  # we use a single xgb model
    pdi_model.train(pdi_x_augmented, pdi_y_augmented)
    torch.save(pdi_model, f'models/pdi_model_{CYCLE}_xgb_{DATE}.pt')


    """ Size model ------------------------------------------------ """

    # Load data, we now include unstable particles to learn from them
    size_x, size_y, size_std, size_id = load_data(cycle=CYCLE, set='size', shuffle=True, omit_unstable=False)

    # hyperparameter optimization
    best_hypers_size = optimize_hyperparameters(x=size_x, y=size_y, std=size_std,
                                               log_file=f'results/size_model_{CYCLE}_hypers_xgb_{DATE}.csv',
                                               n_calls=HYPEROPT_CALLS_XGB,
                                               bootstrap=BOOTSTRAP,
                                               n_folds=N_FOLDS,
                                               augment=AUGMENT,
                                               model='xgb',
                                               method='bayesian')

    # Evaluate model performance with bootstrapped k-fold cross-validation
    size_eval_results, size_rmse = evaluate_model(x=size_x, y=size_y, std=size_std, id=size_id,
                                                filename=f'results/size_model_{CYCLE}_eval_xgb_{DATE}.csv',
                                                hyperparameters=best_hypers_size,
                                                bootstrap=BOOTSTRAP,
                                                n_folds=N_FOLDS,
                                                augment=AUGMENT,
                                                model='xgb')

    # Quickly plot predicted vs true
    scatter(y=size_y, y_hat=size_eval_results['y_hat'], uncertainty=size_eval_results['y_uncertainty'], labels=size_id)

    # Train final model with augmented train data
    size_x_augmented, size_y_augmented = augment_data(size_x, size_y, size_std, n_times=AUGMENT)
    size_model = XGBoostEnsemble(ensemble_size=1, **best_hypers_size)  # we use a single xgb model
    size_model.train(size_x_augmented, size_y_augmented)
    torch.save(size_model, f'models/size_model_{CYCLE}_xgb_{DATE}.pt')

    """ Screen ------------------------------------------------ """

    screen_x, screen_id = load_data(cycle=CYCLE, set='screen', shuffle=True, omit_unstable=False)

    uptake_model = torch.load(f'models/uptake_model_{CYCLE}_bnn_{DATE}.pt')
    pdi_model = torch.load(f'models/pdi_model_{CYCLE}_xgb_{DATE}.pt')
    size_model = torch.load(f'models/size_model_{CYCLE}_xgb_{DATE}.pt')

    screen_df = screen_predict(screen_x, screen_id, uptake_model, pdi_model, size_model,
                               f'results/screen_predictions_{CYCLE}_{DATE}.csv')

    """ Sample acquisition ------------------------------------ """

    picks = acquisition_function(screen_df,
                                 m=int(len(screen_df)*M_ACQUISITION_FRACTION),
                                 k=K_ACQUISITION_BATCHSIZE,
                                 mode=ACQUISITION_MODE,
                                 pdi_cutoff=0.2,
                                 size_cutoff=None)

    # Quick PCA to visualise screening picks sampling
    picks_pca(screen_df, screen_x, picks)

    # Write picks to dataframe
    picks_df = screen_df.loc[screen_df['ID'].isin(picks)]
    picks_df.to_csv(f'results/picks_{CYCLE}_{DATE}.csv', index=False)
