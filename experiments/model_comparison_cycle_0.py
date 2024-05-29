
"""

Script to train models for a given cycle, evaluate models, perform predictions on the screen data, and pick the
formulations that need to be selected next iteration

Derek van Tilborg | 09-03-2023 | Eindhoven University of Technology

"""

import pandas as pd
from nano.hyperopt import optimize_hyperparameters, get_best_hyperparameters
from nano.eval import evaluate_model
from nano.utils import load_data
pd.set_option('display.max_columns', None)


""" Experimental settings ------------------------------------------------ """


# Number of tries during Bayesian hyperparameter optimization.
HYPEROPT_CALLS_XGB = 500
HYPEROPT_CALLS_RF = 500
# Number of times to bootstrap all k-fold cross-validation runs. Creates b differently split k-folds .
BOOTSTRAP = 3
# Number of folds in our k-fold cross-validation.
N_FOLDS = 5
# Number of times to augment the data. 5x means 1x the original data + 4x an augmented copy
AUGMENT = 5

CYCLE = 0
DATE = '28May'

if __name__ == '__main__':

    """ Load data ------------------------------------------------ """

    uptake_x, uptake_y, uptake_std, uptake_id = load_data(cycle=CYCLE, set='uptake', shuffle=True, omit_unstable=True)

    """ BNN model ------------------------------------------------ """

    best_hypers_bnn = get_best_hyperparameters(f'results/uptake_model_{CYCLE}_hypers_bnn_5Apr.csv')

    # Evaluate model performance with bootstrapped k-fold cross-validation
    bnn_eval_results, bnn_rmse = evaluate_model(x=uptake_x, y=uptake_y, std=uptake_std, id=uptake_id,
                                                      filename=f'results/uptake_model_{CYCLE}_eval_bnn_{DATE}.csv',
                                                      hyperparameters=best_hypers_bnn,
                                                      bootstrap=BOOTSTRAP,
                                                      n_folds=N_FOLDS,
                                                      augment=AUGMENT,
                                                      model='bnn')

    """ XGBoost model --------------------------------------------- """

    # hyperparameter optimization
    best_hypers_xgb = optimize_hyperparameters(x=uptake_x, y=uptake_y, std=uptake_std,
                                               log_file=f'results/uptake_model_{CYCLE}_hypers_xgb_{DATE}.csv',
                                               n_calls=HYPEROPT_CALLS_XGB,  # HYPEROPT_CALLS_XGB
                                               bootstrap=BOOTSTRAP,
                                               n_folds=N_FOLDS,
                                               augment=AUGMENT,
                                               model='xgb',
                                               method='bayesian')

    # Evaluate model performance with bootstrapped k-fold cross-validation
    xgb_eval_results, xgb_rmse = evaluate_model(x=uptake_x, y=uptake_y, std=uptake_std, id=uptake_id,
                                                filename=f'results/uptake_model_{CYCLE}_eval_xgb_{DATE}.csv',
                                                hyperparameters=best_hypers_xgb,
                                                bootstrap=BOOTSTRAP,
                                                n_folds=N_FOLDS,
                                                ensemble_size=10,
                                                augment=AUGMENT,
                                                model='xgb')


    """ RFBoost model --------------------------------------------- """

    # hyperparameter optimization
    best_hypers_rf = optimize_hyperparameters(x=uptake_x, y=uptake_y, std=uptake_std,
                                              log_file=f'results/uptake_model_{CYCLE}_hypers_rf_{DATE}.csv',
                                              n_calls=HYPEROPT_CALLS_RF,
                                              bootstrap=BOOTSTRAP,
                                              n_folds=N_FOLDS,
                                              augment=AUGMENT,
                                              model='rf',
                                              method='bayesian')

    # Evaluate model performance with bootstrapped k-fold cross-validation
    rf_eval_results, rf_rmse = evaluate_model(x=uptake_x, y=uptake_y, std=uptake_std, id=uptake_id,
                                              filename=f'results/uptake_model_{CYCLE}_eval_rf_{DATE}.csv',
                                              hyperparameters=best_hypers_rf,
                                              bootstrap=BOOTSTRAP,
                                              n_folds=N_FOLDS,
                                              ensemble_size=10,
                                              augment=AUGMENT,
                                              model='rf')



    """ GP model ---------------------------------------------------- """

    # hyperparameter optimization
    best_hypers_gp = optimize_hyperparameters(x=uptake_x, y=uptake_y, std=uptake_std,
                                              log_file=f'results/uptake_model_{CYCLE}_hypers_gp_{DATE}.csv',
                                              bootstrap=BOOTSTRAP,
                                              n_folds=N_FOLDS,
                                              augment=AUGMENT,
                                              model='gp',
                                              method='grid_search')

    # Evaluate model performance with bootstrapped k-fold cross-validation
    gp_eval_results, gp_rmse = evaluate_model(x=uptake_x, y=uptake_y, std=uptake_std, id=uptake_id,
                                              filename=f'results/uptake_model_{CYCLE}_eval_gp_{DATE}.csv',
                                              hyperparameters=best_hypers_gp,
                                              bootstrap=BOOTSTRAP,
                                              n_folds=N_FOLDS,
                                              augment=AUGMENT,
                                              model='gp')
