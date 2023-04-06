""" Range of hyperparameters that are optimized for each model"""

XGBoost_hypers = {
        # Parameters that we are going to tune.
        'max_depth': [2, 20],               # integer range from 2 - 20
        'min_child_weight': [1, 20],        # integer range from 1 - 20
        'gamma': [0.0, 10.0],               # float range from 0.0 - 10.0
        'n_estimators': [50, 500],          # integer range from 50 - 500
        'eta': [0.1, 1.0],                  # float range from 0.1 - 1.0
        'subsample': [0.1, 1.0],            # float range from 0.1 - 1.0
        'colsample_bytree': [0.1, 1.0],     # float range from 0.1 - 1.0
        'reg_alpha': [0.0, 10.0],           # float range from 0. - 10.0
        'reg_lambda': [0.0, 10.0],          # float range from 0. - 10.0
        # Other parameters
        'objective': ['reg:squarederror'],
        'eval_metric': ["rmse"]
    }


BNN_hypers = {'lr': [1e-4, 1e-3, 1e-5],              # categorical
              'hidden_size': [32],                   # categorical  64
              'epochs': [10000, 20000],              # categorical
              'n_layers': [2, 3]}                    # integer range from 2 - 5
