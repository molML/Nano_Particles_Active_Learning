""" Range of hyperparameters that are optimized for each model"""

XGBoost_hypers = {
        # Parameters that we are going to tune.
        'max_depth': [2, 20],               # integer range from 2 - 20
        'min_child_weight': [1, 20],        # integer range from 1 - 20
        'gamma': [0.0, 10.0],               # float range from 0.0 - 10.0
        'learning_rate': [0.001, 1.0],      # log-uniform float range from 0.0001 - 1.0
        'n_estimators': [50, 500],          # integer range from 50 - 500
        'eta': [0.1, 1.0],                  # float range from 0.1 - 1.0
        'max_delta_step': [0, 10],          # integer range from 0 - 10
        'subsample': [0.1, 1.0],            # float range from 0.1 - 1.0
        'colsample_bytree': [0.1, 1.0],     # float range from 0.1 - 1.0
        'reg_alpha': [0.0, 10.0],           # float range from 0. - 10.0
        'reg_lambda': [0.0, 10.0],          # float range from 0. - 10.0
        # Other parameters
        'objective': ['reg:squarederror'],
        'eval_metric': ["rmse"]
    }

RF_hypers = {'bootstrap': [True],
             'max_depth': [10, 200],            # integer range from 10 - 200
             'max_features': ['auto', 'sqrt'],  # categorical
             'min_samples_leaf': [1, 5],        # integer range from 1 - 5
             'min_samples_split': [2, 10],      # integer range from 2 - 10
             'n_estimators': [50, 2000]}        # integer range from 50 - 2000

BNN_hypers = {'lr': [1e-3, 1e-4, 1e-5],         # categorical
              'hidden_size': [16, 32, 64],      # categorical
              'epochs': [5000, 10000, 20000],   # categorical
              'n_layers': [2, 5]}               # integer range from 2 - 5
