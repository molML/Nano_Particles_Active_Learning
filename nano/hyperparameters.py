""" Range of hyperparameters that are optimized for each model"""
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct, \
    ConstantKernel as C


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


BNN_hypers = {'lr': [1e-4, 1e-3, 1e-5],       # categorical
              'hidden_size': [16, 32, 64],    # categorical  64
              'epochs': [10000],              # categorical
              'n_layers': [3]}                # integer range from 2 - 5


RF_hypers = {'n_estimators': [50, 500],       # integer range from 50 - 500
             'max_depth': [10, 50],           # integer range from 10 - 50
             'min_samples_split': [2, 10]}    # integer range from 2 - 10


GP_hypers = {'kernel': [C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
                        C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5),
                        C(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=1.0, alpha=1.0),
                        C(1.0, (1e-3, 1e3)) * ExpSineSquared(length_scale=1.0, periodicity=3.0,
                                                             length_scale_bounds=(1e-2, 1e2),
                                                             periodicity_bounds=(1e-2, 1e1)),
                        C(1.0, (1e-3, 1e3)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-2, 1e1))],
             'alpha': [1e-3, 1e-2, 1e-1, 1]}  # Regularization term
