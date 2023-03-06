"""

Model Ensemble wrapper for uncertainty estimation. We use a XGBoost model, as it is considered as the state-of-the-art
for tabular data [1].

[1] Ravid Shwartz-Ziv and Amitai Armon, 2021, arXiv:2106.03253

Derek van Tilborg | 06-03-2023 | Eindhoven University of Technology

"""

from xgboost import XGBRegressor
import numpy as np


class XGBoostEnsemble:
    """ Ensemble of n XGBoost regressors, seeded differently """
    def __init__(self, ensemble_size: int = 10, **kwargs) -> None:

        self.models = {i: XGBRegressor(random_state=i, **kwargs) for i in range(ensemble_size)}

    def predict(self, x: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        y_hat = np.array([model.predict(x) for i, model in self.models.items()])
        y_hat_mean = np.mean(y_hat, axis=0)
        y_hat_uncertainty = np.std(y_hat, axis=0)

        return y_hat, y_hat_mean, y_hat_uncertainty

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        for i, m in self.models.items():
            self.models[i] = m.fit(x, y)

    def __repr__(self) -> str:
        return f"Ensemble of {len(self.models)} XGBoost Regressors"
