"""

All ML models. Firstly: model ensemble wrapper for uncertainty estimation with Random Forest and XGBoost.
We use a XGBoost model, as it is considered as the state-of-the-art for tabular data [1], with RandomForest following.
Secondly, we implement a Feed Forward Bayesian Neural Network based on Pyro [2].


[1] Ravid Shwartz-Ziv and Amitai Armon, 2021, arXiv:2106.03253
[2] Bingham, E. et al. (2019). The Journal of Machine Learning Research, 20(1), 973-978.

Derek van Tilborg | 14-03-2023 | Eindhoven University of Technology

Contains:
    - XGBoostEnsemble:  Ensemble of XGBoost models
    - RFEnsemble:       Ensemble of RF models
    - BayesianNN:       NN wrapper with train/predict functions
    - NN:               Bayesian NN based on Pyro

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
from tqdm.auto import trange
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from nano.utils import numpy_to_dataloader
import warnings
warnings.filterwarnings("ignore")


class XGBoostEnsemble:
    """ Ensemble of n XGBoost regressors, seeded differently """
    def __init__(self, ensemble_size: int = 10, **kwargs) -> None:
        self.models = {i: XGBRegressor(random_state=i, **kwargs) for i in range(ensemble_size)}

    def predict(self, x: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        y_hat = np.array([model.predict(x) for i, model in self.models.items()])
        y_hat_mean = np.mean(y_hat, axis=0)
        y_hat_uncertainty = np.std(y_hat, axis=0)

        return y_hat, y_hat_mean, y_hat_uncertainty

    def train(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        for i, m in self.models.items():
            self.models[i] = m.fit(x, y)

    def __repr__(self) -> str:
        return f"Ensemble of {len(self.models)} XGBoost Regressors"


class RFEnsemble:
    """ Ensemble of n XGBoost regressors, seeded differently """
    def __init__(self, ensemble_size: int = 10, **kwargs) -> None:
        self.models = {i: RandomForestRegressor(random_state=i, **kwargs) for i in range(ensemble_size)}

    def predict(self, x: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        y_hat = np.array([model.predict(x) for i, model in self.models.items()])
        y_hat_mean = np.mean(y_hat, axis=0)
        y_hat_uncertainty = np.std(y_hat, axis=0)

        return y_hat, y_hat_mean, y_hat_uncertainty

    def train(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        for i, m in self.models.items():
            self.models[i] = m.fit(x, y)

    def __repr__(self) -> str:
        return f"Ensemble of {len(self.models)} Random Forest Regressors"


class BayesianNN:
    """ Bayesian Neural Network wrapper with train() and predict() functions. """
    def __init__(self, in_feats: int = 5, hidden_size: int = 32, n_layers: int = 3, activation: str = 'relu',
                 seed: int = 42, lr=1e-3, to_gpu: bool = False, epochs: int = 10000):

        # Define some vars and seed random state
        self.lr = lr
        self.train_losses = []
        self.epochs = epochs
        self.epoch = 0
        self.seed = seed
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and to_gpu else "cpu")
        pyro.set_rng_seed(seed)

        # Init model
        self.model = NN(in_feats=in_feats, hidden_size=hidden_size, activation=activation, n_layers=n_layers,
                        to_gpu=to_gpu).to(self.device)

        # Init Guide model
        self.guide = AutoDiagonalNormal(self.model)
        self.guide = self.guide.to(self.device)
        # Init optimizer
        adam = pyro.optim.Adam({"lr": lr})
        # Stochastic Variational Inference
        self.svi = SVI(self.model, self.guide, adam, loss=Trace_ELBO())

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int = None, batch_size: int = 256) -> None:

        # Convert numpy to Torch
        data_loader = numpy_to_dataloader(x, y, batch_size=batch_size)

        # Training loop
        pyro.clear_param_store()
        bar = trange(self.epochs if epochs is None else epochs)
        for epoch in bar:
            running_loss = 0.0
            n_samples = 0
            for batch in data_loader:
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)

                # ELBO gradient and add loss to running loss
                running_loss += self.svi.step(x, y)
                n_samples += x.shape[0]

            loss = running_loss / n_samples
            self.train_losses.append(loss)
            bar.set_postfix(loss=f'{loss:.4f}')

    def predict(self, x, num_samples: int = 500, return_posterior: bool = False, batch_size: int = 256) -> \
            (tensor, tensor, tensor):

        # Convert numpy to Torch
        data_loader = numpy_to_dataloader(x, batch_size=batch_size)

        # Construct predictive distribution
        predictive = Predictive(self.model, guide=self.guide, num_samples=num_samples, return_sites=("obs", "_RETURN"))

        predictions = {'y_hat': {'pred': tensor([], device=self.device), 'mean': tensor([], device=self.device),
                                 'std': tensor([], device=self.device)},
                       'posterior': {'pred': tensor([], device=self.device), 'mean': tensor([], device=self.device),
                                     'std': tensor([], device=self.device)}}

        for batch in data_loader:

            x = batch[0].to(self.device)

            # Reshape if needed
            samples = predictive(x.unsqueeze(0) if len(x.size()) == 1 else x)
            n = len(samples['obs'])
            if len(samples['_RETURN'].size()) == 1 and return_posterior:
                samples['_RETURN'] = samples['_RETURN'].reshape([n, 1])

            # Add predictions from each batch to the dict
            predictions['y_hat']['pred'] = torch.cat((predictions['y_hat']['pred'], samples['obs'].T), 0)
            y_hat_mean = torch.mean(samples['obs'], dim=0)
            predictions['y_hat']['mean'] = torch.cat((predictions['y_hat']['mean'], y_hat_mean), 0)
            y_hat_std = torch.std(samples['obs'], dim=0)
            predictions['y_hat']['std'] = torch.cat((predictions['y_hat']['std'], y_hat_std), 0)

            if return_posterior:
                # Add predictions from each batch to the dict
                predictions['posterior']['pred'] = torch.cat((predictions['posterior']['pred'], samples['obs'].T), 0)
                post_mean = torch.mean(samples['_RETURN'], dim=0)
                predictions['posterior']['mean'] = torch.cat((predictions['posterior']['mean'], post_mean), 0)
                post_std = torch.std(samples['_RETURN'], dim=0)
                predictions['posterior']['std'] = torch.cat((predictions['posterior']['std'], post_std), 0)

        if return_posterior:
            return predictions['y_hat']['pred'], predictions['y_hat']['mean'], predictions['y_hat']['std'], \
                   predictions['posterior']['pred'], predictions['posterior']['mean'], predictions['posterior']['std']

        return predictions['y_hat']['pred'], predictions['y_hat']['mean'], predictions['y_hat']['std']


class NN(PyroModule):
    """ Simple feedforward Bayesian NN. We use PyroSample to place a prior over the weight and bias parameters,
     instead of treating them as fixed learnable parameters. See http://pyro.ai/examples/bayesian_regression.html"""
    def __init__(self, in_feats=5, n_layers: int = 3, hidden_size: int = 32, activation: str = 'relu',
                 to_gpu: bool = True, out_feats: int = 1) -> None:
        super().__init__()

        # Define some vars
        self.n_layers, nh0, nh, nhout = n_layers, in_feats, hidden_size, out_feats
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and to_gpu else "cpu")
        assert 1 <= n_layers <= 5, f"'n_layers' should be between 1 and 5, not {n_layers}"
        assert activation in ['relu', 'gelu', 'elu', 'leaky']

        self.fc0 = PyroModule[nn.Linear](nh0, nh)
        self.fc0.weight = PyroSample(dist.Normal(0., tensor(1.0, device=self.device)).expand([nh, nh0]).to_event(2))
        self.fc0.bias = PyroSample(dist.Normal(0., tensor(1.0, device=self.device)).expand([nh]).to_event(1))

        if n_layers > 1:
            self.fc1 = PyroModule[nn.Linear](nh, nh)
            self.fc1.weight = PyroSample(dist.Normal(0., tensor(1.0, device=self.device)).expand([nh, nh]).to_event(2))
            self.fc1.bias = PyroSample(dist.Normal(0., tensor(1.0, device=self.device)).expand([nh]).to_event(1))

        if n_layers > 2:
            self.fc2 = PyroModule[nn.Linear](nh, nh)
            self.fc2.weight = PyroSample(dist.Normal(0., tensor(1.0, device=self.device)).expand([nh, nh]).to_event(2))
            self.fc2.bias = PyroSample(dist.Normal(0., tensor(1.0, device=self.device)).expand([nh]).to_event(1))

        if n_layers > 3:
            self.fc3 = PyroModule[nn.Linear](nh, nh)
            self.fc3.weight = PyroSample(dist.Normal(0., tensor(1.0, device=self.device)).expand([nh, nh]).to_event(2))
            self.fc3.bias = PyroSample(dist.Normal(0., tensor(1.0, device=self.device)).expand([nh]).to_event(1))

        if n_layers > 4:
            self.fc4 = PyroModule[nn.Linear](nh, nh)
            self.fc4.weight = PyroSample(dist.Normal(0., tensor(1.0, device=self.device)).expand([nh, nh]).to_event(2))
            self.fc4.bias = PyroSample(dist.Normal(0., tensor(1.0, device=self.device)).expand([nh]).to_event(1))

        self.out = PyroModule[nn.Linear](nh, nhout)
        self.out.weight = PyroSample(dist.Normal(0., tensor(1.0, device=self.device)).expand([nhout, nh]).to_event(2))
        self.out.bias = PyroSample(dist.Normal(0., tensor(1.0, device=self.device)).expand([nhout]).to_event(1))

    def forward(self, x: tensor, y: tensor = None) -> tensor:
        # Predict the mean
        x = F.relu(self.fc0(x))
        if self.n_layers > 1:
            x = F.relu(self.fc1(x))
        if self.n_layers > 2:
            x = F.relu(self.fc2(x))
        if self.n_layers > 3:
            x = F.relu(self.fc3(x))
        if self.n_layers > 4:
            x = F.relu(self.fc4(x))
        mu = self.out(x).squeeze()

        # Give the obs argument to the pyro.sample statement to condition on the observed data y_data with a learned
        # observation noise sigma. See http://pyro.ai/examples/bayesian_regression.html
        sigma = pyro.sample("sigma", dist.Uniform(0., tensor(1.0, device=self.device)))
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma), obs=y)

        return mu
