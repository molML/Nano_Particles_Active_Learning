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
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal, init_to_mean
from pyro.infer import SVI, Trace_ELBO, Predictive
from tqdm.auto import trange, tqdm
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from nano.utils import numpy_to_dataloader
import warnings
warnings.filterwarnings("ignore")


class XGBoostEnsemble:
    """ Ensemble of n XGBoost regressors, seeded differently """
    def __init__(self, ensemble_size: int = 10, log_transform: bool = True, **kwargs) -> None:
        self.models = {i: XGBRegressor(random_state=i, **kwargs) for i in range(ensemble_size)}
        self.log_transform = log_transform

    def predict(self, x: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        y_hat = np.array([model.predict(x) for i, model in self.models.items()])
        if self.log_transform:
            y_hat = 10 ** y_hat
        y_hat_mean = np.mean(y_hat, axis=0)
        y_hat_uncertainty = np.std(y_hat, axis=0)

        return y_hat, y_hat_mean, y_hat_uncertainty

    def train(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        if self.log_transform:
            y = np.log10(y)
        for i, m in self.models.items():
            self.models[i] = m.fit(x, y)

    def __repr__(self) -> str:
        return f"Ensemble of {len(self.models)} XGBoost Regressors"


class RFEnsemble:
    """ Ensemble of n Random Forest regressors, seeded differently """
    def __init__(self, ensemble_size: int = 10, log_transform: bool = True, **kwargs) -> None:
        self.models = {i: RandomForestRegressor(random_state=i, **kwargs) for i in range(ensemble_size)}
        self.log_transform = log_transform

    def predict(self, x: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        y_hat = np.array([model.predict(x) for i, model in self.models.items()])
        if self.log_transform:
            y_hat = 10 ** y_hat
        y_hat_mean = np.mean(y_hat, axis=0)
        y_hat_uncertainty = np.std(y_hat, axis=0)

        return y_hat, y_hat_mean, y_hat_uncertainty

    def train(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        if self.log_transform:
            y = np.log10(y)
        for i, m in self.models.items():
            self.models[i] = m.fit(x, y)

    def __repr__(self) -> str:
        return f"Ensemble of {len(self.models)} Random Forest Regressors"


class GP:
    """ Gaussian process regressors """
    def __init__(self, log_transform: bool = True, **kwargs) -> None:
        self.model = GaussianProcessRegressor(**kwargs)
        self.log_transform = log_transform

    def predict(self, x: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        y_hat_mean, y_hat_uncertainty = self.model.predict(x, return_std=True)

        if self.log_transform:
            y_hat_mean = 10 ** y_hat_mean

            variance_original = (10 ** (y_hat_mean ** 2 * np.log(10) ** 2) - 1) * 10 ** (
                        2 * y_hat_uncertainty + y_hat_mean ** 2 * np.log(10) ** 2)
            # Calculate the standard deviation in the original scale
            y_hat_uncertainty = np.sqrt(variance_original)

        y_hat = y_hat_mean

        return y_hat, y_hat_mean, y_hat_uncertainty

    def train(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        if self.log_transform:
            y = np.log10(y)
        self.model.fit(x, y)

    def __repr__(self) -> str:
        return f"Gaussian Process Regressors"



class BayesianNN:
    """ Bayesian Neural Network wrapper with train() and predict() functions. """
    def __init__(self, in_feats: int = 5, hidden_size: int = 32, n_layers: int = 3, activation: str = 'relu',
                 seed: int = 42, lr=1e-3, to_gpu: bool = True, epochs: int = 10000, weight_mu: float = 0.0,
                 weight_sigma: float = 1.0, bias_mu: float = 0.0, bias_sigma: float = 1.0, log_transform: bool = False):

        # Define some vars and seed random state
        self.lr = lr
        self.train_losses = []
        self.epochs = epochs
        self.epoch = 0
        self.seed = seed
        self.log_transform = log_transform
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and to_gpu else "cpu")
        pyro.set_rng_seed(seed)

        # Init model
        self.model = NN(in_feats=in_feats, hidden_size=hidden_size, activation=activation, n_layers=n_layers,
                        to_gpu=to_gpu, weight_mu=weight_mu, weight_sigma=weight_sigma, bias_mu=bias_mu,
                        bias_sigma=bias_sigma).to(self.device)

        # Init Guide model
        self.guide = AutoMultivariateNormal(self.model)
        # self.guide = AutoDiagonalNormal(self.model)
        self.guide = self.guide.to(self.device)
        # Init optimizer
        adam = pyro.optim.Adam({"lr": lr})
        # Stochastic Variational Inference
        self.svi = SVI(self.model, self.guide, adam, loss=Trace_ELBO())

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int = None, batch_size: int = 256) -> None:

        # Convert numpy to Torch
        if self.log_transform:
            y = np.log10(y)
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

        y_hat = tensor([], device=self.device)
        y_hat_mu = tensor([], device=self.device)
        y_hat_sigma = tensor([], device=self.device)
        posterior = tensor([], device=self.device)

        for batch in tqdm(data_loader, 'Sampling predictive distribution'):
            x = batch[0].to(self.device)

            # Reshape if needed
            samples = predictive(x.unsqueeze(0) if len(x.size()) == 1 else x)
            if len(samples['_RETURN'].size()) == 1 and return_posterior:
                samples['_RETURN'] = samples['_RETURN'].reshape([len(samples['obs']), 1])

            preds = 10 ** samples['obs'] if self.log_transform else samples['obs']
            posts = 10 ** samples['_RETURN'] if self.log_transform else samples['_RETURN']

            # Add predictions from each batch to the dict
            y_hat = torch.cat((y_hat, preds.T), 0)
            y_hat_mu = torch.cat((y_hat_mu, torch.mean(preds, dim=0)), 0)
            y_hat_sigma = torch.cat((y_hat_sigma, torch.std(preds, dim=0)), 0)
            posterior = torch.cat((posterior, posts.T), 0)

        if return_posterior:
            return y_hat.cpu(), y_hat_mu.cpu(), y_hat_sigma.cpu(), posterior.cpu()

        return y_hat.cpu(), y_hat_mu.cpu(), y_hat_sigma.cpu()


class NN(PyroModule):
    """ Simple feedforward Bayesian NN. We use PyroSample to place a prior over the weight and bias parameters,
     instead of treating them as fixed learnable parameters. See http://pyro.ai/examples/bayesian_regression.html"""
    def __init__(self, in_feats=5, n_layers: int = 3, hidden_size: int = 32, activation: str = 'relu',
                 to_gpu: bool = True, out_feats: int = 1, weight_mu: float = 0.0, weight_sigma: float = 1.0,
                 bias_mu: float = 0.0, bias_sigma: float = 1.0) -> None:
        super().__init__()

        # Define some vars
        self.n_layers, nh0, nh, nhout = n_layers, in_feats, hidden_size, out_feats
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and to_gpu else "cpu")
        assert 1 <= n_layers <= 5, f"'n_layers' should be between 1 and 5, not {n_layers}"
        assert activation in ['relu', 'gelu', 'elu', 'leaky']

        self.fc0 = PyroModule[nn.Linear](nh0, nh)
        self.fc0.weight = PyroSample(dist.Normal(weight_mu, tensor(weight_sigma, device=self.device)).expand([nh, nh0]).to_event(2))
        self.fc0.bias = PyroSample(dist.Normal(bias_mu, tensor(bias_sigma, device=self.device)).expand([nh]).to_event(1))

        if n_layers > 1:
            self.fc1 = PyroModule[nn.Linear](nh, nh)
            self.fc1.weight = PyroSample(dist.Normal(weight_mu, tensor(weight_sigma, device=self.device)).expand([nh, nh]).to_event(2))
            self.fc1.bias = PyroSample(dist.Normal(bias_mu, tensor(bias_sigma, device=self.device)).expand([nh]).to_event(1))

        if n_layers > 2:
            self.fc2 = PyroModule[nn.Linear](nh, nh)
            self.fc2.weight = PyroSample(dist.Normal(weight_mu, tensor(weight_sigma, device=self.device)).expand([nh, nh]).to_event(2))
            self.fc2.bias = PyroSample(dist.Normal(bias_mu, tensor(bias_sigma, device=self.device)).expand([nh]).to_event(1))

        if n_layers > 3:
            self.fc3 = PyroModule[nn.Linear](nh, nh)
            self.fc3.weight = PyroSample(dist.Normal(weight_mu, tensor(weight_sigma, device=self.device)).expand([nh, nh]).to_event(2))
            self.fc3.bias = PyroSample(dist.Normal(bias_mu, tensor(bias_sigma, device=self.device)).expand([nh]).to_event(1))

        if n_layers > 4:
            self.fc4 = PyroModule[nn.Linear](nh, nh)
            self.fc4.weight = PyroSample(dist.Normal(weight_mu, tensor(weight_sigma, device=self.device)).expand([nh, nh]).to_event(2))
            self.fc4.bias = PyroSample(dist.Normal(bias_mu, tensor(bias_sigma, device=self.device)).expand([nh]).to_event(1))

        self.out = PyroModule[nn.Linear](nh, nhout)
        self.out.weight = PyroSample(dist.Normal(weight_mu, tensor(weight_sigma, device=self.device)).expand([nhout, nh]).to_event(2))
        self.out.bias = PyroSample(dist.Normal(bias_mu, tensor(bias_sigma, device=self.device)).expand([nhout]).to_event(1))

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
