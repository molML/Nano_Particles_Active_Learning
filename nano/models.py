"""

Model Ensemble wrapper for uncertainty estimation. We use a XGBoost model, as it is considered as the state-of-the-art
for tabular data [1].

[1] Ravid Shwartz-Ziv and Amitai Armon, 2021, arXiv:2106.03253

Derek van Tilborg | 06-03-2023 | Eindhoven University of Technology

"""

import numpy as np
import torch
import torch.nn as nn
from torch import tensor
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
from tqdm.auto import trange
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
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


class NN(PyroModule):
    def __init__(self, in_feats=5, n_layers: int = 3, hidden_size: int = 32, activation: str = 'relu',
                 to_gpu: bool = True):
        super().__init__()
        nh0, nh = in_feats, hidden_size
        self.n_layers = n_layers
        assert 1 <= n_layers <= 5, f"'n_layers' should be between 1 and 5, not {n_layers}"
        assert activation in ['relu', 'gelu', 'elu', 'leaky']

        self.device = torch.device("cuda:0" if torch.cuda.is_available() and to_gpu else "cpu")

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

        self.out = PyroModule[nn.Linear](nh, 1)
        self.out.weight = PyroSample(dist.Normal(0., tensor(1.0, device=self.device)).expand([1, nh]).to_event(2))
        self.out.bias = PyroSample(dist.Normal(0., tensor(1.0, device=self.device)).expand([1]).to_event(1))

        if activation == 'relu':
            self.act = nn.ReLU()
        if activation == 'elu':
            self.act = nn.ELU()
        if activation == 'gelu':
            self.act = nn.GELU()
        if activation == 'leaky':
            self.act = nn.LeakyReLU()

    def forward(self, x, y=None):

        x = self.act(self.fc0(x))
        if self.n_layers > 1:
            x = self.act(self.fc1(x))
        if self.n_layers > 2:
            x = self.act(self.fc2(x))
        if self.n_layers > 3:
            x = self.act(self.fc3(x))
        if self.n_layers > 4:
            x = self.act(self.fc4(x))
        mu = self.out(x).squeeze()

        sigma = pyro.sample("sigma", dist.Uniform(0., tensor(1.0, device=self.device)))
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
        return mu


class BayesianNN:
    def __init__(self, in_feats: int = 5, hidden_size: int = 64, n_layers: int = 3, activation: str = 'relu',
                 seed: int = 42, lr=1e-3, to_gpu: bool = False, epochs: int = 10000):

        self.lr = lr
        self.train_losses = []
        self.epochs = epochs
        self.epoch = 0
        self.seed = seed
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and to_gpu else "cpu")
        pyro.set_rng_seed(seed)

        # Init model
        self.model = NN(in_feats=in_feats, hidden_size=hidden_size, activation=activation, n_layers=n_layers,
                        to_gpu=to_gpu)
        self.model = self.model.to(self.device)

        # Init Guide model
        self.guide = AutoDiagonalNormal(self.model)
        self.guide = self.guide.to(self.device)
        # Init optimizer
        adam = pyro.optim.Adam({"lr": lr})
        # Stochastic Variational Inference
        self.svi = SVI(self.model, self.guide, adam, loss=Trace_ELBO())

    def train(self, data_loader, epochs: int = None):

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

    def test(self):
        raise NotImplementedError

    def predict(self, data_loader, num_samples: int = 500, return_posterior: bool = False):

        predictive = Predictive(self.model, guide=self.guide, num_samples=num_samples, return_sites=("obs", "_RETURN"))

        predictions = {'y_hat': {'pred': tensor([], device=self.device), 'mean': tensor([], device=self.device),
                                 'std': tensor([], device=self.device)},
                       'posterior': {'pred': tensor([], device=self.device), 'mean': tensor([], device=self.device),
                                     'std': tensor([], device=self.device)}}

        for batch in data_loader:

            x = batch[0].to(self.device)
            samples = predictive(x.unsqueeze(0) if len(x.size()) == 1 else x)
            n = len(samples['obs'])
            if len(samples['_RETURN'].size()) == 1 and return_posterior:
                samples['_RETURN'] = samples['_RETURN'].reshape([n, 1])

            predictions['y_hat']['pred'] = torch.cat((predictions['y_hat']['pred'], samples['obs'].T), 0)
            y_hat_mean = torch.mean(samples['obs'], dim=0)
            predictions['y_hat']['mean'] = torch.cat((predictions['y_hat']['mean'], y_hat_mean), 0)
            y_hat_std = torch.std(samples['obs'], dim=0)
            predictions['y_hat']['std'] = torch.cat((predictions['y_hat']['std'], y_hat_std), 0)

            if return_posterior:
                predictions['posterior']['pred'] = torch.cat((predictions['posterior']['pred'], samples['obs'].T), 0)
                post_mean = torch.mean(samples['_RETURN'], dim=0)
                predictions['posterior']['mean'] = torch.cat((predictions['posterior']['mean'], post_mean), 0)
                post_std = torch.std(samples['_RETURN'], dim=0)
                predictions['posterior']['std'] = torch.cat((predictions['posterior']['std'], post_std), 0)

        if return_posterior:
            return predictions['y_hat']['pred'], predictions['y_hat']['mean'], predictions['y_hat']['std'], \
                   predictions['posterior']['pred'], predictions['posterior']['mean'], predictions['posterior']['std']

        return predictions['y_hat']['pred'], predictions['y_hat']['mean'], predictions['y_hat']['std']

# y_hat_5 = samples['obs'].kthvalue(int(n * 0.05), dim=0)[0]
# predictions['y_hat']['5%'] = torch.cat((predictions['y_hat']['5%'], y_hat_5), 0)
# y_hat_95 = samples['obs'].kthvalue(int(n * 0.95), dim=0)[0]
# predictions['y_hat']['95%'] = torch.cat((predictions['y_hat']['95%'], y_hat_95), 0)
