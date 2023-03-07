"""

Model Ensemble wrapper for uncertainty estimation. We use a XGBoost model, as it is considered as the state-of-the-art
for tabular data [1].

[1] Ravid Shwartz-Ziv and Amitai Armon, 2021, arXiv:2106.03253

Derek van Tilborg | 06-03-2023 | Eindhoven University of Technology

"""

from xgboost import XGBRegressor
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
import numpy as np
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


class AnchoredNNEnsemble:
    """ Ensemble of L2 anchored neural networks """
    def __init__(self, ensemble_size: int = 10, **kwargs) -> None:
        pass

    def predict(self, x):
        pass

    def test(self, x, y):
        pass

    def fit(self, x):
        pass

    def __repr__(self):
        return ""


class AnchoredMLP:
    def __init__(self, in_feats: int = 5, out_feats: int = 1, hidden_size: int = 64, n_layers: int = 3,
                 anchored: bool = True, seed: int = 42, l2_lambda: float = 0.001, lr=0.0005):

        self.l2_lambda = l2_lambda
        self.lr = lr
        self.train_losses = []
        self.val_losses = []
        self.epochs = 0
        self.seed = seed
        self.anchored = anchored

        self.model = MLP(in_feats=in_feats, out_feats=out_feats, hidden_size=hidden_size, n_layers=n_layers)

        # create the infrastructure needed to train the model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Save initial weights in the model for the anchored regularization and move them to the gpu
        if anchored:
            self.model.anchor_weights = deepcopy({i: j for i, j in self.model.named_parameters()})
            self.model.anchor_weights = {i: j.to(self.device) for i, j in self.model.anchor_weights.items()}

        # Move the whole model to the gpu
        self.model = self.model.to(self.device)

    # def train(self, train_loader, val_loader=None, epochs: int = 200, print_every_n: int = 2):
    #     """ Train a model for n epochs
    #
    #     :param print_every_n: (int) print progress every n epochs
    #     :param train_loader: Torch geometric data loader with training data
    #     :param val_loader: Torch geometric data loader with validation data (optional)
    #     :param epochs: (int) number of epochs to train
    #     """
    #
    #     for epoch in range(epochs):
    #         self.model.train(True)
    #         loss = self._one_epoch(train_loader)
    #         self.train_losses.append(loss)
    #
    #         val_loss = 0
    #         if val_loader is not None:
    #             val_pred, val_y = self.test(val_loader)
    #             val_loss = self.loss_fn(torch.tensor(val_pred), torch.tensor(val_y))
    #         self.val_losses.append(val_loss)
    #
    #         self.epochs += 1
    #
    #         if self.epochs % print_every_n == 0:
    #             print(f"Epoch {self.epochs} | Train Loss {loss} | Val Loss {val_loss}")

    # def _one_epoch(self, train_loader):
    #     """ Perform one pass of the train data through the model and perform backprop
    #
    #     :param train_loader: Torch geometric data loader with training data
    #     :return: loss
    #     """
    #     # Enumerate over the data
    #     for idx, batch in enumerate(train_loader):
    #
    #         # Move batch to gpu
    #         batch.to(self.device)
    #
    #         # Transform the batch of graphs with the model
    #         self.optimizer.zero_grad()
    #         self.model.train(True)
    #         y_hat = self.model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
    #
    #         # Calculating the loss and gradients
    #         loss = self.loss_fn(y_hat, batch.y)
    #
    #         if self.anchored:
    #             # Calculate the total anchored L2 loss
    #             l2_loss = 0
    #             for param_name, params in self.model.named_parameters():
    #                 anchored_param = self.model.anchor_weights[param_name]
    #
    #                 l2_loss += (self.l2_lambda/batch.num_graphs) * torch.mul(params - anchored_param,
    #                                                                          params - anchored_param).sum()
    #
    #             # Add anchored loss to regular loss according to Pearce et al. (2018)
    #             loss = loss + l2_loss
    #
    #         if not loss > 0:
    #             print(idx)
    #
    #         # Update using the gradients
    #         loss.backward()
    #         self.optimizer.step()
    #
    #     return loss
    #
    # def test(self, data_loader):
    #     """ Perform testing
    #
    #     :param data_loader:  Torch geometric data loader with test data
    #     :return: A tuple of two 1D-tensors (predicted, true)
    #     """
    #     y_pred, y = [], []
    #     with torch.no_grad():
    #         for batch in data_loader:
    #             batch.to(self.device)
    #             pred = self.model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
    #             y.extend([i for i in batch.y.tolist()])
    #             y_pred.extend([i for i in pred.tolist()])
    #
    #     return torch.tensor(y_pred), torch.tensor(y)
    #
    # def predict(self, data_loader):
    #     """ Predict bioactivity on molecular graphs
    #
    #     :param data_loader:  Torch geometric data loader with molecular graphs
    #     :return: A 1D-tensors of predicted values
    #     """
    #     y_pred = []
    #     with torch.no_grad():
    #         for batch in data_loader:
    #             batch.to(self.device)
    #             y_hat = self.model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
    #             y_pred.extend([i for i in y_hat.tolist()])
    #
    #     return torch.tensor(y_pred)

    def __repr__(self):
        return f"{self.model}"


class MLP(torch.nn.Module):
    def __init__(self, in_feats: int = 5, out_feats: int = 1, hidden_size: int = 64, n_layers: int = 3):
        super(MLP, self).__init__()

        self.in_feats = in_feats
        self.out_feats= out_feats
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.fc = torch.nn.ModuleList()
        for k in range(n_layers):
            self.fc.append(Linear(in_feats if k == 0 else hidden_size, hidden_size))
        self.lin_out = Linear(hidden_size, out_feats)

    def reset_parameters(self):
        for lin in self.fc:
            lin.reset_parameters()
        self.lin_out.reset_parameters()

    def forward(self,  x: Tensor) -> Tensor:
        for lin in self.fully_connected:
            x = F.elu(lin(x))
        x = self.lin_out(x)

        return x
