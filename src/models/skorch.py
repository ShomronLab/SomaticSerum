import torch
from models.base_model import BaseModel
from skorch import NeuralNetBinaryClassifier
from torch import nn


class ClassifierModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.nn = nn.Sequential(
            nn.Linear(n_features, 1),
            nn.Sigmoid(),
            #             nn.Linear(hidden_dim, hidden_dim),
            #             nn.Sigmoid(),
            #             nn.Dropout(0.05),
            #             nn.Linear(hidden_dim, 1),
            #             nn.Sigmoid()
        )

    def forward(self, X=None, **kwargs):
        if X is None:
            return None
        X = X.reshape(-1, 1, n_features)
        X = self.nn(X)
        X = torch.reshape(X, (-1, 1)).float()
        return X


class SkorchModel(BaseModel):
    def __init__(self, nn_module: nn.Module, **sk_params):
        self.nn_module = nn_module
        super().__init__(**sk_params)

    def create_base(self, **sk_params):
        clf = NeuralNetBinaryClassifier(
            self.nn_module,
            **sk_params
        )
        return clf

    def fit(self, X, y=None, *args, **kwargs):
        return super().fit(X.values.astype('float32'), y.values.astype('float32'), *args, **kwargs)

    def predict_proba(self, X, *args, **kwargs):
        return super().predict_proba(X.values.astype('float32'), *args, **kwargs)

    @classmethod
    def optuna_params(cls, trial):
        return {
            'batch_size': trial.suggest_int("batch_size", 16, 64),
            'max_epochs': trial.suggest_int("max_epochs", 16, 64),
            'lr': trial.suggest_loguniform('lr', 1e-5, 1.),
            'momentum': trial.suggest_loguniform('momentum', 1e-5, 1.)
        }
