
import logging

import pandas as pd
from lightgbm import DaskLGBMClassifier

import torch
from models.base_model import BaseModel
from torch import nn


class PandasDataset(torch.utils.data.Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(
            X.values, dtype=torch.float32)
        self.y = torch.tensor(
            y.values, dtype=torch.float32).type(torch.LongTensor)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TorchModel(BaseModel):

    def __init__(self, nn_module: nn.Module, **sk_params):
        self.nn_module = nn_module
        super().__init__(**sk_params)

    def create_base(self, **sk_params):
        clf = self.nn_module
        return clf

    def fit(self, X, y=None, batch_size=32, epochs=10, learning_rate=0.01, device=None,  *args, **kwargs):
        model = self._base_clf
        loss_fn = torch.nn.CrossEntropyLoss()
        model_params = list(self._base_clf.parameters())
        optimizer = torch.optim.AdamW(
            model_params, lr=learning_rate)  # , eps=1e-08, weight_decay=0.01)

        metric = pd.DataFrame(columns=['Epoch', 'Sensitivity', 'Specificity',
                                       'PPV', 'NPV', 'FPR', 'FNR', 'FDR', 'Accuracy', 'F1', 'Class'])

        ds = PandasDataset(X, y)
        train_dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=True, num_workers=1)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for epoch in range(1, epochs+1):
            # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
            model.train()
            train_loss = 0.0
            num_train_correct = 0
            num_train_examples = 0

            for x, y in train_dl:
                optimizer.zero_grad()
                x, y = x.to(device), y.to(device)
                yhat = model(x)
                loss = loss_fn(yhat, y)

                loss.backward()
                optimizer.step()

                train_loss += loss.data.item()  # * x.size(0)
                num_train_correct += (torch.max(yhat, 1)[1] == y).sum().item()
                num_train_examples += x.shape[0]
            train_acc = num_train_correct / num_train_examples
            train_loss = train_loss / len(train_dl.dataset)

            logging.info('Epoch %3d/%3d, train loss: %5.4f, train acc: %5.4f' %
                         (epoch, epochs, train_loss, train_acc))

    def predict_proba(self, X, *args, **kwargs):
        return super().predict_proba(X.values, *args, **kwargs)

    @classmethod
    def optuna_params(cls, trial):
        return {
            'n_hidden': trial.suggest_int("n_hidden", 2, 32),
            'epochs': trial.suggest_int("epochs", 2, 32),
            'batch_size': trial.suggest_int("batch_size", 16, 64),
            'verbose': 0
        }
