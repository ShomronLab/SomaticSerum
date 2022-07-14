from abc import ABCMeta

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class BaseModel(BaseEstimator, ClassifierMixin, metaclass=ABCMeta):

    def __init__(self, **sk_params):
        self._base_clf = self.create_base(**sk_params)
        self._base_clf._estimator_type = "classifier"
        self._estimator_type = "classifier"
        self.classes_ = [0, 1]

    def create_base(self, **sk_params):
        raise NotImplementedError

    @property
    def feature_importances_(self):
        return self._base_clf.feature_importances_

    def feature_importance(self, cols=None):
        if cols is None:
            return self.feature_importances_
        else:
            feature_importances = pd.Series(
                data=self.feature_importances_,
                index=cols
            )
            return feature_importances.sort_values()

    @classmethod
    def optuna_params(cls, trial):
        raise NotImplementedError

    def set_params(self, **params):
        clf_params = {k: v for k, v in params.items() if k in self._clf_params()}
        self._base_clf.set_params(**clf_params)

        for param, value in params.items():
            if param in self._cls_params():
                setattr(self, param, value)

        return self

    def get_params(self, deep=True):
        clf_params = self._base_clf.get_params()
        cls_params = {param: getattr(self, param) for param in self._cls_params()}
        return {**cls_params, **clf_params}

    def _cls_params(self):
        params = type(self).__init__.__code__.co_varnames
        return params[1:-1]

    def _clf_params(self):
        return self._base_clf.get_params()

    def fit(self, X, y=None, *args, **kwargs):
        return self._base_clf.fit(X, y, *args, **kwargs)

    def predict_proba(self, X, y=None, *args, **kwargs):
        return self._base_clf.predict_proba(X, *args, **kwargs)

    def predict(self, X, y=None, *args, **kwargs):
        return (self.predict_proba(X, y=None, *args, **kwargs)[:, 1] > 0.5).astype(int)
