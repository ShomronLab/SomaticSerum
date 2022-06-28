from sklearn.linear_model import LogisticRegression

from models.base_model import BaseModel


class LogisticModel(BaseModel):

    def create_base(self, **sk_params):
        return LogisticRegression(**sk_params)

    @classmethod
    def optuna_params(cls, trial):
        return {
            'C': trial.suggest_loguniform('C', 1e-5, 1.)
        }

    @property
    def feature_importances_(self):
        return self._base_clf.coef_.reshape(-1)
