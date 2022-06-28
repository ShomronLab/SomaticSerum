from lightgbm import LGBMClassifier

from models.base_model import BaseModel


class LGBMModel(BaseModel):

    def create_base(self, **sk_params):
        return LGBMClassifier(**sk_params)

    def fit(self, X, y=None, *args, **kwargs):
        return super().fit(X.values, y.values, *args, **kwargs)

    def predict_proba(self, X, *args, **kwargs):
        return super().predict_proba(X.values, *args, **kwargs)

    @classmethod
    def optuna_params(cls, trial):
        return {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "n_estimators": trial.suggest_int("n_estimators", 5, 300),
            "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
            "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }
