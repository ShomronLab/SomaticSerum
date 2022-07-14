from models.base_model import BaseModel
from xgboost import XGBClassifier


class XGBModel(BaseModel):

    def create_base(self, **sk_params):
        return XGBClassifier(**sk_params)

    def fit(self, X, y=None, *args, **kwargs):
        return super().fit(X.values, y.values, *args, **kwargs)

    def predict_proba(self, X, *args, **kwargs):
        return super().predict_proba(X.values, *args, **kwargs)

    @classmethod
    def optuna_params(cls, trial):
        param = {
            "objective": "binary:logistic",
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
            "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
            "n_estimators": trial.suggest_int("n_estimators", 5, 300),
        }

        if param["booster"] == "gbtree" or param["booster"] == "dart":
            param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
            param["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
            param["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
            param["grow_policy"] = trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"])
        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical(
                "sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical(
                "normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_loguniform(
                "rate_drop", 1e-8, 1.0)
            param["skip_drop"] = trial.suggest_loguniform(
                "skip_drop", 1e-8, 1.0)
        return param
