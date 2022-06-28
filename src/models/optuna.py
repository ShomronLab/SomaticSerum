import optuna
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split


class OptunaCV(BaseEstimator, ClassifierMixin):

    def __init__(self, estimators: list, metric, n_trials=10, direction="maximize", split_frac=0.7):
        self.estimators = estimators
        self.metric = metric
        self.n_trials = n_trials
        self.direction = direction
        self.split_frac = split_frac
        self.best_clf = None

        self._estimator_type = "classifier"

    def fit(self, X, y=None, *args, **kwargs):
        self.best_clf = OptunaCV.optimize(
            X,
            y,
            self.estimators,
            self.metric,
            self.n_trials,
            self.direction,
            self.split_frac,
            *args, **kwargs
        )
        return self.best_clf.fit(X, y, *args, **kwargs)

    def predict(self, X, y=None, *args, **kwargs):
        if self.best_clf is None:
            raise Exception("optuna hasn't been optimize!")

        return self.best_clf.predict(X, *args, **kwargs)

    def predict_proba(self, X, y=None, *args, **kwargs):
        if self.best_clf is None:
            raise Exception("optuna hasn't been optimize!")

        return self.best_clf.predict_proba(X, *args, **kwargs)

    @staticmethod
    def optimize(X, y, estimators: list, metric, n_trials=10, direction="maximize", split_frac=0.7, *args, **kwargs):
        def optuna_objective(trial):
            classifier = trial.suggest_categorical("classifier", estimators)
            params = classifier.optuna_params(trial)
            clf = classifier().set_params(**params)

            xx_train, xx_val, yy_train, yy_val = train_test_split(X, y, train_size=split_frac, shuffle=True)
            clf.fit(xx_train, yy_train)
            yy_val_pred_score = clf.predict_proba(xx_val)[:, 1]

            return metric(yy_val, yy_val_pred_score)

        study = optuna.create_study(direction=direction)
        study.optimize(optuna_objective, n_trials=n_trials, *args, **kwargs)
        print(study.best_value)

        params = study.best_params
        classifier = params.pop('classifier')
        best_clf = classifier().set_params(**params)
        return best_clf
