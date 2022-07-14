from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from models.base_model import BaseModel


def custom_f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = TP / (Positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = TP / (Pred_Positives + K.epsilon())
        return precision

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class KerasModel(BaseModel):
    metric = custom_f1

    def __init__(self, n_hidden=10, **sk_params):
        super().__init__(**sk_params)
        self.n_hidden = n_hidden

    def create_base(self, **sk_params):
        def create_model():
            # create model
            model = Sequential()
            model.add(Dense(self.n_hidden, activation='relu'))
            #     model.add(Dense(8, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            # Compile model
            model.compile(loss='binary_crossentropy',
                          optimizer='adam', metrics=[custom_f1])
            return model

        # create model
        clf = KerasClassifier(build_fn=create_model, **sk_params)
        return clf

    def fit(self, X, y=None, *args, **kwargs):
        return super(KerasModel, self).fit(X.values, y.values, *args, **kwargs)

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
