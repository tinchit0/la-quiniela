import pickle
from sklearn.ensemble import GradientBoostingClassifier


class QuinielaModel:
    def __init__(self):
        self.model = GradientBoostingClassifier()

    def train(self, train_data):
        features = [
            "win_punct",
            "lost_punct",
            "points_relative_index",
            "last5_home",
            "last5_away",
            "last_season_points_home",
            "last_season_points_away",
        ]
        target = "result"
        X, y = train_data[features], train_data[target]
        self.model.fit(X, y)

    def predict(self, predict_data):
          features = [
            "win_punct",
            "lost_punct",
            "points_relative_index",
            "last5_home",
            "last5_away",
            "last_season_points_home",
            "last_season_points_away",
        ]
        predict_data = predict_data[features]
        predicted_results = self.model.predict(predict_data)
        return predicted_results

    @classmethod
    def load(cls, filename):
        """ Load model from file """
        with open(filename, "rb") as f:
            model = pickle.load(f)
            assert type(model) == cls
        return model

    def save(self, filename):
        """ Save a model in a file """
        with open(filename, "wb") as f:
            pickle.dump(self, f)
