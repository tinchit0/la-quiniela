import pickle
from sklearn.ensemble import RandomForestRegressor

class QuinielaModel:

    def __init__(self):
        self.model = RandomForestRegressor(n_estimators = 100)

    def train(self, train_data):
        x_train = train_data[['home_team_num', 'away_team_num', 'home_team_rank', 'away_team_rank']]
        y_train = train_data['winner']
        self.model.fit(x_train,y_train)
        


    def predict(self, predict_data):
        # Do something here to predict
        x_predict = predict_data[['home_team_num', 'away_team_num', 'home_team_rank', 'away_team_rank']]
        prediction = self.model.predict(x_predict).astype(int)
        return prediction 

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
