import pickle


class QuinielaModel:

    def train(self, train_data):
        # Do something here to train the model
        pass

    def predict(self, predict_data):
        # Do something here to predict
        return ["X" for _ in range(len(predict_data))]

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
