from tensorflow import keras
from model.interface.modelCollection import ModelCollection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB


class ParentModel(ModelCollection): # parent type
    def __init__(self, n_token, vec_dim, output_dim):
        super().__init__(n_token, vec_dim, output_dim)

    def mapping(self, model_name):
        if model_name == 'cnn':
            return self.cnn_model()
        elif model_name == 'lstm':
            return self.lstm_model()
        elif model_name == 'dense':
            return self.dense_model()

    def cnn_model(self):
        model = keras.models.Sequential([
            keras.layers.Conv2D(filters=12, kernel_size=(5, 5), activation='relu',
                                input_shape=(self.n_token, self.vec_dim, 1), name='conv_1'),
            keras.layers.MaxPool2D(pool_size=(1, 2)),
            keras.layers.Conv2D(filters=24, kernel_size=(5, 5), activation='relu', name='conv_2'),
            keras.layers.MaxPool2D(pool_size=(1, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(50, activation='softmax'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(self.output_dim, activation=None)])

        return model

    def lstm_model(self):
        pass

    def dense_model(self):
        pass

    def rf_model(self):
        model = RandomForestClassifier(n_estimators=300, max_depth=3)
        return model

    def svm_model(self):
        model = LinearSVC()
        return model

    def nb_model(self):
        model = MultinomialNB()
        return model