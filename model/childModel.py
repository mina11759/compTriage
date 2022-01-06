from tensorflow import keras
from model.interface.modelCollection import ModelCollection
from keras.models import Model
from skmultilearn.adapt import MLkNN


class ChildModel(ModelCollection): # parent type
    def __init__(self, n_token, vec_dim, output_dim):
        super().__init__(n_token, vec_dim, output_dim)

    def get_layer_weight(self, model):
        layer_name = 'conv_1'
        for idx, layer in enumerate(model.layers):
            if layer_name in layer.name:
                return idx, layer

    def mapping(self, model_name):
        if model_name == 'cnn':
            return self.cnn_model()
        elif model_name == 'lstm':
            return self.lstm_model()
        elif model_name == 'dense':
            return self.dense_model()
        elif model_name == 'mlknn':
            return self.mlknn_model()
        else:
            print("[ERROR] rewrite model name in terminal")

    def cnn_model(self):
        model = keras.models.Sequential([
            keras.layers.Conv2D(filters=12, kernel_size=(5, 5), activation='relu',
                                input_shape=(self.n_token, self.vec_dim, 1), name='conv_1'),
            keras.layers.Activation('relu'),
            keras.layers.Conv2D(filters=24, kernel_size=(5, 5), activation='relu', name='conv_2'),
            keras.layers.MaxPool2D(pool_size=(1, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(50, activation='softmax'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(self.output_dim, activation=None)])

        for layer in model.layers:
            if layer.name == 'conv_1':
                target_idx, target_layer = self.get_layer_weight(model)
                layer.set_weights(model.layers[target_idx].get_weights())

        return model

    def lstm_model(self):
        pass

    def dense_model(self):
        pass

    def mlknn_model(self):
        model = MLkNN(k=3)
        return model