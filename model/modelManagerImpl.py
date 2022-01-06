import keras.models
import tensorflow as tf
from model.interface.modelManager import ModelManager
from model.interface.modelCollection import ModelCollection
from embed.interface.embedManager import EmbedManager
from data.dataManagerImpl import DataManagerImpl
from modelMapper import ModelMapper
from tensorflow.keras.models import load_model
import tensorflow.keras
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
import numpy as np
from keras.models import Model
from sklearn.model_selection import train_test_split
from metrics import MetricsAtTopK


class ModelManagerImpl(ModelManager):
    def __init__(self, feature, label, e_manager: EmbedManager, d_manager: DataManagerImpl, model: Model, model_name, ctgr):
        super().__init__(feature, label, e_manager, d_manager, model, model_name, ctgr)

    def train(self):
        self.split()
        self.model.summary()

        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['acc', MetricsAtTopK(1).recall_at_k]
        )

        print("[INFO] Embed Dimmension : ", self.e_manager.embed_dim)

        self.model.fit(
            self.train_x, self.train_y,
            validation_data=(self.valid_x, self.valid_y),
            epochs=50,
            shuffle=True
        )

        self.model.evaluate(
            self.test_x,
            self.test_y,
            verbose=2
        )

    def evaluation(self):
        pass

    def save_model(self):
        self.model.save('./output/' + self.d_manager.data_name + '_' + self.ctgr + '_' + self.e_manager.h_level + '_' +
                        self.e_manager.embed_name + '_' + self.model_name + '.h5', save_format='tf')

    def load_model(self):
        model = load_model('./output/' + self.d_manager.data_name + '_' + self.ctgr + '_' + self.e_manager.h_level + '_' +
                           self.e_manager.embed_name + '_' + self.model_name + '.h5')
        return model

    def get_layer_weight(self, target_layer_name):
        layer_name = str(target_layer_name)
        for idx, layer in enumerate(self.model.layers):
            if layer_name in layer.name:
                return idx, layer

    def split(self, test_ratio=0.4, valid_ratio=0.5):
        self.train_x, temp_x, self.train_y, temp_y = train_test_split(
            self.feature, self.label.tolist(), test_size=test_ratio)

        self.valid_x, self.test_x, self.valid_y, self.test_y = train_test_split(
            temp_x, temp_y, test_size=valid_ratio)

        self.train_x = np.array(tf.expand_dims(self.train_x, axis=-1))
        self.valid_x = np.array(tf.expand_dims(self.valid_x, axis=-1))
        self.test_x = np.array(tf.expand_dims(self.test_x, axis=-1))

        self.train_y = np.array(self.train_y)
        self.valid_y = np.array(self.valid_y)
        self.test_y = np.array(self.test_y)

        print("[CHECK] train_x shape : ", np.shape(self.train_x))
        print("[CHECK] valid_x shape : ", np.shape(self.valid_x))
        print("[CHECK] test_x shape : ", np.shape(self.test_x))
        print("[CHECK] train_y shape : ", np.shape(self.train_y))
        print("[CHECK] valid_y shape : ", np.shape(self.valid_y))
        print("[CHECK] test_y shape : ", np.shape(self.test_y))

    def get_test_data(self):
        return self.test_x, self.test_y

    def get_train_data(self):
        return self.train_x, self.train_y