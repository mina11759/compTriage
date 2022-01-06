from tensorflow import keras
import abc
from abc import ABCMeta


class ModelCollection(metaclass=ABCMeta): # parent, child
    def __init__(self, n_token, vec_dim, output_dim):
        self.n_token = n_token
        self.vec_dim = vec_dim
        self.output_dim = output_dim

    @abc.abstractmethod
    def mapping(self, model_name):
        pass

    @abc.abstractmethod
    def cnn_model(self):
        pass

    @abc.abstractmethod
    def lstm_model(self):
        pass

    @abc.abstractmethod
    def dense_model(self):
        pass