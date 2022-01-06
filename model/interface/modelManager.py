from abc import ABCMeta
from data.dataManagerImpl import DataManagerImpl
from embed.bertEmbedManager import BertEmbedManager
from embed.interface.embedManager import EmbedManager
from keras.models import Model
from modelMapper import ModelMapper
import abc


class ModelManager(metaclass=ABCMeta):
    def __init__(self, feature, label, e_manager: EmbedManager, d_manager: DataManagerImpl, model: Model, model_name, ctgr):
        self.feature = feature
        self.label = label
        self.e_manager = e_manager
        self.d_manager = d_manager
        self.model: Model = model
        self.model_name = model_name
        self.ctgr = ctgr

        self.train_x = None
        self.train_y = None
        self.valid_x = None
        self.valid_y = None
        self.test_x = None
        self.test_y = None

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def evaluation(self):
        pass

    @abc.abstractmethod
    def save_model(self):
        pass

    @abc.abstractmethod
    def load_model(self):
        pass

    @abc.abstractmethod
    def get_layer_weight(self, target_layer):
        pass

    @abc.abstractmethod
    def split(self, test_ratio=0.2, valid_ratio=0.2):
        pass

    @abc.abstractmethod
    def get_train_data(self):
        pass

    @abc.abstractmethod
    def get_test_data(self):
        pass