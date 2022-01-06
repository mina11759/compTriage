from abc import ABCMeta
import abc
from embed.interface.embedManager import EmbedManager


class DataManager(metaclass=ABCMeta):
    def __init__(self, data_name, t_level, category, e_manager: EmbedManager):
        self.data_name = data_name
        self.t_level = t_level # sentence, word
        self.e_manager = e_manager
        self.ctgr = category

        self.max_limit_token = None
        self.min_limit_token = None
        self.max_n_token = None # 확인용

    @abc.abstractmethod
    def clean_data(self, data):
        pass

    @abc.abstractmethod
    def load_data(self):
        pass