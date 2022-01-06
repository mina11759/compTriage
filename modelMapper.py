from model.childModel import ChildModel
from model.parentModel import ParentModel
from embed.interface.embedManager import EmbedManager
from data.interface.dataManager import DataManager


class ModelMapper:
    def __init__(self, embed_manager: EmbedManager):
        self.e_manager = embed_manager

    def get_h_model(self):
        """
        :param model_name: such as 'cnn', 'lstm'
        :return: model: Model
        """
        embed_dim = self.e_manager.max_n_title_token + self.e_manager.max_n_token

        if self.e_manager.h_level == 'parent':
            p_model = ParentModel(embed_dim, self.e_manager.embed_dim, self.e_manager.output_dim)
            return p_model
        elif self.e_manager.h_level == 'child':
            c_model = ChildModel(embed_dim, self.e_manager.embed_dim, self.e_manager.output_dim)
            return c_model
        else:
            print("[ERROR] there is no existed model.")