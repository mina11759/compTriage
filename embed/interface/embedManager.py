from abc import ABCMeta


class EmbedManager(metaclass=ABCMeta):
    def __init__(self, h_level, embed_name):
        self.embed_name = embed_name
        self.h_level = h_level
        self.max_n_token = 0
        self.max_n_title_token = 0
        self.output_dim = 0

    def feature_embedding(self, titles, descriptions):
        pass

    def label_embedding(self, labels):
        pass