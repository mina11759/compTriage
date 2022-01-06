from embed.interface.embedManager import EmbedManager
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from bert_embedding import BertEmbedding
import numpy as np
from tqdm import tqdm


class BertEmbedManager(EmbedManager):
    def __init__(self, h_level, embed_name):
        super().__init__(h_level, embed_name)
        self.embed_dim = 768
        self.noise_idx = []
        self.title_noise_idx = []
        self.final_noise = []

    def feature_embedding(self, titles, descriptions):
        bert_embed = BertEmbedding()
        description_embedding_list = []
        pad_embedding_list = []

        title_embedding_list = []
        pad_title_embedding_list = []

        feature = []

        print("[CHECK] 1tt", np.shape(descriptions))
        print("[CHECK] 2tt", np.shape(titles))

        print("[INFO] Description Embedding Using BERT LM..")

        # Description Embedding Process
        # 3중 for문 어떻게 못하나?

        for description in tqdm(descriptions):
            result = bert_embed(description)
            token_vector_for_desc = []
            for sentence in result:
                tokens_vectors = sentence[1]  # 하나의 sentence에 대한 token vectors
                for token_vec in tokens_vectors:
                    token_vector_for_desc.append(np.array(token_vec))

            description_embedding_list.append(token_vector_for_desc)
            token_num = np.shape(token_vector_for_desc)[0]
            if self.max_n_token < token_num:
                self.max_n_token = token_num

        for idx, description in tqdm(enumerate(description_embedding_list)): # description padding
            description = np.array(description)
            token_num = np.shape(description)[0]
            if token_num == 0:
                print("dddddddddddddddddddddddddddddddddddddddddddddddddddd")
                self.noise_idx.append(idx)
                continue
            p_vec = np.zeros((self.max_n_token - token_num, self.embed_dim)) # vector for padding
            c_vec = np.concatenate((description, p_vec))
            pad_embedding_list.append(c_vec)

        print("[CHECK] 2", np.shape(pad_embedding_list))
        # Title Embedding Process

        titles_after_bert = bert_embed(titles)
        for title in titles_after_bert:
            title_tokens = title[0]
            embedded_title_token = title[1]

            token_vector_for_title = []
            for token_vector in embedded_title_token:
                token_vector_for_title.append(np.array(token_vector))

            title_embedding_list.append(token_vector_for_title)

        print("[CHECK] 2dddddddd", np.shape(title_embedding_list))

        for title in title_embedding_list:
            token_num = np.shape(title)[0]
            if self.max_n_title_token < token_num:
                self.max_n_title_token = token_num

        for idx, title in tqdm(enumerate(title_embedding_list)): # title padding
            title = np.array(title)
            token_num = np.shape(title)[0]
            if token_num == 0:
                print("dddddddddddddddddddddddd")
                self.title_noise_idx.append(idx)
                continue
            p_vec = np.zeros((self.max_n_title_token - token_num, 768))
            c_vec = np.concatenate((title, p_vec))
            pad_title_embedding_list.append(c_vec)

        print("[CHECK] 1", np.shape(pad_title_embedding_list))
        noise = self.noise_idx + self.title_noise_idx
        noise = set(noise)
        self.final_noise = list(noise)

        print("[CHECK] 1", np.shape(pad_title_embedding_list))
        print("[CHECK] 2", np.shape(pad_embedding_list))

        pad_title_embedding_list = np.array(np.delete(pad_title_embedding_list, self.noise_idx, axis=0)) ##
        pad_embedding_list = np.array(np.delete(pad_embedding_list, self.title_noise_idx, axis=0))

        if len(pad_embedding_list) != len(pad_title_embedding_list):
            print("[ERROR] Differ from paded desciprions & titles")

        print("[CHECK] 1", np.shape(pad_title_embedding_list))
        print("[CHECK] 2", np.shape(pad_embedding_list))

        for idx in range(len(pad_title_embedding_list)): # title + description
            c_vec = np.concatenate((pad_title_embedding_list[idx], pad_embedding_list[idx]))
            feature.append(c_vec)

        feature = np.array(feature)
        print("[CHECK] feature shape : ", np.shape(feature))
        return feature

    def label_embedding(self, labels):
        if self.h_level == 'parent':
            # using One-Hot Encoding
            reshape_label = np.reshape(labels, (-1, 1))
            enc = OneHotEncoder()
            label = enc.fit_transform(reshape_label).toarray()
            label = np.delete(label, self.final_noise, axis=0)
            self.output_dim = np.shape(label)[1]
            return label

        elif self.h_level == 'child':
            # using Multi-label Encoding
            size = int(labels.size/len(labels))
            reshaped_labels = []
            for idx in range(size):
                a = np.array(labels).T[idx]
                reshaped_labels.append(a)

            reshaped_labels = np.array(reshaped_labels)

            enc = MultiLabelBinarizer(sparse_output=True)
            label = enc.fit_transform(reshaped_labels).toarray()
            label = np.delete(label, self.final_noise, axis=0)
            self.output_dim = np.shape(label)[1]
            print("[CHECK] label shape : ", np.shape(label))
            return label

        else:
            print("[ERROR] input a certain hierarchy level in terminal.")