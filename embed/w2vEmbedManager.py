from embed.interface.embedManager import EmbedManager
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from data.interface.dataManager import DataManager
from tqdm import tqdm
import gensim as gensim
import numpy as np
import nltk


class W2VEmbedManager(EmbedManager):
    def __init__(self, level, embed_name):
        super().__init__(level, embed_name)
        self.embed_dim = 300
        self.noise_idx = []
        self.title_noise_idx = []
        self.final_noise = []

    def feature_embedding(self, titles, descriptions):
        feature = []
        desc_word_list = []
        pad_word_embedding = []
        # tokenize_titles = []
        title_word_list = []
        title_word_num_list = []   # for calculateing maximum word num

        pad_title_embedding = []

        print("load word2vec model..")
        word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',
                                                                         binary=True)
        print("complete load word2vec model!")

        # --- description --- #

        for words in tqdm(descriptions):
            one_desc_word_list = []
            for one_word in words:
                if one_word in word2vec_model:
                    word_vec = word2vec_model[one_word]
                    one_desc_word_list.append(word_vec)
                else:
                    continue

            token_num = np.shape(one_desc_word_list)[0]
            if self.max_n_token < token_num:
                self.max_n_token = token_num

            desc_word_list.append(one_desc_word_list)

        print("[CHECK] max word num (description) :", self.max_n_token)
        print("word_padding..")
        for idx, w_vec in tqdm(enumerate(desc_word_list)):  # w_vec : word vector  # memory kill
            n_word = np.shape(w_vec)[0]
            if self.max_n_token > n_word:
                if n_word == 0:
                    self.noise_idx.append(idx)
                    continue
                p_vec = np.zeros(((self.max_n_token - n_word), self.embed_dim))
                c_vec = np.concatenate([w_vec, p_vec])
                pad_word_embedding.append(c_vec)

            else:
                pad_word_embedding.append(w_vec)

        # --- title --- #

        for idx, t_title in tqdm(enumerate(titles)):
            title_word_num = len(t_title)  # 하나의 title에 있는 word의 개수
            title_word_num_list.append(title_word_num)
            one_title_word_list = []
            for a_word in t_title:  # 각 단어들을 벡터화
                if a_word in word2vec_model:
                    title_word_vec = word2vec_model[a_word]
                    one_title_word_list.append(title_word_vec)
                else:
                    # self.title_noise_idx.append(idx)
                    continue
            if len(one_title_word_list) == 0:
                if idx not in self.noise_idx:
                    self.title_noise_idx.append(idx)
            title_word_list.append(one_title_word_list)

        self.max_n_title_token = max(title_word_num_list)
        print("[INFO] MAXIMUM TITLE WORD NUM : ", self.max_n_title_token)
        # max_title_word_num = 0
        for idx, w_vec in tqdm(enumerate(title_word_list)):
            num_word = np.shape(w_vec)[0]
            if self.max_n_title_token > num_word:
                temp_vec = np.zeros(((self.max_n_title_token - num_word), 300))
                pad_vec = np.concatenate([w_vec, temp_vec])
                pad_title_embedding.append(pad_vec)

            else:
                pad_title_embedding.append(w_vec)

        noise = self.noise_idx + self.title_noise_idx
        noise = set(noise)
        self.final_noise = list(noise)

        pad_title_embedding = np.array(np.delete(pad_title_embedding, self.final_noise, axis=0))
        pad_word_embedding = np.array(np.delete(title_word_list, self.final_noise, axis=0))

        instance_num = np.shape(pad_title_embedding)[0]
        for idx in range(instance_num):
            c_vec = np.concatenate((pad_title_embedding[idx], pad_word_embedding[idx]))
            print("[CHECK] c_vec shape[0] : ", np.shape(c_vec)[0])
            feature.append(c_vec)

        feature = np.array(feature)
        print("[CHECK] feature shape : ", np.shape(feature))

        return feature

    def label_embedding(self, labels): # noise 제거 안된것
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
            return label

        else:
            print("[ERROR] input a certain hierarchy level in terminal.")