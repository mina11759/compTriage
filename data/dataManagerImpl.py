from data.interface.dataManager import DataManager #
from embed.interface.embedManager import EmbedManager
import csv, re, string
import numpy as np
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer


class DataManagerImpl(DataManager):
    def __init__(self, data_name, t_level, category, e_manager: EmbedManager):
        super().__init__(data_name, t_level, category, e_manager)

    def clean_data(self, data):
        # 1. Remove \r
        current_description = data.replace("\r", " ")
        current_description = current_description.replace("\n", " ")
        # print('curr1 : ', current_description)

        # 2. Remove URLs
        current_description = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "",
            current_description
        )
        # print('curr2 : ', current_description)

        # 3. Remove stack trace
        stack_trace = [
            "Stack trace:", "stack:", "Backtrace", "Trace:", "stack trace:",
            "calltrace:", "<!doctype", "<!DOCTYPE", "gdb info:",
            "<HTML>", "INFO", "chrome", "WARNING:", "Results:", '"'
        ]
        for st in stack_trace:
            start_loc = current_description.find(st)
            current_description = current_description[:start_loc]

        # 4. Remove hex code
        current_description = re.sub(r"(\w+)0x\w", "", current_description)

        # 5. change to lower case
        current_description = current_description.lower()

        if self.t_level == 'sentence':
            # 6. Sentence Tokenize
            current_description_tokens = sent_tokenize(current_description)

            # 7. strip trailing puctuation marks
            current_description_filter = [word.strip(string.punctuation) for word in current_description_tokens]

            return current_description_filter

        elif self.t_level == 'word':
            current_description_tokens = word_tokenize(current_description)
            stop_words = set(stopwords.words('english'))
            current_description_tokens_stopwords = []
            for word in current_description_tokens:
                if word not in stop_words:
                    current_description_tokens_stopwords.append(word)

            curr_description_filter = [
                word.strip(string.punctuation) for word in current_description_tokens_stopwords
            ]
            curr_data = [x for x in curr_description_filter if x]

            return curr_data

        else:
            print("[ERROR] Rewrite token level option !")

    def load_data(self):
        """
        :return: feature, label
        """
        file_name = './data/' + self.data_name + '_' + self.e_manager.h_level + '_' + self.ctgr + '.csv' # 좀더 세분화?
        print("[INFO] load data..")

        token_num_list = []
        titles = []
        descriptions = []  # feature
        components = []  # 추후에 label로 사용
        description = []  # 정제된 description
        components_1 = []
        components_2 = []
        components_3 = []  # 하위 component들

        if self.t_level == 'sentence':
            self.min_limit_token, self.max_limit_token = 1, 50
        else:
            self.min_limit_token, self.max_limit_token = 15, 250

        with open(file_name, 'rt', encoding='latin_1') as file:
            rdr = csv.DictReader(file)
            for line in tqdm(rdr):
                for key, value in line.items():
                    if key == 'title':
                        titles.append(value)
                        # print('title : ', titles)

                    elif key == 'description':
                        description.append(value)
                        # print('description : ', description)
                        for token in description:
                            cleaned_data = self.clean_data(token)
                        num_token = len(cleaned_data)
                        if num_token == 1:
                            continue
                        elif num_token > self.max_limit_token or num_token < self.min_limit_token:
                            continue

                    elif key == 'top_component':
                        components.append(value)
                    elif key == 'component_1':
                        components_1.append(value)
                    elif key == 'component_2':
                        components_2.append(value)
                    elif key == 'component_3':
                        components_3.append(value)
                        # print('component : ', components)

                # 정제시킨 description append
                descriptions.append(cleaned_data)
                token_num_list.append(num_token)
            self.max_n_token = max(token_num_list)
            # ----------------------------------------------------------------------#
            print('[INFO] max token number : ', self.max_n_token) # Sentence or Word

            feature = self.e_manager.feature_embedding(titles, descriptions)
            if self.e_manager.h_level == 'parent':
                label = self.e_manager.label_embedding(components)
                return feature, label

            else:
                labels = np.array([components_1, components_2, components_3])
                label = self.e_manager.label_embedding(labels)
                return feature, label