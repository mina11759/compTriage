"""
https://bugtriage.mybluemix.net/
github.com/imgarylai/bert-embedding

1. Description split
    - description의 1차 정제 : 불용어, \n, url, stack trace 등등
    - 정제된 description은 여전히 여러 개의 문장으로 구성됨.
      nltk 라이브러리의 sent_tokenize를 활용하여 split (문장 단위로 split하는 함수)
    - data 정제 : 정규표현식 re를 사용하면 빠르게 정제 가능
2. Padding
    - description들의 문장 개수가 달라서 동일하게 맞춰줘야함
    - 최대 문장 개수가 몇개인지 알고, 그 개수만큼 padding --> max_sent_num에 저장됨
    - padding token? ex) <pad>, zero-padding..

"""

import csv, re, string
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from nltk.tokenize import sent_tokenize
from sklearn.preprocessing import MultiLabelBinarizer
from bert_embedding import BertEmbedding
from tqdm import tqdm


# title은 정제할 필요가 없음 ---> 없나,,? title에도 대문자 있음
# description에 대해서만 정제
def clean_description(v):
    # 1. Remove \r
    current_description = v.replace("\r", " ")
    current_description = current_description.replace("\n", " ")
    #print('curr1 : ', current_description)

    # 2. Remove URLs
    current_description = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "",
        current_description
    )
    #print('curr2 : ', current_description)

    # 3. Remove stack trace
    stack_trace = [
        "Stack trace:", "stack:", "Backtrace", "Trace:", "stack trace:",
        "calltrace:", "<!doctype", "<!DOCTYPE", "gdb info:",
        "<HTML>", "INFO", "chrome", "WARNING:", "Results:", '"'
    ]
    for st in stack_trace:
        start_loc = current_description.find(st)
        current_description = current_description[:start_loc]
    #print('curr3 : ', current_description)

    # 4. Remove hex code
    current_description = re.sub(r"(\w+)0x\w", "", current_description)
    #print('curr4 : ', current_description)

    # 5. change to lower case
    current_description = current_description.lower()
    #print('curr5 : ', current_description)

    # 6. Sentence Tokenize
    current_description_tokens = sent_tokenize(current_description)
    #print('curr6 : ', current_description_tokens)

    # 7. strip trailing puctuation marks
    current_description_filter = [word.strip(string.punctuation) for word in current_description_tokens]
    #print('curr7 : ', current_description_filter)

    # 8. Join the lists
    current_data = current_description_filter
    #print('curr_data : ', current_data)

    return current_data


# padding : 문장 개수 맞춰주기
def padding(description, max_sent_num):
    print("--- In padding phase ---")
    for item in description:
        while len(item) < max_sent_num:
            item.append(0)
    final_descriptions = np.array(description)
    return final_descriptions


#component - one hot vector 이용
def one_hot_vector(component):
    print("--- In component reshape phase ---")
    reshaped_component = np.reshape(component, (-1, 1))
    enc = OneHotEncoder()
    one_hot = enc.fit_transform(reshaped_component).toarray() # train에서는 fit_transform 사용
    #print('changed components : ', one_hot)
    return one_hot


# 하위 component
def bottom_components(component1, component2, component3):
    component = np.array([component1, component2, component3])
    size = int(component.size/len(component))
    reshaped_component = []
    for i in range(size):
        a = np.array(component).T[i]
        reshaped_component.append(a)

    reshaped_component = np.array(reshaped_component)

    enc = MultiLabelBinarizer(sparse_output=True)
    one_hot = enc.fit_transform(reshaped_component).toarray()

    return one_hot


def prepare_dataset(dataset, min_sent=1, max_sent=50):
    print("--- Start loading data ---")

    sent_num_list = []
    titles = []
    descriptions = [] # feature
    components = [] # 추후에 label로 사용
    description = [] # 정제된 description

    with open(dataset, 'rt', encoding='latin_1') as file: #FCREPO_Sort_top
    #with open(dataset, 'rt', encoding='cp949') as file: #Qpid_Integration_top

        rdr = csv.DictReader(file)
        for line in tqdm(rdr):
            for k, v in line.items():
                if k == 'title':
                    titles.append(v)
                    #print('title : ', titles)

                elif k == 'description':
                    description.append(v)
                    #print('description : ', description)
                    for v in description:
                        cleaned_data = clean_description(v)
                    num_sentence = len(cleaned_data)
                    if num_sentence == 1:
                        continue
                    elif num_sentence > max_sent or num_sentence < min_sent:
                        continue

                elif k == 'top_component':
                    components.append(v)
                    #print('component : ', components)

            # 정제시킨 description append
            descriptions.append(cleaned_data)
            sent_num_list.append(num_sentence)
        max_sent_num = max(sent_num_list)
 # ----------------------------------------------------------------------#
        print('max sentence number : ', max_sent_num)
        # new_data = padding(descriptions, max_sent_num)
        label = one_hot_vector(components)

        bert_embedding = BertEmbedding()
        embedding_list = []
        print("description embedding..")
        for description in tqdm(descriptions):
            result = bert_embedding(description)
            token_vector_for_desc = []
            for sentence in result:
                tokens_weight = sentence[1] # 하나의 sentence에 대한 token vectors
                for token_vec in tokens_weight:
                    token_vector_for_desc.append(np.array(token_vec))

            embedding_list.append(token_vector_for_desc)

        max_token_num = 0
        for description in embedding_list:
            token_num = np.shape(description)[0]
            if max_token_num < token_num:
                max_token_num = token_num

    # ------------------------------------------------------------------- #

        print("[INFO] Max_token_num for description : ", max_token_num)

        # padding
        pad_embedding_list = []
        noise_idx = [] # token 개수 0인 description filtering
        for index, description in tqdm(enumerate(embedding_list)):
            np_description = np.array(description)
            token_num = np.shape(description)[0]
            if token_num == 0:
                noise_idx.append(index)
                continue
            pad_vec = np.zeros((max_token_num - token_num, 768))
            c_vec = np.concatenate((np_description, pad_vec))
            pad_embedding_list.append(c_vec)


        print("np.shape(pad_embedding_list): ", np.shape(pad_embedding_list))
        print("noise :", noise_idx)

        pad_embedding_list = np.array(pad_embedding_list)
# ============================================================================== #
        title_embedding_list = []
        result = bert_embedding(titles)
        print(len(result))
        for title in result:
            title_tokens = title[0]
            title_token_weights = title[1]

            token_vec_list = [] # element: np
            for token_weight in title_token_weights:
                token_vec_list.append(np.array(token_weight))

            title_embedding_list.append(token_vec_list)

        max_title_token_num = 0
        for title in title_embedding_list:
            token_num = np.shape(title)[0]
            if max_title_token_num < token_num:
                max_title_token_num = token_num

        print("[INFO] max_title_token_num : ", max_title_token_num)

        # padding
        pad_title_embedding_list = [] # list(np)
        # noise_idx = [] # token 개수 0인 description filtering
        print("title padding..")
        for index, title in tqdm(enumerate(title_embedding_list)):
            np_title = np.array(title)
            token_num = np.shape(title)[0]
            if token_num == 0:
                # noise_idx.append(index)
                continue
            pad_vec = np.zeros((max_title_token_num - token_num, 768))
            c_vec = np.concatenate((np_title, pad_vec))
            pad_title_embedding_list.append(c_vec)

        pad_title_embedding_list = np.array(np.delete(pad_title_embedding_list, noise_idx, axis=0))

        # check
        if len(pad_title_embedding_list) != len(pad_embedding_list):
            print(np.shape(pad_title_embedding_list))
            print(np.shape(pad_embedding_list))
            print(len(pad_title_embedding_list))
            print(len(pad_embedding_list))
            print("길이가 다릅니다")

        feature = []
        for i in range(len(pad_title_embedding_list)):
            c_vec = np.concatenate((pad_title_embedding_list[i], pad_embedding_list[i]))
            # print("[INFO] c_vec = {0} + {1}".format(len(pad_title_embedding_list[i]), len(pad_embedding_list[i])))
            feature.append(c_vec)

        feature = np.array(feature)
        label = np.delete(label, noise_idx, axis=0)

        print("[INFO] feature shape : ", np.shape(feature))
        print("[INFO] label shape : ", np.shape(label))

        return feature, label


def bottom_prepare_dataset(dataset, min_sent=1, max_sent=50):
    print("--- Start loading data ---")

    sent_num_list = []
    titles = []
    descriptions = [] # feature
    components = [] # 추후에 label로 사용
    description = [] # 정제된 description
    components_1 = []
    components_2 = []
    components_3 = [] # 하위 component들

    with open(dataset, 'rt', encoding='latin_1') as file: #FCREPO_Sort_top
        rdr = csv.DictReader(file)
        for line in tqdm(rdr):
            for k, v in line.items():
                if k == 'title':
                    titles.append(v)

                elif k == 'description':
                    description.append(v)
                    for v in description:
                        cleaned_data = clean_description(v)
                        # print(cleaned_data)
                    num_sentence = len(cleaned_data)
                    if num_sentence == 1:
                        continue
                    elif num_sentence > max_sent or num_sentence < min_sent:
                        continue

                # top component
                elif k == 'top_component':
                    components.append(v)

                # 하위 component
                elif k == 'component_1':
                    components_1.append(v)
                elif k == 'component_2':
                    components_2.append(v)
                elif k == 'component_3':
                    components_3.append(v)

            # 정제시킨 description append
            # print(cleaned_data)
            descriptions.append(cleaned_data)
            sent_num_list.append(num_sentence)
        max_sent_num = max(sent_num_list)

        print('max sentence number : ', max_sent_num)
        # new_data = padding(descriptions, max_sent_num)
        if np.shape(components)[0] != 0:
            label = one_hot_vector(components) # [FIND]

        bert_embedding = BertEmbedding()
        embedding_list = []
        print("description embedding..")
        for description in tqdm(descriptions):
            result = bert_embedding(description)
            token_vector_for_desc = []
            for sentence in result:
                tokens_weight = sentence[1] # 하나의 sentence에 대한 token vectors
                for token_vec in tokens_weight:
                    token_vector_for_desc.append(np.array(token_vec))

            embedding_list.append(token_vector_for_desc)

        max_token_num = 0
        for description in embedding_list:
            token_num = np.shape(description)[0]
            if max_token_num < token_num:
                max_token_num = token_num

        print("[INFO] Max_token_num for description : ", max_token_num)

        # padding
        pad_embedding_list = []
        noise_idx = [] # token 개수 0인 description filtering
        for index, description in tqdm(enumerate(embedding_list)):
            np_description = np.array(description)
            token_num = np.shape(description)[0]
            if token_num == 0:
                noise_idx.append(index)
                continue
            pad_vec = np.zeros((max_token_num - token_num, 768))
            c_vec = np.concatenate((np_description, pad_vec))
            pad_embedding_list.append(c_vec)

        print("np.shape(pad_embedding_list): ", np.shape(pad_embedding_list))
        print("noise :", noise_idx)

        pad_embedding_list = np.array(pad_embedding_list)

        title_embedding_list = []
        result = bert_embedding(titles)
        print(len(result))
        for title in result:
            title_tokens = title[0]
            title_token_weights = title[1]

            token_vec_list = [] # element: np
            for token_weight in title_token_weights:
                token_vec_list.append(np.array(token_weight))

            title_embedding_list.append(token_vec_list)

        max_title_token_num = 0
        for title in title_embedding_list:
            token_num = np.shape(title)[0]
            if max_title_token_num < token_num:
                max_title_token_num = token_num

        print("[INFO] max_title_token_num : ", max_title_token_num)

        # padding
        pad_title_embedding_list = [] # list(np)
        # noise_idx = [] # token 개수 0인 description filtering
        print("title padding..")
        for index, title in tqdm(enumerate(title_embedding_list)):
            np_title = np.array(title)
            token_num = np.shape(title)[0]
            if token_num == 0:
                # noise_idx.append(index)
                continue
            pad_vec = np.zeros((max_title_token_num - token_num, 768))
            c_vec = np.concatenate((np_title, pad_vec))
            pad_title_embedding_list.append(c_vec)

        pad_title_embedding_list = np.array(np.delete(pad_title_embedding_list, noise_idx, axis=0))

        # check
        if len(pad_title_embedding_list) != len(pad_embedding_list):
            print(np.shape(pad_title_embedding_list))
            print(np.shape(pad_embedding_list))
            print(len(pad_title_embedding_list))
            print(len(pad_embedding_list))
            print("길이가 다릅니다")

        feature = []
        for i in range(len(pad_title_embedding_list)):
            c_vec = np.concatenate((pad_title_embedding_list[i], pad_embedding_list[i]))
            # print("[INFO] c_vec = {0} + {1}".format(len(pad_title_embedding_list[i]), len(pad_embedding_list[i])))
            feature.append(c_vec)

        feature = np.array(feature)
        # label = np.delete(label, noise_idx, axis=0) # [FIND]

        # 하위 component label
        sublabel = bottom_components(components_1, components_2, components_3)
        sublabel = np.delete(sublabel, noise_idx, axis=0)

        print("[INFO] feature shape : ", np.shape(feature))
        # print("[INFO] label shape : ", np.shape(label))
        print("[INFO] sublabel shape : ", np.shape(sublabel))

        return feature, sublabel



