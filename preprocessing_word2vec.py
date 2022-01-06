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
from nltk.corpus import stopwords
import nltk
import gensim as gensim
from nltk.tokenize import sent_tokenize
from sklearn.preprocessing import MultiLabelBinarizer
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
    for rm_log in stack_trace:
        start_loc = current_description.find(rm_log)
        # current_desc.strip(' '+current_desc+' ')
        current_description = current_description[:start_loc]

    # 4. Remove hex code
    current_desc = re.sub(r"(\w+)0x\w+", "", current_description)

    # 5. Change to lower case
    current_desc = current_desc.lower()

    # 6. Tokenize
    current_desc_tokens = nltk.word_tokenize(current_desc)
    # 7. Stopword
    stop_words = set(stopwords.words('english'))
    current_desc_tokens_stopwords = []
    for word in current_desc_tokens:
        if word not in stop_words:
            current_desc_tokens_stopwords.append(word)

    # 7. Strip trailing punctuation marks
    current_desc_filter = [
        word.strip(string.punctuation) for word in current_desc_tokens_stopwords
    ]

    # 8. Join the lists
    current_data = current_desc_filter
    current_data = [x for x in current_data if x]  # list(filter(None, current_data))

    return current_data


#component - one hot vector 이용
def one_hot_vector(component):
    print("--- In component reshape phase ---")
    reshaped_component = np.reshape(component, (-1, 1))
    enc = OneHotEncoder()
    one_hot = enc.fit_transform(reshaped_component).toarray() # train에서는 fit_transform 사용
    #print('changed components : ', one_hot)
    return one_hot

def bottom_components(component1, component2, component3):

    component = np.array([component1, component2, component3])
    size = int(component.size/len(component))
    reshaped_component = []
    for i in range(size):
        a = np.array(component).T[i]
        reshaped_component.append(a)

    reshaped_component = np.array(reshaped_component)
    #print('shape : ', reshaped_component.shape)
    #print('reshaped component : ', reshaped_component)

    enc = MultiLabelBinarizer(sparse_output=True)
    one_hot = enc.fit_transform(reshaped_component).toarray()
    labels = enc.classes_

    #print('labels : ', labels)
    #print(' one hot : ', one_hot)

    return one_hot

def word_prepare_dataset(dataset, min_word=15, max_word=250):
    print("Start Load data for word2vec..")

    word_num_list = []
    titles = []
    descriptions = [] # feature
    components = [] # 추후에 label로 사용
    description = [] # 정제된 description

    with open(dataset, 'rt', encoding='latin_1') as file: #FCREPO_Sort_top

 #---------------------------------------------------------------#
        rdr = csv.DictReader(file)
        for line in tqdm(rdr):
            for k, v in line.items():
                if k == 'title':
                    titles.append(v)

                elif k == 'description':
                    description.append(v)
                    for v in description:
                        cleaned_data = clean_description(v)
                    num_word = len(cleaned_data)
                    if num_word == 1:
                        continue
                    elif num_word > max_word or num_word < min_word:
                        continue

                elif k == 'top_component':
                    components.append(v)

            descriptions.append(cleaned_data)
            word_num_list.append(num_word)

        print("[INFO] description num : ", len(descriptions))
        print("[INFO] title num : ", len(titles))

        # print(word_num_list)
        max_word_num = max(word_num_list)
 # ---------------------------------------------------------------#
        print("load word2vec model..")
        word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',
                                                                     binary=True)
        print("complete load word2vec model!")
        desc_word_list = []
        for words in tqdm(descriptions):
            one_desc_word_list = []
            for one_word in words:
                if one_word in word2vec_model:
                    word_vec = word2vec_model[one_word]
                    one_desc_word_list.append(word_vec)
                else:
                    continue

            desc_word_list.append(one_desc_word_list)
 #
        print("word_padding..")

        word_noise = []
        pad_word_embedding = []
        for idx, w_vec in tqdm(enumerate(desc_word_list)):  # w_vec : word vector  # memory kill
            num_word = np.shape(w_vec)[0]
            if max_word_num > num_word:
                if num_word == 0:
                    word_noise.append(idx)
                    continue
                temp_vec = np.zeros(((max_word_num - num_word), 300))
                pad_vec = np.concatenate([w_vec, temp_vec])
                pad_word_embedding.append(pad_vec)

            else:
                pad_word_embedding.append(w_vec)

        # print("dddddddddddddd", np.shape(pad_word_embedding)[0])
        # description은 알아서 noise를 거름
        print("np.shape(pad_word_list) before deleting noise index : ", np.shape(pad_word_embedding))
        # pad_word_embedding = np.array(np.delete(pad_word_embedding, word_noise, axis=0))
        # print("np.shape(pad_word_list) after deleting noise index : ", np.shape(pad_word_embedding))

        print("word noise idx !")
        print(word_noise)

        tokenize_titles = []
        for title in titles:
            curr_title = nltk.word_tokenize(title)
            tokenize_titles.append(curr_title)

    # -----------------------------------------------------------------------------------------#

        title_word_list = []
        title_noise = []
        title_word_num_list = [] # for calculateing maximum word num
        for idx, t_title in tqdm(enumerate(tokenize_titles)):
            title_word_num = len(t_title) # 하나의 title에 있는 word의 개수
            title_word_num_list.append(title_word_num)
            one_title_word_list = []
            for a_word in t_title: # 각 단어들을 벡터화
                if a_word in word2vec_model:
                    title_word_vec = word2vec_model[a_word]
                    one_title_word_list.append(title_word_vec)
                else:
                    continue
            if len(one_title_word_list) == 0:
                if not idx in word_noise:
                    word_noise.append(idx)
            title_word_list.append(one_title_word_list)

        title_word_list = np.array(np.delete(title_word_list, word_noise, axis=0))  # error

        max_title_word_num = max(title_word_num_list)
        print("[INFO] MAXIMUM TITLE WORD NUM : ", max_title_word_num)
        pad_title_embedding = []
        # max_title_word_num = 0
        for idx, w_vec in tqdm(enumerate(title_word_list)):  # w_vec : word vector  # memory kill
            num_word = np.shape(w_vec)[0]
            if max_title_word_num > num_word:
                temp_vec = np.zeros(((max_title_word_num - num_word), 300))
                pad_vec = np.concatenate([w_vec, temp_vec])
                pad_title_embedding.append(pad_vec)
                # print(np.shape(pad_vec))

            else:
                pad_title_embedding.append(w_vec)

# ------------------------------------------------------------------------------------------#

        print("[TEST] max")

        print("[CHECK] : title_word_list shape : ", np.shape(pad_title_embedding))
        print("[CHECK] : final_word_embedding shape : ", np.shape(pad_word_embedding))

        feature = []
        instance_num = np.shape(title_word_list)[0]
        for i in range(instance_num):
            concat_vec = np.concatenate((pad_title_embedding[i], pad_word_embedding[i]))
            feature.append(concat_vec)

        label = one_hot_vector(components)
        label = np.delete(label, word_noise, axis=0)

        print("[INFO] feature shape : {0}".format(np.shape(feature)))
        print("[INFO] label shape : {0}".format(np.shape(label)))

        print()
        print("-- Word Embedding End --")
        print()

        return feature, label




