from preprocessing_bert import bottom_prepare_dataset
import tensorflow

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import *

import tempfile
import os
from tensorflow import keras
from tensorflow.keras.utils import Sequence


# split datasets to train, test, valid sets
def split_dataset(feature, label):
    # ndarray to list
    # label = label.tolist()

    # train, valid, test data: ratio(6:2:2)
    train_x, test_x, train_y, test_y = train_test_split(feature, label.tolist(), train_size=0.6)
    valid_x, test_x, valid_y, test_y = train_test_split(test_x, test_y, train_size=0.5)

    return train_x, train_y, test_x, test_y, valid_x, valid_y


if __name__ == '__main__':
    # prepare_dataset('FCREPO_Sort_top.csv')
    feature, sublabel = bottom_prepare_dataset('Qpid_Integration_bottom_2.csv')

    """
    Data :
        - Qpid_Integration_bottom_1.csv -> acc: 0.786, re@k: 0.781
        - Qpid_Integration_bottom_2.csv -> acc: 0...., re@k : 1..........
        - FCREPO_Sort_bottom_1.csv -> acc: 0.125, re@k: 0.792
        - FCREPO_Sort_bottom_2.csv -> acc: 0.422, re@k: 1..
        

    Embedding : 
        - using bert : prepare_dataset('dataset_name.csv')
        - using Word2Vec : word_prepare_dataset('dataset_name.csv')

    """
    # feature, label = word_prepare_dataset('test.csv')

    # vec_dim = 300
    vec_dim = 768 # option
    feature_num = np.shape(feature)[0]
    token_num = np.shape(feature)[1]

    # split datasets
    train_x, train_y, test_x, test_y, valid_x, valid_y = split_dataset(feature, sublabel)

    print("[CHECK] np.shape(train_x) : ", np.shape(train_x))
    train_x = np.array(tf.expand_dims(train_x, axis=-1))
    valid_x = np.array(tf.expand_dims(valid_x, axis=-1))
    test_x = np.array(tf.expand_dims(test_x, axis=-1))

    train_y = np.array(train_y)
    valid_y = np.array(valid_y)
    test_y = np.array(test_y)

    output_dim = np.shape(test_y)[1]

    """
        top_model_name :
            - top_model_fcrepo_word.h5
            - top_model_fcrepo_bert.h5
            - top_model_qpid_word.h5
            - top_model_qpid_bert.h5
    """

    top_model_name = 'result/top_model_qpid_bert.h5'
    top_model = tensorflow.keras.models.load_model(top_model_name)

    bottom_model(top_model, train_x, train_y, valid_x, valid_y, test_x, test_y, token_num, vec_dim, output_dim)
