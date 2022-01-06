from data.dataManagerImpl import DataManagerImpl
from embed.w2vEmbedManager import W2VEmbedManager
from embed.bertEmbedManager import BertEmbedManager
from model.modelManagerImpl import ModelManagerImpl
from modelMapper import ModelMapper
from model.interface.modelCollection import ModelCollection
# top_model -> bottom_model
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix



"""
    - 1, test data load
    - 2, modeling and model_load
    - 3, evaluation
    
    main for Parent Model
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Component Prediction with Hierarchical Structure")
    parser.add_argument('--data', dest='data_name', default='', help='')
    parser.add_argument('--ctgr', dest='category_number', default='0', help='data category number (0, 1, 2 ..)')
    parser.add_argument('--h_level', dest='h_level', default='', help='hierarchy level (parent, child)')
    parser.add_argument('--t_level', dest='t_level', default='', help='token level (word, sentence)')
    parser.add_argument('--embed', dest='embed', default='', help='embedding method (bert, word2vec ..)')
    parser.add_argument('--model', dest='model_name', default='', help='ML Algorithm (cnn, lstm etc..)')

    args = parser.parse_args()

    data_name = str(args.data_name)
    ctgr = str(args.category_number)
    h_level = str(args.h_level)
    t_level = str(args.t_level)
    embed = str(args.embed)
    model_name = str(args.model_name)

    # --------------------------------------------- #
    if embed == 'bert':
        embed_manager = BertEmbedManager(h_level, embed)
        data_manager = DataManagerImpl(data_name, t_level, ctgr, embed_manager)
        feature, label = data_manager.load_data()

        print("[CHECK] embed_dim : ", embed_manager.embed_dim)
        print("[CHECK] output_dim : ", embed_manager.output_dim)

        model = ModelMapper(embed_manager).get_h_model().mapping(model_name)
        # print("[CHECK] model type : ", type(model)) # --> Sequential Type
        model_manager = ModelManagerImpl(feature, label, embed_manager, data_manager, model, model_name, ctgr)
        model_manager.train()
        model_manager.save_model()

    elif embed == 'w2v': # Not yet implement
        embed_manager = W2VEmbedManager(h_level, embed)
        data_manager = DataManagerImpl(data_name, t_level, ctgr, embed_manager)
        feature, label = data_manager.load_data()

        print("[CHECK] embed_dim : ", embed_manager.embed_dim)
        print("[CHECK] output_dim : ", embed_manager.output_dim)

        model = ModelMapper(embed_manager).get_h_model().mapping(model_name)
        # print("[CHECK] model type : ", type(model)) # --> Sequential Type
        model_manager = ModelManagerImpl(feature, label, embed_manager, data_manager, model, model_name)
        model_manager.train()
        model_manager.save_model()

    elif embed == 'bert_classic':
        embed_manager = BertEmbedManager(h_level, embed)
        data_manager = DataManagerImpl(data_name, t_level, ctgr, embed_manager)
        feature, label = data_manager.load_data()

        print("[CHECK] embed_dim : ", embed_manager.embed_dim)
        print("[CHECK] output_dim : ", embed_manager.output_dim)

        model = ModelMapper(embed_manager).get_h_model().mapping(model_name)
        model_manager = ModelManagerImpl(feature, label, embed_manager, data_manager, model, model_name, ctgr)
        model_manager.split()

        train_x = np.squeeze(model_manager.train_x, axis=3)
        test_x = np.squeeze(model_manager.test_x, axis=3)

        train_x = np.reshape(train_x, (np.shape(train_x)[0], -1))
        test_x = np.reshape(test_x, (np.shape(test_x)[0], -1))

        model.fit(train_x, model_manager.train_y)
        y_pred = model.predict(test_x)
        print(multilabel_confusion_matrix(test_x, y_pred))

    # GPT, ELMo / RF, SVM etc 추가 예정

    else:
        print("[ERROR] Rewrite embed type in terminal.")
        exit()

