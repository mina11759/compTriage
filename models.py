import keras
import tensorflow
import tensorflow as tf
from keras.models import Model
from metrics import MetricsAtTopK

def get_layer_weight(model: Model):
    layer_name = 'conv_1'
    for idx, layer in enumerate(model.layers):
        if layer_name in layer.name:
            return idx, layer


def top_model(train_x, train_y, valid_x, valid_y, test_x, test_y, token_num, vec_dim):
    with tf.device('/device:GPU:0'):
        top_CNN_model = keras.models.Sequential([
                    keras.layers.Conv2D(filters=12, kernel_size=(5, 5), activation='relu',
                                        input_shape=(token_num, vec_dim, 1), name='conv_1'),
                    keras.layers.MaxPool2D(pool_size=(1, 2)),
                    keras.layers.Conv2D(filters=24, kernel_size=(5, 5), activation='relu', name='conv_2'),
                    keras.layers.MaxPool2D(pool_size=(1, 2)),
                    keras.layers.Flatten(),
                    keras.layers.Dense(50, activation='softmax'),
                    keras.layers.Dropout(0.5),
                    keras.layers.Dense(2, activation=None)])

        # Convolutional Neural Networks(CNN) model for top-component
        # top_CNN_model = CNN()
        top_CNN_model.summary()
        # opt = RMSprop(lr=0.01)
        top_CNN_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        top_CNN_model.fit(train_x, train_y, epochs=50, validation_data=(valid_x, valid_y), shuffle=True)

    test_loss, accuracy = top_CNN_model.evaluate(train_x, train_y, verbose=2)
    print('\nTest loss : ', test_loss)
    print('Test accuracy :', accuracy)

    pred = top_CNN_model.predict(test_x)

    print('Prediction : ', pred.shape)
    print('Test labels : ', test_y.shape)

    # -- 다른 데이터, 다른 임베딩 기법을 쓸 때마다 이름 바꿔주는게 좋음
    top_CNN_model.save('result/top_model_qpid_bert.h5', save_format='h5')
    top_CNN_model.save_weights('result/top_model_weight_qpid_bert.h5')


def bottom_model(model, train_x, train_y, valid_x, valid_y, test_x, test_y, token_num, vec_dim, output_dim):
    target_idx, target_layer = get_layer_weight(model)
    K = 5
    metrics = MetricsAtTopK(k=K)

    with tf.device('/device:GPU:0'):
        bottom_CNN_model = keras.models.Sequential([
            keras.layers.Conv2D(filters=12, kernel_size=(5, 5), activation='relu',
                                input_shape=(token_num, vec_dim, 1), name='conv_1'),
            keras.layers.Activation('relu'),
            keras.layers.Conv2D(filters=24, kernel_size=(5, 5), activation='relu', name='conv_2'),
            keras.layers.MaxPool2D(pool_size=(1, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(50, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),

            keras.layers.Dense(output_dim, activation='softmax')])

        for layer in bottom_CNN_model.layers:
            if layer.name == 'conv_1':
                layer.set_weights(model.layers[target_idx].get_weights())

        bottom_CNN_model.summary()
        bottom_CNN_model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['acc', metrics.recall_at_k]
        )

        bottom_CNN_model.fit(train_x, train_y, epochs=50, validation_data=(valid_x, valid_y), shuffle=True)

    score = bottom_CNN_model.evaluate(test_x, test_y, verbose=2)
    """
    verbose = 학습 중 출력되는 문구를 설정합니다.
    0 : X, 1 : progress bar, 2 : 미니 배치당 손실 정보
    """

    # print('Prediction : ', pred.shape)
    # print('Test labels : ', test_y.shape)

    bottom_CNN_model.save('bottom_output/bottom_2_qpid_model_bert_{0}.h5'.format(K), save_format='h5')
    bottom_CNN_model.save_weights("bottom_output/bottom_2_qpid_weight_bert_{0}.h5".format(K), save_format='h5')

    print('\nTest loss : {0}'.format(score[0]))
    print('\nTest Accuracy : {0}'.format(score[1]))
    print('Test recall@{0} : {1}'.format(K, score[2]))