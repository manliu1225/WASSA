import numpy as np
import collections
import nltk
from sklearn.cross_validation import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.engine import Input
from keras import optimizers
import keras.backend as keras_backend
from keras.models import Sequential
# from attention_lstm import  Attention
from keras.layers import  Activation, Convolution1D, Dense, Dropout, Lambda, SpatialDropout1D, Dense, Bidirectional
from keras.layers.merge import Concatenate, Average, Maximum
from keras.layers.pooling import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Model, load_model
# from ..tweetokenize import Tokenizer
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
import argparse
import codecs
import sklearn.utils.class_weight
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import ModelCheckpoint
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':1})))
np.random.seed(42)

class BiLSTMClassifier(BaseEstimator, TransformerMixin):
    ''' test classification using lstm'''
    def __init__(self, MAX_VOCAB_SIZE = 2000, EPOCHS = 100, MAX_LENGTH = 20, BATCH_SIZE = 128, EMBED_SIZE = 4000, nb_filter=512):
        ### filter number 500 is ok
        self.EPOCHS = EPOCHS
        self.MAX_LENGTH = MAX_LENGTH
        self.BATCH_SIZE = BATCH_SIZE
        self.nb_filter = nb_filter

    def set_params(self, **params):
        for key, value in params.items():
            if key.startswith('dropout1'): self.dropout1 = value
            if key.startswith('dropout2'): self.dropout2 = value
        return self


    def build_lstm(self, output_dim, X):

        def max_1d(X):
            return keras_backend.max(X,axis=1)

        nb_filter = self.nb_filter     
        loss_function = "categorical_crossentropy"
        maxpool = Lambda(max_1d, output_shape=(nb_filter,))
        # embedded = Embedding(X.shape[0], X.shape[1], input_length=self.MAX_LENGTH)(sequence)
        # 4 convolution layers (each 1000 filters)
        conv_filters = []
        S = Input(shape=[X.shape[1], X.shape[2]])
        for filters in [3, 5,7]:
            filtermodel = Convolution1D(nb_filter=nb_filter,
                                         filter_length=filters,
                                         border_mode="same",
                                         activation="relu",
                                         subsample_length=1)(S)
            # poollayer = MaxPooling1D(pool_size=2, strides=None, padding='valid')(filtermodel)
            # poollayer = maxpool(filtermodel)
            # print filtermodel.output_shape
            # keras.layers.pooling.MaxPooling1D(pool_size=2, strides=None, padding='valid')
            # maxpoollayer = maxpool(filtermodel)
            # print filtermodel.output_shape
            conv_filters.append(filtermodel)
        # cnn = [Convolution1D(filter_length=filters, nb_filter=10, border_mode="same") for filters in [2, 3, 5, 7]]
        # concatenate
        # merged_cnn = merge([cnn(embedded) for cnn in cnn], mode="concat")
        print type(conv_filters)
        print isinstance(conv_filters, list)
        print len(conv_filters)
        # self.model = Concatenate(conv_filters)
        # self.model.add(Concatenate()(conv_filters))
        # # self.model.add(Reshape((1, self.model.output_shape[1])))
        # # print self.model.output_shape
        # self.model.add(Attention(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2)))
        # # print self.model.output_shape
        # # self.model.add(Reshape((1, self.model.output_shape[1])))
        # # print self.model.output_shape
        # self.model.add(Bidirectional(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2)))

        # # softmax output layer
        # self.model.add(Dense(output_dim=output_dim, activation="softmax"))

        ####

        # the complete omdel
        merged_cnn = Concatenate()(conv_filters)
        # self.model.add(Reshape((1, self.model.output_shape[1])))
        # print self.model.output_shape
        # attentlstm = Attention(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2))(conv_filters)
        # print self.model.output_shape
        # self.model.add(Reshape((1, self.model.output_shape[1])))
        # print self.model.output_shape
        blstm = Bidirectional(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))(merged_cnn)

        # softmax output layer
        dense = Dense(output_dim=output_dim, activation="softmax")(blstm)

        # the complete omdel

        # try using different optimizers and different optimizer configs
        self.model = Model(input = S, output=dense)
        # load_weights from the checkpoint model
        # self.model.load_weights('./checkpoitn_models/lstm/weights.01.hdf5')
        rmsprop  = optimizers.RMSprop(lr=0.01, decay=0.05)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        return self.model
    def fit(self, X, y):
        # class_weight = sklearn.utils.class_weight.compute_class_weight('balanced',np.unique(y), y)
        # self.class_weight_dict = dict(enumerate(class_weight))
        # y = np_utils.to_categorical(y, 5)
        # self.model = Sequential()
        # print X.shape[1]
        # print X.shape[2]
        # self.model.add(Bidirectional(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2), input_shape=(X.shape[1], X.shape[2])))
        # print self.model.output_shape
        # # self.model.add(Bidirectional(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2), input_shape=(1, X.shape[1])))
        # self.model.add(Dense(20, activation='softmax'))
        # print self.model.output_shape
        # self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        # self.model.fit(X, y, class_weight=self.class_weight_dict, epochs=self.EPOCHS, verbose=1)

        # class_weight = sklearn.utils.class_weight.compute_class_weight('balanced',np.unique(y), y)
        # self.model = self.build_lstm(6, X)
        # y = np_utils.to_categorical(y, 6)
        # filepath="./checkpoitn_models/lstm/weights.{epoch:02d}.hdf5"
        # checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        # callbacks_list = [checkpoint]

        # self.model.fit(X, y, batch_size=self.BATCH_SIZE, class_weight=class_weight, verbose=1, epochs=self.EPOCHS, callbacks=callbacks_list)
        # self.model.save('./models/{}_{}_model.h5'.format(datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), self.__class__.__name__))

        self.model = load_model('./models/2018-07-11_13_48_06_BiLSTMClassifier_model.h5')

        return self

    def predict_proba(self, X):
        print 'predict ...'
        return self.model.predict(X)

    def predict(self, X):
        y_pred = []
        y_pred_li = self.predict_proba(X)
        for e in y_pred_li:
            y_pred.append(list(e).index(max(list(e))))
        return np.array(y_pred)

