import numpy as np
import collections
import nltk
from sklearn.cross_validation import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.engine import Input
import keras.backend as keras_backend
from keras.models import Sequential
from attention_lstm import  Attention
from keras.layers import  Activation, Convolution1D, Dense, Dropout, Lambda, SpatialDropout1D, Dense, Bidirectional
from keras.layers.merge import Concatenate, Average, Maximum
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Model
# from ..tweetokenize import Tokenizer
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
import argparse
import codecs
import sklearn.utils.class_weight
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from datetime import datetime
from keras.callbacks import ModelCheckpoint
from keras import optimizers

# KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':2})))
np.random.seed(42)


class BiLSTMSentiClassifier(BaseEstimator, TransformerMixin):
    ''' test classification using lstm'''
    def __init__(self, MAX_VOCAB_SIZE = 2000, EPOCHS = 40, MAX_LENGTH = 50, BATCH_SIZE = 128, EMBED_SIZE = 4000):
        self.EPOCHS = EPOCHS
        self.MAX_LENGTH = MAX_LENGTH
        self.BATCH_SIZE = BATCH_SIZE

    def set_params(self, **params):
        for key, value in params.items():
            if key.startswith('dropout1'): self.dropout1 = value
            if key.startswith('dropout2'): self.dropout2 = value
        return self


    def build_lstm(self, output_dim, X):
        self.model = Sequential()
        self.model.add(Dense(512, activation="relu", kernel_initializer="uniform"))
        self.model.add(Bidirectional(LSTM(units=128, dropout=0.5, recurrent_dropout=0.2), input_shape=(X.shape[1], X.shape[2])))
        # self.model.add(Bidirectional(LSTM(units=64, dropout=0.5, recurrent_dropout=0.2)), input_shape=(1, X.shape[1]))
        self.model.add(Dense(20, activation='softmax'))
        # softmax output layer
        self.model.add(Dense(output_dim, activation='softmax'))
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
        class_weight = sklearn.utils.class_weight.compute_class_weight('balanced',np.unique(y), y)
        self.model = self.build_lstm(6, X)
        y = np_utils.to_categorical(y, 6)
        filepath="./checkpoitn_models/blstm_senti/weights.{epoch:02d}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        self.model.fit(X, y, batch_size=self.BATCH_SIZE, class_weight=class_weight, verbose=1, epochs=self.EPOCHS, callbacks=callbacks_list)
        self.model.save('./models/blstm_senti/{}_{}_model.h5'.format(datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), self.__class__.__name__))

        # self.model = load_model('./models/2018-07-08_10_47_36_ConNetlassifier_model.h5')

        # self.model.load_weights('./checkpoitn_models/conNet/weights.03.hdf5')


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

