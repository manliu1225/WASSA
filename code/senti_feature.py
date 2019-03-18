from . import load_instances, BasicTokenizer
from argparse import ArgumentParser
from collections import defaultdict
import numpy as np
import logging
import re
import os
# from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline, make_union
from keras.preprocessing.sequence import pad_sequences
# from . import BaseSentimentClassifier, PureTransformer, ListCountVectorizer, BasicTokenizer, NGramTransformer, CharNGramTransformer, CMUArkTweetPOSTagger, CMUArkTweetBrownClusters, LowercaseTransformer, Negater
# from .lexicons import NRCEmotionLexicon, NRCHashtagEmotionLexicon, MaxDiffTwitterLexicon, NRCHashtagSentimentWithContextUnigrams, NRCHashtagSentimentWithContextBigrams, NRCHashtagSentimentLexiconUnigrams, NRCHashtagSentimentLexiconBigrams, Sentiment140WithContextUnigrams, Sentiment140WithContextBigrams, Sentiment140LexiconUnigrams, Sentiment140LexiconBigrams, YelpReviewsLexiconUnigrams, YelpReviewsLexiconBigrams, AmazonLaptopsReviewsLexiconUnigrams, AmazonLaptopsReviewsLexiconBigrams, MPQAEffectLexicon, MPQASubjectivityLexicon, HarvardInquirerLexicon, BingLiuLexicon, AFINN111Lexicon, SentiWordNetLexicon, LoughranMcDonaldLexicon
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib

# from .bilstm import BiLSTMClassifier
# from .conNet import ConNetlassifier
logger = logging.getLogger(__name__)
W2V_FILE = os.path.join(os.getcwd(), "word-embedding/glove-tweet/output/text_tweets.m20.vocab.w2.a0.75.v100.txt")
'''liumandeMacBook-Air:twitter_emoij_prediction liuman$ python test.py --X_train /Users/liuman/Documents/semeval2018/twitter_emoij_prediction/data/data-split-20171027/1/t_train_1.text --y_train /Users/liuman/Documents/semeval2018/twitter_emoij_prediction/data/data-split-20171027/1/t_train_1.labels --X_test /Users/liuman/Documents/semeval2018/twitter_emoij_prediction/data/data-split-20171027/1/t_test_1.text --y_test /Users/liuman/Documents/semeval2018/twitter_emoij_prediction/data/data-split-20171027/1/t_test_1.labels '''


parser = ArgumentParser(description='test')
parser.add_argument('--X_train', type=open,  metavar='file', default=None, required=True, help='List of files to use as training data.')
parser.add_argument('--y_train', type=open, metavar='file', default=None, help='List of files to use as test data. Can be empty if only cross validation experiments are desired.')
parser.add_argument('--X_test', type=open,  metavar='file', default=None, required=True, help='List of files to use as training data.')
parser.add_argument('--y_test', type=open, metavar='file', default=None, help='List of files to use as test data. Can be empty if only cross validation experiments are desired.')
parser.add_argument('--pred_f', help='predicted y.')
parser.add_argument('--save_model', type=str, help='Save model')
parser.add_argument('--load_model', type=str, help='Load model')
parser.add_argument('--X_train_f',  required=True, help='List of files to use as training data.')
parser.add_argument('--X_test_f', required=True, help='List of files to use as training data.')

# parser.add_argument('--num_leaves', default=64, type=int, help='parameters for classifier')
# parser.add_argument('--n_estimators', default=300, help='parameters for classifier')

args = parser.parse_args()

X_train, y_train = load_instances(args.X_train, args.y_train)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test, y_test = load_instances(args.X_test, args.y_test)
X_test = np.array(X_test)
y_test = np.array(y_test)

class MeanEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
      self.MAX_LENGTH = 0.0
      total_len = 0.0
      for e in X: 
        total_len += len(e)
        self.MAX_LENGTH = max(self.MAX_LENGTH, len(e))
        print('max lenght is {}'.format(self.MAX_LENGTH))
      return self

    def transform(self, X):
      X_new = []
      for line in X:
        X_new.append([self.word2vec[word] if word in self.word2vec else np.zeros(self.dim) for word in line])
      X_new = np.array(X_new)
      X = pad_sequences(X_new, self.MAX_LENGTH)
      # return X.reshape(X.shape[0],X.shape[1]*X.shape[2])
      print(X.shape)
      return X

    def get_params(self, deep=True): return dict()

class W2Vembedding(MeanEmbeddingVectorizer):
  def __init__(self, **kwargs):
    # W2V_FILE = os.path.join(os.getcwd(), "tweets/resources/tweet_glove/glove.twitter.27B.50d.txt")
    with open(W2V_FILE, "rb") as lines:
      w2v = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}
    super(W2Vembedding, self).__init__(w2v)

  def get_params(self, deep=True): return dict()


with open(W2V_FILE, "rb") as lines: w2v = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}
preprocessor = make_pipeline(
  BasicTokenizer(), 
  MeanEmbeddingVectorizer(w2v),
  )
senti_X_train = preprocessor.fit_transform(X_train, y_train)
senti_X_test = preprocessor.fit_transform(X_test, y_test)

# print(X_train[:len(X_train)/2].shape)
# print(X_train[len(X_train)/2:].shape)

# senti_X_train_firsthalf = preprocessor.fit_transform(X_train[:len(X_train)/2], y_train[:len(X_train)/2])
# senti_X_test_firsthalf = preprocessor.fit_transform(X_test[:len(X_test)/2], y_test[:len(X_test)/2])

# senti_X_train_lasthalf = preprocessor.fit_transform(X_train[len(X_train)/2:], y_train[len(X_train)/2:])
# senti_X_test_lasthalf = preprocessor.fit_transform(X_test[len(X_test)/2:], y_test[len(X_test)/2:])

# senti_X_train = np.vstack((senti_X_train_firsthalf , senti_X_train_lasthalf))
# senti_X_train = np.vstack((senti_X_test_firsthalf , senti_X_test_lasthalf))

joblib.dump(senti_X_train, args.X_train_f, compress=True)
joblib.dump(senti_X_test, args.X_test_f, compress=True)
print('save successfully! good luck manman')

