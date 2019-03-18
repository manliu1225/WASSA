from collections import defaultdict
import numpy as np
import logging
import re
import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.linear_model import LogisticRegression
from keras.preprocessing.sequence import pad_sequences
from . import BaseSentimentClassifier, PureTransformer, ListCountVectorizer, BasicTokenizer, NGramTransformer, CharNGramTransformer, CMUArkTweetPOSTagger, CMUArkTweetBrownClusters, LowercaseTransformer, Negater
# from .lexicons import NRCEmotionLexicon, NRCHashtagEmotionLexicon, MaxDiffTwitterLexicon, NRCHashtagSentimentWithContextUnigrams, NRCHashtagSentimentWithContextBigrams, NRCHashtagSentimentLexiconUnigrams, NRCHashtagSentimentLexiconBigrams, Sentiment140WithContextUnigrams, Sentiment140WithContextBigrams, Sentiment140LexiconUnigrams, Sentiment140LexiconBigrams, YelpReviewsLexiconUnigrams, YelpReviewsLexiconBigrams, AmazonLaptopsReviewsLexiconUnigrams, AmazonLaptopsReviewsLexiconBigrams, MPQAEffectLexicon, MPQASubjectivityLexicon, HarvardInquirerLexicon, BingLiuLexicon, AFINN111Lexicon, SentiWordNetLexicon, LoughranMcDonaldLexicon
from sklearn.base import BaseEstimator, TransformerMixin
from .bilstm import BiLSTMClassifier
from .conNet import ConNetlassifier
logger = logging.getLogger(__name__)
W2V_FILE = os.path.join(os.getcwd(), "word-embedding/glove-tweet/output/text_tweets.m20.vocab.w2.a0.75.v100.txt")
'''liumandeMacBook-Air:twitter_emoij_prediction liuman$ python test.py --X_train /Users/liuman/Documents/semeval2018/twitter_emoij_prediction/data/data-split-20171027/1/t_train_1.text --y_train /Users/liuman/Documents/semeval2018/twitter_emoij_prediction/data/data-split-20171027/1/t_train_1.labels --X_test /Users/liuman/Documents/semeval2018/twitter_emoij_prediction/data/data-split-20171027/1/t_test_1.text --y_test /Users/liuman/Documents/semeval2018/twitter_emoij_prediction/data/data-split-20171027/1/t_test_1.labels '''
class Print(BaseEstimator, TransformerMixin):
  def fit(self, x, y=None):
    print x.shape
    return self
  def transformer(X):
    return X

class BiLSTMSentimentClassifier(BaseSentimentClassifier):

  def __init__(self, **kwargs):
    super(BiLSTMSentimentClassifier, self).__init__(classifier=BiLSTMClassifier(), **kwargs)

  def _make_preprocessor(self):
    with open(W2V_FILE, "rb") as lines:
      w2v = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}
    preprocessor = make_pipeline(
      BasicTokenizer(), 
      MeanEmbeddingVectorizer(w2v),
      )

    return preprocessor

class ConNetSentimentClassifier(BaseSentimentClassifier):

  def __init__(self, **kwargs):
    super(ConNetSentimentClassifier, self).__init__(classifier=ConNetlassifier(), **kwargs)

  def _make_preprocessor(self):
    with open(W2V_FILE, "rb") as lines:
      w2v = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}
    preprocessor = make_pipeline(
      BasicTokenizer(), 
      MeanEmbeddingVectorizer(w2v),
      )

    return preprocessor

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

class TfidfEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])
        
class W2Vembedding(MeanEmbeddingVectorizer):
  def __init__(self, **kwargs):
    # W2V_FILE = os.path.join(os.getcwd(), "tweets/resources/tweet_glove/glove.twitter.27B.50d.txt")
    with open(W2V_FILE, "rb") as lines:
      w2v = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}
    super(W2Vembedding, self).__init__(w2v)

  def get_params(self, deep=True): return dict()
