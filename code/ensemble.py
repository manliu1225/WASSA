from . import load_instances, BaseSentimentClassifier, PureTransformer, ListCountVectorizer, BasicTokenizer, NGramTransformer, CharNGramTransformer, CMUArkTweetPOSTagger, CMUArkTweetBrownClusters, LowercaseTransformer, Negater
from argparse import ArgumentParser
import codecs
import pickle
from sklearn.metrics import classification_report, f1_score
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer, f1_score
from sklearn.ensemble import VotingClassifier
from sklearn.externals import joblib
from .lstm2 import BiLSTMSentimentClassifier, ConNetSentimentClassifier
# from .lstm2 import ConNetSentimentClassifier
from .ffnn_senti import ffNNSentimentclassifier
from .lightgbm_allfeatures import GBDTSentimentClassifier

# import os

parser = ArgumentParser(description='test')
parser.add_argument('--X_train', type=open,  metavar='file', default=None, required=True, help='List of files to use as training data.')
parser.add_argument('--y_train', type=open, metavar='file', default=None, help='List of files to use as test data. Can be empty if only cross validation experiments are desired.')
parser.add_argument('--X_test', type=open,  metavar='file', default=None, required=True, help='List of files to use as training data.')
parser.add_argument('--y_test', type=open, metavar='file', default=None, help='List of files to use as test data. Can be empty if only cross validation experiments are desired.')
parser.add_argument('--pred_f', help='predicted y.')
parser.add_argument('--estimator', required=True, choices=['gbdt', 'lstmclf', 'conclf', 'ffnnclf', 'lstmsenticlf', 'enclf'], help='give the estimator you want to use.')
# parser.add_argument('--estimator', required=True, choices=['gbdt', 'lstmclf', 'ffnnclf', 'lstmsenticlf', 'enclf'], help='give the estimator you want to use.')
parser.add_argument('--save_model', type=str, help='Save model')
parser.add_argument('--load_model', type=str, help='Load model')
# parser.add_argument('--num_leaves', default=64, type=int, help='parameters for classifier')
# parser.add_argument('--n_estimators', default=300, help='parameters for classifier')

args = parser.parse_args()

X_train, y_train = load_instances(args.X_train, args.y_train)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test, y_test = load_instances(args.X_test, args.y_test)
X_test = np.array(X_test)
y_test = np.array(y_test)

global LABEL
LABEL = {
0:'anger',
1:'disgust',
2:'fear',
3:'joy',
4:'sad',
5:'surprise'}


gbdt = GBDTSentimentClassifier(classifier__num_leaves=32, classifier__n_estimators=600)
lstmclf = BiLSTMSentimentClassifier()
# conclf = ConNetSentimentClassifier()
ffnnclf = ffNNSentimentclassifier()
# # lstmsenticlf = BiLSTMSentiFeatureClassifier()
# # enclf = VotingClassifier(estimators = [('gbdt', gbdt), ('ffnn', ffnnclf), ('con', conclf), ('lstm', lstmclf)], voting='soft', weights=[2, 2, 2, 1], n_jobs = 1)
enclf = VotingClassifier(estimators = [('ffnn', ffnnclf), ('lstm', lstmclf)], voting='soft', weights=[2, 1], n_jobs = 1)
# enclf = VotingClassifier(estimators = [('gbdt', gbdt), ('ffnn', ffnnclf), ('lstm', lstmclf)], voting='soft', weights=[2, 2, 1], n_jobs = 1)
# print 'estimator is {}'.format(args.estimator)



if args.load_model: estimator = joblib.load(args.load_model) 
else: 
	if args.estimator == 'lstmclf': estimator = lstmclf
	# elif args.estimator == 'conclf': estimator = conclf
	elif args.estimator == 'enclf': estimator = enclf
	elif args.estimator == 'ffnnclf': estimator = ffnnclf
	# elif args.estimator == 'lstmsenticlf': estimator = lstmsenticlf
	else: estimator = gbdt
	# print(estimator.get_params().keys())
	# estimator = eval(args.estimator)
	estimator.fit(X_train, y_train)

# estimator = conclf
# estimator.fit(X_train, y_train)

# parameters = {'classifier__num_leaves':[16,32,64, 128], 'classifier__n_estimators':[100, 300, 700, 1000], 'classifier__min_data':[5]}
# macrof1_scorer = make_scorer(f1_score, average='macro')
# estimator = GridSearchCV(gbdt, parameters, scoring=macrof1_scorer,n_jobs=16, verbose=3, cv=3)
# estimator.fit(X_train, y_train)
# bp = dictestimator.best_params_
# print('best model is {}'.format(estimator.best_params_))


	# joblib.dump(estimator, 'gbdt_leaves{}_nestimators{}.model'.format(args.num_leaves, args.n_estimators))
y_pred = estimator.predict(X_test)
# with open('data/{}/{}_pred_labels.txt'.format(str(estimator).strip("( )"),str(args.X_test).split('.')[0].split('/')[-1]), 'w') as f:
if args.pred_f:
	with open(args.pred_f, 'w') as f:
	    for e in y_pred:
	        # print e
	        f.write('{}'.format(LABEL[e]))
	        f.write('\n')

print(classification_report(y_test, y_pred))
print(f1_score(y_test, y_pred, average = 'macro'))
if args.save_model:
	joblib.dump(estimator, args.save_model)
# dict_para = estimator.named_steps['classifier'].get_params()
# print('parameters are {}'.format(', '.join(['{}:{}'.format(k, v) for k,v in dict_para.items()])))
