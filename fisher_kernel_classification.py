import os
import time
import random
import csv

import numpy as np
import scipy.sparse as spa
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

import utils
from gensim_lda import gensim_lda
from fisher import fisher_score, estimate_information

logger = utils.get_logger()
root = "./datasets/2classes/"

"""
	corpus' documents
"""
logger.info('traing data and test data creation')
files = {path:[os.path.join(path, name) for name in files] for path, subdirs, files in os.walk(root)}

train_data = list()
test_data = list()
train_classes = list()
test_classes = list()
train_dict = dict()

data = list()

for path,docs in files.iteritems():

	num_tests = len(docs) / 10
	num_trains = len(docs) - num_tests
	# num_trains = num_tests
	random.shuffle(docs)

	train_data += docs[:num_trains]
	test_data += docs[num_trains:num_trains+num_tests]
	data += train_data + test_data

	train_classes += [path] * num_trains
	test_classes += [path] * num_tests

D_train = len(train_data)
print D_train
D_test = len(test_data)
print D_test

shuffled_train_data = random.sample(train_data,D_train)
shuffled_train_classes = [train_classes[train_data.index(d)] for d in shuffled_train_data]

train_classes_array = np.array([[c] for c in shuffled_train_classes])
# test_classes_array = np.array([[c] for c in test_classes])
unique_classes = np.unique(train_classes_array)

"""
	vocabulary
"""
logger.info('vocabulary extraction')
vectorizer = CountVectorizer(input='filename',decode_error='replace',stop_words='english',min_df=0.01,max_df=1.)
vectorizer.fit(data)
V = len(vectorizer.vocabulary_)
print V

"""
	BoWs
"""
logger.info('bow arrays creation')
train_array = vectorizer.transform(shuffled_train_data)
test_array = vectorizer.transform(test_data)

with open('./results/fisher_kernel_2classes.csv', 'wb') as csvfile:
	spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
	for num_topics in range(2,10):

		logger.info("classification with {} topics".format(num_topics))

		logger.info("lda topic extraction")
		"""
			lda : training model
		"""
		time0 = time.time()
		lda,train_bow = gensim_lda(train_array,num_topics,6,40)
		time2 = time.time()
		logger.info('lda time {}s'.format(time2 - time0))

		"""
			SGD with Fisher kernel 
		"""

		clf = SGDClassifier(shuffle=True)

		logger.info('information matrix estimation')
		time1 = time.time()

		info = estimate_information(train_array,lda,num_topics,V,D_train)
		info = np.linalg.inv(info)
		info = spa.linalg.splu(info)
		info = info.solve(np.eye(info.shape[0]))

		time2 = time.time()
		logger.info('information estimation time {}s'.format(time2 - time1))

		logger.info('SGD classifier')
		time1 = time.time()

		for i,score in enumerate(fisher_score(train_array,lda,num_topics,V,D_train)):
			kernel = np.dot(score,info)
			kernel = np.dot(kernel,score.T)
			clf.partial_fit(kernel, train_classes_array[i], unique_classes) 

		time2 = time.time()
		logger.info('sgd time {}s'.format(time2 - time1))

		logger.info('test data classification')
		time1 = time.time()

		corrects = 0

		for i,score in enumerate(fisher_score(test_array,lda,num_topics,V,D_test)):
			kernel = np.dot(score,info)
			kernel = np.dot(kernel,score.T)
			c = clf.predict(kernel) 
			# print 'class predicted {}'.format(c[0])
			# print 'real class {}'.format(test_classes[i])
			if c[0] == test_classes[i]:
				corrects += 1

		time2 = time.time()
		logger.info('classification time {}s'.format(time2 - time1))
		precision = corrects * 100. / D_test
		logger.info('presicion {}'.format(precision))
		logger.info('total time {}s'.format(time2 - time0))
		spamwriter.writerow([num_topics, precision, time2 - time0])