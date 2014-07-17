import os
import time
import random
import csv

import numpy as np
import scipy.sparse as spa
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

from gensim_lda import gensim_lda
from fisher import fisher_score

root = "./datasets/2classes/"

"""
	corpus' documents
"""
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

"""
	vocabulary
"""
vectorizer = CountVectorizer(input='filename',decode_error='replace',stop_words='english',min_df=0.001,max_df=1.)
vectorizer.fit(data)
V = len(vectorizer.vocabulary_)
print V

"""
	BoWs
"""
train_array = vectorizer.transform(shuffled_train_data)
test_array = vectorizer.transform(test_data)

"""
	classification
"""

with open('./results/fisher_kernel_2classes.csv', 'wb') as csvfile:
	spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
	for num_topics in [1, 10, 100, 200, 300, 400, 500, 600, 700]:
		"""
			lda : training model
		"""
		time1 = time.time()
		lda,train_bow = gensim_lda(train_array,num_topics,6,40)
		time2 = time.time()
		print 'lda time {}'.format(time2 - time1)

		"""
			SGD with Fisher score 
		"""
		classes = np.array([[c] for c in shuffled_train_classes])
		clf = SGDClassifier(shuffle=True)

		unique_classes = np.unique(classes)

		time1 = time.time()
		for i,score in enumerate(fisher_score(train_array,lda,num_topics,V,D_train)):
			clf.partial_fit(score, classes[i], unique_classes) 

		time2 = time.time()
		print 'sgd time {}'.format(time2 - time1)

		"""
			SGD with Fisher score 
		"""
		classes = np.array([[c] for c in test_classes])
		time1 = time.time()

		went_fine = 0

		for i,score in enumerate(fisher_score(test_array,lda,num_topics,V,D_test)):
			c = clf.predict(score) 
			if c[0] == test_classes[i]:
				went_fine += 1

		time2 = time.time()
		precision = went_fine * 100. / D_test
		print 'percent good {}'.format(precision)
		print 'classification time {}'.format(time2 - time1)

		spamwriter.writerow([num_topics, precision, time2 - time1])