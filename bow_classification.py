import os
import time
import random

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

root = "./datasets/2classes/"

files = {path:[os.path.join(path, name) for name in files] for path, subdirs, files in os.walk(root)}


train_data = list()
test_data = list()
train_classes = list()
test_classes = list()

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
print 'number of training documents : {}'.format(D_train)
D_test = len(test_data)
print 'number of test documents : {}'.format(D_test)

"""
	vocabulary
"""
vectorizer = CountVectorizer(input='filename',decode_error='replace',stop_words='english',min_df=0.001,max_df=1.)
vectorizer.fit(data)
V = len(vectorizer.vocabulary_)
print "number of words of the vocabulary : {}".format(V)

"""
	BoWs arrays
"""
train_array = vectorizer.transform(train_data)
test_array = vectorizer.transform(test_data)

"""
	SVM : documents classification
"""
classes = np.array([c for c in train_classes])
unique_classes = np.unique(classes)

print 'classes {}'.format(unique_classes)
clf = SGDClassifier(shuffle=True)
time1 = time.time()
clf.fit(train_array, classes)
time2 = time.time()
print 'svm time {}'.format(time2 - time1)

"""
	SVM : documents prediction
"""
classes = np.array([c for c in test_classes])
time3 = time.time()
classes_found = clf.predict(test_array)
time2 = time.time()
print 'svm time {}'.format(time2 - time3)

well_detected = sum(classes == classes_found)
precision = well_detected * 100. / D_test

print
print 'classification time {}'.format(time2 - time1)
print 'precision {}'.format(precision)