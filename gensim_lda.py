from gensim import corpora, models, similarities
import linecache
import numpy as np
import itertools


def load_bow_from_file(file_name):
	""" BoW creation """
	bow = list()
	V = int(linecache.getline(file_name, 2))
	with open(file_name, 'r') as input_file:
		for _ in xrange(3):
		    next(input_file)
		for line in input_file:
			doc, word, rec = map(int,line.split())
			try:
				d = bow[doc -1]
			except:
				d = list()
				bow.append(d)
			d.append((word,float(rec)))
	return bow,V

def load_bow_from_array(array):
	""" BoW creation """
	bow = [list() for _ in range(array.shape[0])]
	c_array = array.tocoo()
	for i,j,v in itertools.izip(c_array.row, c_array.col, c_array.data):
		d = bow[i]
		d.append((j,float(v)))
	return bow

def list_to_array(i_list,V):
	""" BoW creation """
	o_array = np.zeros((1,V),dtype=np.float32)
	# print i_list
	for v,r in i_list:
		o_array[0,v] = r
	return o_array

def phis_to_array(lda,K,V):
	phis = lda.show_topics(K,topn=V,formatted=False)
	output = np.zeros((V,K),dtype=np.float16)
	for t,dist in enumerate(phis):
		for p,v in dist:
			# print "t : {} p : {} v : {}".format(t,p,v)
			output[v,t] = p
	return output

def thetas_to_array(lda,bow,K):
	D = len(bow)
	thetas = lda[bow]
	output = np.zeros((D,K),dtype=np.float16)
	for d,dist in enumerate(thetas):
		for t,p in dist:
			output[d,t] = p
	return output 

def gensim_lda(corpus,num_topics,p,c):
	bow = load_bow_from_array(corpus)
	lda = models.LdaModel(bow, num_topics=num_topics,passes=p,chunksize=c)
	return lda,bow