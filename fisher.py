from sklearn.preprocessing import normalize
import scipy.sparse as spa

import numexpr as ne
import numpy as np

from gensim_lda import load_bow_from_array, thetas_to_array, phis_to_array, list_to_array

def generator(bow,thetas,D,V):
	for d,b in zip(np.split(thetas,D),bow):
		yield d,list_to_array(b,V)

def fisher_score(bow_array,lda_model,K,V,D):
	bow_list = load_bow_from_array(bow_array)
	thetas = thetas_to_array(lda_model,bow_list,K)
	# print 'shape thetas {}'.format(thetas.shape)
	phis = phis_to_array(lda_model,K,V)
	# print 'shape phis {}'.format(phis.shape)
	# print 'thetas {}, {}'.format(thetas.min(),thetas.max())
	# print 'phis {}, {}'.format(phis.astype(np.float16).min(),phis.astype(np.float16).max())
	for i,(t,n) in enumerate(generator(bow_list,thetas,D,V)):
		# compute responsabilities for the current document
		resp = ne.evaluate("phis * t")
		resp = spa.dok_matrix(resp)
		resp_normalized = normalize(X=resp.astype(np.float), norm='l1', axis=1)

		diff = resp_normalized - t
		alphas = np.sum(n * diff,axis=0)

		id_n = np.eye(V) * n
		resp = id_n * resp_normalized
		# print 'resp {}'.format(resp.shape)
		sum_resp = np.sum(resp,axis=0)
		# print 'sum resp {}'.format(sum_resp.shape)
		bethas = resp - phis * sum_resp

		alphas = np.reshape(alphas,(1,-1))
		bethas = np.reshape(bethas,(1,-1))

		yield np.concatenate((alphas.astype(np.float32),bethas.astype(np.float32)),axis=1)

def estimate_information(bow_array,lda_model,K,V,D):
	inf_matrix = np.zeros((K + K * V,K + K * V),dtype=np.float32)
	for i,score in enumerate(fisher_score(bow_array,lda_model,K,V,D)):
		# score = spa.dok_matrix(score)
		new = np.dot(score.T, score)
		inf_matrix += new
	return inf_matrix / D