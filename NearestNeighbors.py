import numpy as np
import scipy as sp
from class_utils import load_bow_representation
from sklearn.neighbors import KNeighborsClassifier

class NearestNeighbors:
	def __init__(self, k=1):
		self.mask = None
		self.X = None
		self.y = None
		self.k = k

	def fit(self, X, y):
			"""
			Initializes a 1-NN classifier

			:param X:  (n,d) data numpy array (to be stored as sparse)
			:param y:  (n,) target numpy array

			"""
			# Quick preproccesing: ignore all tokens occurring fewer than 10 times.
			self.mask = np.sum(X, axis=0) > 10
			self.X = X[:, self.mask]
			self.y = y

	def predict(self, X):
		"""
		Predicts a target label for unseen data point.

		:param x:	(m, d) numpy array of m unseen data points

		:returns: 	(m, ) numpy array of integer values for each 
					predicted label, i.e. 1 for spam and 0 for ham.
		"""
		raise Exception("Not implemented in base class!")
		# return np.apply_along_axis(self.__predict_single, axis=1, arr=X)

	def __predict_single(self, x):
		"""
		Predicts a target label for unseen data point.

		:param x:	(d, ) numpy array of one data point

		:returns: 	(int) integer value for the predicted label,
					i.e. 1 for spam and 0 for ham.
		"""
		distances = self.distance(x)
		index_of_closest = np.argmin(distances)
		return self.y[index_of_closest]

	def score(self, x, y):
		"""
		Calculates classifier training or testing accuracy if new data is passed to score

		:param X:  (n,d) data numpy array
		:param y:  (n,) target numpy array
		:return: (double) accuracy
		"""
		res = self.predict(x)
		return np.mean(np.equal(self.predict(x), y))

	def majority(self, neighbors):
		return int(sum(neighbors) / self.k >= 1/2)

class L2NN(NearestNeighbors):
	def predict(self, x):
		"""
		Predicts a target label for unseen data point using L2

		:param x:	(m, d) numpy array of m unseen data points

		:returns: 	(m, ) numpy array of integer values for each 
					predicted label, i.e. 1 for spam and 0 for ham.
		"""
		norm = lambda x1, x2 : (np.power(x1 - x2, 2).sum())
		x = x[:, self.mask]
		pairwise_distances = sp.spatial.distance.cdist(x, self.X, metric=norm)
		min_distance_indices = np.argpartition(pairwise_distances, kth=self.k, axis = 1)[:, :self.k]
		labels = np.array([self.majority([self.y[idx] for idx in row]) for row in min_distance_indices])
		return labels
		# offset_matrix = self.X - x
		# return np.linalg.norm(offset_matrix, ord=2, axis=1)

class L1NN(NearestNeighbors):
	def predict(self, x):
		"""
		Predicts a target label for unseen data point using L1

		:param x:	(m, d) numpy array of m unseen data points

		:returns: 	(m, ) numpy array of integer values for each 
					predicted label, i.e. 1 for spam and 0 for ham.
		"""
		norm = lambda x1, x2 : np.sum(np.abs(x1 - x2))
		x = x[:, self.mask]
		pairwise_distances = sp.spatial.distance.cdist(x, self.X, metric=norm)
		min_distance_indices = np.argpartition(pairwise_distances, kth=self.k, axis = 1)[:, :self.k]
		labels = np.array([self.majority([self.y[idx] for idx in row]) for row in min_distance_indices])
		return labels

class LInfNN(NearestNeighbors):
	def predict(self, x):
		"""
		Predicts a target label for unseen data point using L1

		:param x:	(m, d) numpy array of m unseen data points

		:returns: 	(m, ) numpy array of integer values for each 
					predicted label, i.e. 1 for spam and 0 for ham.
		"""
		norm = lambda x1, x2 : np.max(np.abs(x1 - x2))
		x = x[:, self.mask]
		pairwise_distances = sp.spatial.distance.cdist(x, self.X, metric=norm)
		min_distance_indices = np.argpartition(pairwise_distances, kth=self.k, axis = 1)[:, :self.k]
		labels = np.array([self.majority([self.y[idx] for idx in row]) for row in min_distance_indices])
		return labels

def split_train_test(X, y, train_perc = .5):
    N = X.shape[0]
    
    train_indx = np.random.choice(range(0,N), int(N*train_perc), replace=False)
    test_indx = [i for i in range(0, N) if i not in train_indx]
    
    return X[train_indx,:], X[test_indx,:], y[train_indx],y[test_indx]

def main():
	bow, w2i, i2w = load_bow_representation('bag_of_words.npy', 'word2id', 'id2word')
	X = bow[:, :-1]
	# print(X.shape)
	y = bow[:, -1]
	results = np.zeros((3,4))
	for model_idx, model in enumerate([L1NN(1), L1NN(3), L1NN(5)]):
	        for split_idx, split in enumerate([0.9, 0.75, 0.5, 0.25]):
	                training_data, testing_data, training_labels, testing_labels = split_train_test(X, y, train_perc=split)
	                print("Running Model {} with split {}".format(model_idx, split))
	                model.fit(training_data, training_labels)
	                r = model.score(testing_data, testing_labels)
	                results[model_idx, split_idx] = r
	                print(r)
	np.save("1NNresults.npy", results)

if __name__ == '__main__':
	main()
