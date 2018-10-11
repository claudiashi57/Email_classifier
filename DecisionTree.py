import numpy as np
from collections import Counter


def _gini(y):
    classes, counts = np.unique(y, return_counts=True)

    return 1 - sum([(c / len(y)) ** 2 for c in counts])


def _split(X, y, var, value):
    """
    Splits data according to var where var <= value

    :param X:  (n,d) data numpy array
    :param y:  (n,) target numpy array
    :param var: integer index of variable to split on
    :param value: value of variable to split on

    :returns X_left: data where var<=value
    :returns X_right: data where var>value
    :returns y_lef: target where var<=value
    :returns y_right: target where var>value

    """

    true_indx = X[:, var] <= value

    # split data
    X_left = X[true_indx, :]
    X_right = X[~true_indx, :]

    # split target variable
    y_left = y[true_indx]
    y_right = y[~true_indx]

    # percent in each side
    p_left = len(y_left) / len(y)
    p_right = len(y_right) / len(y)

    return X_left, X_right, y_left, y_right


def _gain(X, y, var, value):
    """
    Splits data according to var where var <= value and calculates gain

    :param X:  (n,d) data numpy array
    :param y:  (n,) target numpy array
    :param var: integer index of variable to split on
    :param value: value of variable to split on

    :returns gain: difference between gini of parent and weighted children

    """
    true_indx = X[:, var] <= value

    # split target variable
    y_left = y[true_indx]
    y_right = y[~true_indx]

    # percent in each side
    p_left = len(y_left) / len(y)
    p_right = len(y_right) / len(y)

    return _gini(y) - (p_left * _gini(y_left) + p_right * _gini(y_right)), len(np.unique(y_right))


def _best_split(X, y):
    """
    Iterates over all variables of X and unique values y to find maximum gain

    :param X  (n,d) numpy array
    :param y  (n,) numpy array

    :returns maximumum gain, variable index, value to split (<=)

    """
    # Store all splits
    splits = {}
    vars = range(0, X.shape[1])

    # Loop over all variables
    for var in vars:

        var_values = np.unique(X[:, var])

        # Loop over all unique values
        for v in var_values:

            split_gain, n_classes_right = _gain(X, y, var, v)
            splits[split_gain] = (var, v)

            if n_classes_right == 1:
                # Since we are looping over all unique values of y increasing
                # If there is only one class with y values greater than the current value,
                # we don't need to check anymore splits.
                break

    max_gain = np.max(list(splits.keys()))

    return max_gain, splits[max_gain][0], splits[max_gain][1]


class Node():

    def __init__(self, X, y, id2word, depth, max_depth = 4, min_split_samples = 10):

        self.depth = depth
        self.max_depth = max_depth
        self.min_split_samples = min_split_samples
        self.children = {}
        self.leaf = False
        self.id2word = id2word

        if len(y)==0:
            print('Empty!!')
        elif (len(np.unique(y)) == 1 or depth == max_depth or len(y) < min_split_samples):
            #print('Found a leaf!')
            self.leaf = True
            self.set(y)
        else:
            self.fit(X, y)

    def fit(self, X, y):

        """
        Finds optimal split given (X, y) and builds a left and right child each with the data
        that satisfies the splitting condition.

        :param X:  (n,d) data numpy array
        :param y:  (n,) target numpy array

        """

        self.max_gain, self.var, self.value = _best_split(X, y)

        X_left, X_right, y_left, y_right = _split(X, y, self.var,self.value)

        #print('Splitting on {}<={}'.format(self.id2word[self.var],self.value))

        self.children = {'L': Node(X_left, y_left,depth = self.depth+1 , max_depth=self.max_depth, id2word=self.id2word, min_split_samples=self.min_split_samples),
                         'R': Node(X_right, y_right,depth = self.depth+1, max_depth=self.max_depth,id2word=self.id2word, min_split_samples=self.min_split_samples)}

    def set(self, y):

        """
        Sets the prediction value for a node (only called for leaves)

        :param y:  (n,) target numpy array
        """

        self.leaf_prediction = Counter(y).most_common(1)[0][0]

    def predict(self, x):

        """
        Predicts a single data observation by sending it down to the proper leaf and returning the majority class.

        :param x: (1, d) data numpy array
        :return: prediction of node if a leaf, otherwise returns the prediction from the correct child
        """
        if self.leaf:
            return self.leaf_prediction
        else:
            if x[self.var]<=self.value:
                return self.children['L'].predict(x)
            else:
                return self.children['R'].predict(x)

    def print(self):

        """
        Prints the splits of a node and all children.
        """
        if self.leaf:
            print('{} Predict {}'.format(''.join(['\t' for d in range(0, self.depth-1)]),self.leaf_prediction))
        else:
            print('{}{}<={}'.format(''.join(['\t' for d in range(0, self.depth-1)]), self.id2word[self.var],self.value))
            self.children['L'].print()
            print('{}{}>{}'.format(''.join(['\t' for d in range(0, self.depth - 1)]), self.id2word[self.var],
                                    self.value))
            self.children['R'].print()

class DecisionTree():

    def __init__(self, id2word, max_depth = 6, min_split_samples = 10):

        self.X = None
        self.y = None
        self.id2word = id2word
        self.max_depth = max_depth
        self.min_split_samples = min_split_samples
        self.tree = None

    def fit(self, X, y):

        """
        Initializes a tree with Node at depth = 1

        :param X:  (n,d) data numpy array
        :param y:  (n,) target numpy array

        """
        self.X = X
        self.y = y
        self.tree = Node(X, y, depth = 0, id2word = self.id2word, max_depth=self.max_depth, min_split_samples=self.min_split_samples)

    def predict(self, X):
        """
        Initializes a tree with Node at depth = 1

        :param X:  (n,d) data numpy array
        :returns: (n,) numpy array of predictions
        """

        return np.array([self.tree.predict(X[i,:]) for i in range(0, X.shape[0])])

    def score(self, X = None, y = None):

        """
        Calculates classifier training or testing accuracy if new data is passed to score

        :param X:  (n,d) data numpy array
        :param y:  (n,) target numpy array
        :return: (double) accuracy
        """

        if (X is None) or (y is None):
            return np.mean(self.predict(self.X) == self.y)
        else:
            return np.mean(self.predict(X) == y)

    def print(self):

        """
        Prints the splits of the tree.
        """

        self.tree.print()


def main():

    from class_utils import load_bow_representation
    import os
    #Path to data
    path = '/Users/Sam/Desktop/School/Machine Learning/HW1/hw1code'
    bag_of_words, word2id, id2word = load_bow_representation(os.path.join(path, 'bag_of_words.npy'),
                                                             os.path.join(path, 'word2id'),
                                                             os.path.join(path, 'id2word'))

    X = bag_of_words[:, range(0, bag_of_words.shape[1] - 1)]
    y = bag_of_words[:, bag_of_words.shape[1] - 1]
    N = X.shape[0]

    DT = DecisionTree(id2word, max_depth=3)
    DT.fit(X, y)
    DT.print()

if __name__ == '__main__':
	main()
