
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from itertools import product
import numpy.random as rand
import os
from class_utils import load_bow_representation
from DecisionTree import DecisionTree

path = '/Users/Sam/Desktop/School/Machine Learning/HW1/hw1code'
bag_of_words, word2id, id2word = load_bow_representation(os.path.join(path, 'bag_of_words.npy'),
                                                         os.path.join(path, 'word2id'),
                                                         os.path.join(path, 'id2word'))

X = bag_of_words[:,range(0, bag_of_words.shape[1]-1)]
y = bag_of_words[:,bag_of_words.shape[1]-1]
N = X.shape[0]


# In[2]:


#Split training and testing
def split_train_test(X, y, train_perc = .8):
    N = X.shape[0]
    
    train_indx = rand.choice(range(0,N), int(N*train_perc), replace=True)
    test_indx = [i for i in range(0, N) if i not in train_indx]
    
    return X[train_indx,:], X[test_indx,:], y[train_indx],y[test_indx]


# In[3]:


# Split into k folds 
def split_k_folds(X, y, k = 4):
    
    fold_indx = []
    indx = range(0,N)
    rem = N%k
    for i in range(0, k):
        fold_indx.append(rand.choice(indx, N//k + int(i<k), replace=True))
        indx = [x for x in indx if x not in fold_indx[i]]
        
    return fold_indx


# In[10]:


depths = np.linspace(2,26,13)
folds = 4
fold_ids = split_k_folds(X, y, k= 4)
cv_acc = pd.DataFrame(list(product(depths, range(0,folds), [0],[0])), columns = ['depth','fold','train_acc','test_acc'])
trees = {d:[] for d in depths}

for i in range(0, len(depths)):
    
    print("Depth: {}".format(depths[i]))
    DT = DecisionTree( id2word, max_depth=depths[i])
    
    for j in range(0, folds):
        test_indx = fold_ids[j]
        train_indx = np.array([indx for fold in (fold_ids[0:j] + fold_ids[(j+1):folds]) for indx in fold])
    
        X_train = X[train_indx,:]
        y_train = y[train_indx]
        X_test = X[test_indx,:]
        y_test = y[test_indx]
    
        print("Fitting decision tree...")
        DT.fit(X_train, y_train)
        trees[depths[i]] = DT
        
        train_acc = DT.score()
        test_acc = DT.score(X_test, y_test)
        print('\tFold: {}, Accuracy:{}'.format(j, DT.score(X_test, y_test)))
        cv_acc.loc[(cv_acc.depth==depths[i]) & (cv_acc.fold == j),'train_acc'] = train_acc
        cv_acc.loc[(cv_acc.depth==depths[i]) & (cv_acc.fold == j),'test_acc'] = test_acc


# ### Plot of Train and Test Accuracy by Depth

# In[95]:


(cv_acc.groupby('depth')[['train_acc','test_acc']]
 .mean().reset_index()
 .plot(kind = 'line',x = 'depth',y=['train_acc','test_acc'], title = 'Test vs Train Acc (Decision Trees)'))


# In[23]:


import pickle
with open('cv/decision_tree_cv.pickle', 'wb') as f:
    pickle.dump(cv_acc, f,pickle.HIGHEST_PROTOCOL)


# In[14]:


with open('cv/decision_tree_cv.pickle', 'rb') as f:
    cv_acc = pickle.load(f)


# ### Test different training sizes (10%,25%,50%)
# 
# After testing the depth parameter with k fold cross validation, we see that the test error is converging around ~94 to 95. Lets take an arbitrary but depth 4 (to save computation time) and compare performance on test data with varying training sizes. 

# In[90]:


rounds = 4
depth = 6
train_sizes = [.25,.5,.75,.9]
train_size_acc = pd.DataFrame(list(product(range(0,rounds),train_sizes, [0],[0])), columns = ['round','train_size','train_acc','test_acc'])

for ts in train_sizes:
    
    print("Training size: {}".format(ts))
    DT = DecisionTree( id2word, max_depth=depth)
    
    for r in range(0, rounds):
        
        X_train, X_test, y_train, y_test = split_train_test(X, y, train_perc = ts)
        
        print('Fitting trees....')
        DT.fit(X_train, y_train)
        
        train_acc = DT.score()
        test_acc = DT.score(X_test, y_test)
        print('\tRound: {}, Accuracy:{}'.format(r, DT.score(X_test, y_test)))
        train_size_acc.loc[(train_size_acc.train_size==ts) & (train_size_acc['round'] == r),'train_acc'] = train_acc
        train_size_acc.loc[(train_size_acc.train_size==ts) & (train_size_acc['round'] == r),'test_acc'] = test_acc

    
    


# In[93]:


import pickle
with open('cv/decision_tree_train_sizes.pickle', 'wb') as f:
    pickle.dump(train_size_acc, f,pickle.HIGHEST_PROTOCOL)


# In[94]:


(train_size_acc.groupby('train_size')[['train_acc','test_acc']]
 .mean().reset_index()
 .plot(kind = 'line',x = 'train_size',y=['train_acc','test_acc'], title = 'Test vs Train Acc (Decision Trees)'))

