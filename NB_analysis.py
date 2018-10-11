
# coding: utf-8

# In[1]:


import numpy as np
from class_utils import *
from DecisionTree import *
import os


# In[6]:


import math


# In[2]:


path = '/Users/Claudia/Desktop/ML_HW1/eron1'
bag_of_words, word2id, id2word = load_bow_representation('bag_of_words.npy', 'word2id','id2word')


# In[3]:


X = bag_of_words[:,range(0, bag_of_words.shape[1]-1)]
y = bag_of_words[:,bag_of_words.shape[1]-1]


# In[4]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,train_size = 4137)


# In[122]:


X.shape


# In[90]:


class NaiveBayes():   
    def __init__(self,x_train, y_train, x_test,y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.trainnum = self.x_train.shape[0]
        self.testnum = self.x_test.shape[0]
        self.dim = self.x_train.shape[1]
    
    def Prior(self,y):
        count = len(np.where(y_train == y)[0])
        prior = count/self.trainnum
        return prior 


    def getindex(self,y):
        index = np.where(y_train == y)[0]
        return index

    def Likelihood(self,y):
        Likelihood_arr = []
        index = self.getindex(y)
        total_each_class = (x_train[index, :].sum() + self.dim)
        for d in range(self.dim):
            count_class_per_word = x_train[index,d].sum() + 1
            likelihood = count_class_per_word/total_each_class
            log_likelihood = math.log10(likelihood)
            Likelihood_arr.append(log_likelihood)
        return Likelihood_arr
            
    def predict(self, x, spam_likelihood, ham_likelihood):
        spam_sum_log_likelihood = 0
        ham_sum_log_likelihood = 0
        for d in range(self.dim):
            if x[d] > 0:
                spam_sum_log_likelihood += x[d]*spam_likelihood[d]
                ham_sum_log_likelihood += x[d]*ham_likelihood[d]
    
        spam_posterior = spam_sum_log_likelihood + math.log10(self.Prior(1))
        ham_posterior = ham_sum_log_likelihood + math.log10(self.Prior(0))
        
        if (spam_posterior == 0) and (ham_posterior == 0):
            p_predict_spam = 0
        else: 
            p_predict_spam = spam_posterior - ham_posterior
       # print(p_predict_spam)
        
        return p_predict_spam
    
    def confusionmatrix(self):
        TP=FP=TN=FN=0
        spam_likelihood = self.Likelihood(1)
        ham_likelihood = self.Likelihood(0)
        for i in range(self.testnum):
            if self.predict(self.x_test[i], spam_likelihood,ham_likelihood) >0:
                if self.y_test[i] == 1:
                    TP += 1
                else:
                    FP += 1
            else: 
                if self.y_test[i] == 0:
                    TN += 1
                else:
                    FN += 1
        return TP,FP,TN,FN
    
    def score(self):
        TP,FP,TN,FN = self.confusionmatrix()
        accuracy = (TP+TN)/(TP+FP+TN+FN)
        recall = (TP)/(TP + FN)
        percision = (TP)/(TP+FP)
        F1 = 2*(percision*recall)/(percision + recall)
        print("accuracy: ", accuracy, " recal: ", recall," percision: ", percision, " F1:", F1)
        return accuracy,recall,percision,F1


# In[91]:


classifier1 = NaiveBayes(x_train,y_train,x_test,y_test)


# In[92]:


classifier1.score()


# In[93]:


def trydiffsplit():
    train_accuracy_arr = []
    test_accuracy_arr = []
    
    train_recall_arr = []
    test_recall_arr = []
    
    train_percision_arr = []
    test_percision_arr = []
    
    train_f1_arr = []
    test_f1_arr = []
    
    for i in [0.1,0.2,0.3,0.4, 0.5, 0.6,0.7,0.8,0.9]:
        x_train0, x_test0, y_train0, y_test0 = train_test_split(X,y,train_size = int(i*(X.shape[0])))

        classifier_test = NaiveBayes(x_train0,y_train0,x_test0,y_test0)
        classifier_train = NaiveBayes(x_train0,y_train0,x_train0,y_train0)
        
        test_accuracy,test_recall,test_percision,test_F1 = classifier_test.score()
        train_accuracy,train_recall,train_percision,train_F1 = classifier_train.score()
        
        test_accuracy_arr.append(test_accuracy)
        train_accuracy_arr.append(train_accuracy)
        
        test_recall_arr.append(test_recall)
        train_recall_arr.append(train_recall)
        
        test_percision_arr.append(test_percision)
        train_percision_arr.append(train_percision)
        
        train_f1_arr.append(train_F1)
        test_f1_arr.append(test_F1)
        
    return  train_accuracy_arr,test_accuracy_arr, train_recall_arr, test_recall_arr, train_percision_arr,test_percision_arr,train_f1_arr,test_f1_arr



# In[94]:


train_accuracy_arr,test_accuracy_arr, train_recall_arr, test_recall_arr, train_percision_arr,test_percision_arr,train_f1_arr,test_f1_arr = trydiffsplit() 


# In[97]:


train_accuracy_arr


# In[116]:


import matplotlib.pyplot as plt
plt.plot(train_accuracy_arr, label="train")
plt.plot(test_accuracy_arr, label="test")
plt.xlabel("split training ")
plt.ylabel("Accuracy")
plt.ylim(0.95,1)
labels = ["10%","20%","30%","40%", "50%", "60%","70%","80%","90%"]
plt.xticks(np.arange(10), ("10%","20%","30%","40%", "50%", "60%","70%","80%","90%"))
plt.legend()
plt.show()


# In[117]:


plt.plot(train_recall_arr, label="train")
plt.plot(test_recall_arr, label="test")
plt.xlabel("split training ")
plt.ylabel("recall")
plt.ylim(0.95,1)
labels = ["10%","20%","30%","40%", "50%", "60%","70%","80%","90%"]
plt.xticks(np.arange(10), ("10%","20%","30%","40%", "50%", "60%","70%","80%","90%"))
plt.legend()
plt.show()


# In[119]:


plt.plot(train_percision_arr, label="train")
plt.plot(test_percision_arr, label="test")
plt.xlabel("split training ")
plt.ylabel("percision")
plt.ylim(0.9,1)
labels = ["10%","20%","30%","40%", "50%", "60%","70%","80%","90%"]
plt.xticks(np.arange(10), ("10%","20%","30%","40%", "50%", "60%","70%","80%","90%"))
plt.legend()
plt.show()

