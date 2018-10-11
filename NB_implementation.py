import numpy as np
from class_utils import *
"""This is the algorithm implementation of """

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
