import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition

import warnings
import sys
from random import shuffle
from math import sqrt
import math

#Sklearn -- Classification
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

#Sklearn -- Metrics
from sklearn.metrics import accuracy_score, confusion_matrix


def plotPCASpectrum(data):
	pca = decomposition.PCA()
	pca.fit(data)
	plt.figure(1, figsize=(4, 3))
	plt.clf()
	plt.axes([.2, .2, .7, .7])
	plt.plot(pca.explained_variance_, linewidth=2)
	plt.axis('tight')
	plt.xlabel('n_components')
	plt.ylabel('explained_variance_')
	plt.show()


def KFoldSVM(inputs,outputs,metric,k=10,printMatrixConfusion = False):

    lista = list(range(1,len(inputs))) 
    shuffle(lista)
    a = np.asarray(lista) % k   
    
    scores = np.zeros(k)
    scoresT = np.zeros(k)
    recall = np.zeros(k)
    average_precision = np.zeros(k)
    
    for fold in range(0,k):
        treinoIndex = np.where(a != fold)[0]
        testeIndex =  np.where(a == fold)[0]        
        
        inputsTrain =  np.array([inputs[y,:] for y in treinoIndex])       
        outputsTrain = np.array([outputs[y] for y in treinoIndex],dtype="int32")
       
        clf = svm.SVC(kernel=metric[1], C=metric[2], gamma=metric[3])    
        clf.fit(inputsTrain,outputsTrain)       
                        
        testOutputs = [outputs[y] for y in testeIndex] 
        testInputs = np.array([inputs[y,:]  for y in testeIndex])

        predicts = clf.predict(testInputs)       
        scores[fold] = accuracy_score(testOutputs, predicts)
        
        if printMatrixConfusion:
        	matrixConfusion = confusion_matrix(testOutputs, predicts)
        	print(matrixConfusion)
   
    print(metric[0] + " = Score:%2.2e[+/- %2.2e]"%(np.mean(scores),np.std(scores)))
    
    return np.mean(scores)

def KFoldNB(inputs,outputs,k = 8):
   
    lista = list(range(1,len(inputs))) 
    shuffle(lista)
    a = np.asarray(lista) % k   
    
    scores = np.zeros(k)
    
    for fold in range(0,k):
        treinoIndex = np.where(a != fold)[0]
        testeIndex =  np.where(a == fold)[0]        
        
        inputsTrain =  np.array([inputs[y,:] for y in treinoIndex])       
        outputsTrain = np.array([outputs[y] for y in treinoIndex],dtype="int32")
       
        clf = GaussianNB()
        
        clf.fit(inputsTrain, outputsTrain)
        
        testOutputs = [outputs[y] for y in testeIndex] 
        testInputs = np.array([inputs[y,:]  for y in testeIndex])   
        predicts = clf.predict(testInputs)
        scores[fold] = accuracy_score(testOutputs, predicts)
        
    print("Naive Bayes:" + " = Score:%2.2e[+/- %2.2e]"%(np.mean(scores),np.std(scores)))
    
    return np.mean(scores),np.std(scores)