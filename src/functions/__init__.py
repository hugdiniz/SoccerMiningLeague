import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition

import warnings
import sys
from random import shuffle
from math import sqrt
import math
import decimal

#Sklearn -- Classification
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

#Sklearn -- Metrics
from sklearn.metrics import accuracy_score, confusion_matrix

#all == choose all session with not trainning for test. nextStage == choose only next stage to test
def errorForSession(inputsSession,outputSession,functionCreateMetric,date,inputBeforeSession =[],outputBeforeSession =[], weightsFunction = -1,init=-1,test="all"):
    uniqueDate = np.sort(np.unique(date))
    pos = init
    
    if init == -1:
        index = 0
        pos = uniqueDate[index]
    else:
        index = np.where(pos == uniqueDate)[0]

    scores = []
    while(pos < uniqueDate[-1]):
        indexMatrixTrain = np.where(pos >= date)[0]
        if(test == "all"):
            indexMatrixTest = np.where(pos < date)[0]
        else:
            indexMatrixTest = np.where(uniqueDate[index+1] == date)[0]

        clf = functionCreateMetric()

        if(len(inputBeforeSession) > 0):            
            inputClf = np.concatenate((inputBeforeSession,inputsSession[indexMatrixTrain,:]),axis=0)
            outputClf =  np.concatenate((outputBeforeSession,outputSession[indexMatrixTrain]),axis=0)
        else:            
            inputClf = inputsSession[indexMatrixTrain,:]
            outputClf = outputSession[indexMatrixTrain]

        if(weightsFunction != -1):
            weights = weightsFunction(inputClf)
            clf.fit(inputClf,outputClf,weights)
        else:  
            clf.fit(inputClf,outputClf)

        predicts = clf.predict(inputsSession[indexMatrixTest])
        score = accuracy_score(outputSession[indexMatrixTest],predicts)
        scores = np.concatenate((scores,[score]),axis=0)
       
        pos = uniqueDate[index+1]
        index = index + 1

    return scores

def holdoutPlotConfusionMatrix(clf,inputs,outputs,percent=80):

    size = len(outputs)
    lista = list(range(size))
    sizePercent = int(np.floor((percent * size) / 100))    
    testInputsIndex = np.asarray(lista)[range(sizePercent,size)]
    trainInputsIndex = np.asarray(lista)[range(sizePercent)]

    inputsTrain =  np.array([inputs[y,:] for y in trainInputsIndex])       
    outputsTrain = np.array([outputs[y] for y in trainInputsIndex],dtype="int32")

    testInputs = np.array([inputs[y,:]  for y in testInputsIndex])
    testOutputs = np.array([outputs[y] for y in testInputsIndex],dtype="int32")

    clf.fit(inputsTrain, outputsTrain)
    y_predict = clf.predict(testInputs)
    plotConfusionMatrix(testOutputs, y_predict)




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


def defineTeamWin(dataHome,dataAway):
    result = np.asarray([],dtype=int)   
    for x in range(dataHome.size):
        homeGoals = round(decimal.Decimal(dataHome[x]),4)
        awayGoals = round(decimal.Decimal(dataAway[x]),4)
        if homeGoals > awayGoals:
            result = np.concatenate((result,[1]))
        elif homeGoals == awayGoals:
            result = np.concatenate((result,[0]))
        else:
            result = np.concatenate((result,[-1]))

    return result



def KFoldRC(metric,inputs,outputs,k=10,printMatrixConfusion = False,printPredictionForTeam = False):

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
        outputsTrain = np.array([outputs[y,:] for y in treinoIndex],dtype="int32")
       
        clf1 = svm.SVR() 
        clf1.fit(inputsTrain,outputsTrain[:,0])

        clf2 = svm.SVR()   
        clf2.fit(inputsTrain,outputsTrain[:,1])       
                        
        testOutputs = np.array([outputs[y,:] for y in testeIndex]) 
        testInputs = np.array([inputs[y,:]  for y in testeIndex])

        predicts1 = clf1.predict(testInputs)
        predicts2 = clf2.predict(testInputs)
        

        testOutputsResult = defineTeamWin(testOutputs[:,0],testOutputs[:,1])
        predictsResult = defineTeamWin(predicts1,predicts2)

        scores[fold] = accuracy_score(testOutputsResult,predictsResult)
        
        if printMatrixConfusion:
            matrixConfusion = confusion_matrix(testOutputs, predicts)
            print(matrixConfusion)
   
    print(" = Score:%2.2e[+/- %2.2e]"%(np.mean(scores),np.std(scores)))
    
    return np.mean(scores)


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

def plotConfusionMatrix(y, y_predict):
    #print("Plotting graphic : ",arquivo)
    classes = ['Derrota','Empate','Vitória']
    
    cm = confusion_matrix(y, y_predict)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    
    width, height = cm.shape
    for i in range(width):
        for j in range(height):
            ax.annotate(str(cm[i][j]), xy=(j, i), 
                        horizontalalignment='center',
                        verticalalignment='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('Y')
    plt.xlabel('Prediction')
    #plt.savefig(arquivo)
    plt.show()
    #return plt