from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score,GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from KNN import KNN
from KNN import LDA
from KNN import RandomForest
import matplotlib.pyplot as plt

"""Used for training/test error (no tuning)"""

def Evaluation(model,foldDataTraining,foldDataTest):
    numberOfFolds = len(foldDataTraining)
    trainingErrors = []
    testErrors = []
    labels = [-9, -2, 0, 1, 2, 5, 6, 8, 9] #Ensure same matrix dimension for confusion matrix
    totTrainingConfusionMatrix = np.zeros((len(labels),len(labels)))
    totTestConfusionMatrix = np.zeros((len(labels),len(labels)))


    for k in range(numberOfFolds):
        train = foldDataTraining[k]
        test = foldDataTest[k]

        xTrain = train[:, 1:]
        yTrain = train[:, 0]
        xTest = test[:, 1:]
        yTest = test[:, 0]

        #Fitting to training data
        model.classifier.fit(xTrain,yTrain)

        #Training error
        yPredtraining = model.classifier.predict(xTrain)
        trainingErrorForFold = np.mean(np.where(yPredtraining != yTrain,1,0))
        trainingErrors.append(trainingErrorForFold)

        #Cross validation error
        yPredTest = model.classifier.predict(xTest)
        testErrorForFold = np.mean(np.where(yPredTest != yTest,1,0))
        testErrors.append(testErrorForFold)

        #Confusion matrix
        confusionMatrixTraining = confusion_matrix(yTrain,yPredtraining,labels=labels)
        confusionMatrixTest = confusion_matrix(yTest,yPredTest,labels=labels)

        totTrainingConfusionMatrix += confusionMatrixTraining
        totTestConfusionMatrix += confusionMatrixTest

    #Get total confusion matrix (normalize over folds?)
    totTrainingConfusionMatrix /= numberOfFolds
    totTestConfusionMatrix /= numberOfFolds

    return trainingErrors,testErrors,totTrainingConfusionMatrix,totTestConfusionMatrix

"""Tuning models using double cross-validation"""
def doubleCV(foldDataTraining, foldDataTest, model, paramGrid):
    numberOfFolds = len(foldDataTraining)
    #For each fold, perform inner CV
    outerScores = [] #Accuracy, essentially same as 1-testerror... (can remove)
    trainingErrors = []
    testErrors = []
    data_grid = []

    for k in range(numberOfFolds):

        train = foldDataTraining[k]
        test = foldDataTest[k]

        xTrain = train[:, 1:]
        yTrain = train[:, 0]
        xTest = test[:, 1:]
        yTest = test[:, 0]

        innerCV = KFold(n_splits=3, shuffle = True, random_state=42)

        gridSearch = GridSearchCV(
            estimator= model,
            param_grid=paramGrid,
            cv = innerCV,
            scoring='accuracy'
        )

        gridSearch.fit(xTrain,yTrain)

        bestModel = gridSearch.best_estimator_
        data_grid.append(gridSearch.cv_results_)
        #xTest unbiased (not used for training)
        yPred = bestModel.predict(xTest)

        #Training Error (biased since model trained on xTrain)
        yPredTraining = bestModel.predict(xTrain)

        trainingErrorForFold = np.mean(np.where(yPredTraining != yTrain,1,0))
        trainingErrors.append(trainingErrorForFold)

        #Test Error i.e cross validation error after tuning
        testErrorForFold = np.mean(np.where(yPred != yTest,1,0))
        testErrors.append(testErrorForFold)

    return outerScores, trainingErrors, testErrors, bestModel, data_grid


"""Trying different hyperparameters -> CV error vs hyperparam plot"""
def CVErrorVSHyperparam(model_class, hyperparam_name, hyperparam_values, foldDataTraining, foldDataTest):

    meanErrors = []
    for val in hyperparam_values:
        kwargs = {hyperparam_name: val}
        classInstance = model_class(**kwargs)
        _, testErrors, _, _ = Evaluation(classInstance, foldDataTraining, foldDataTest)
        meanErrors.append(np.mean(testErrors))

    return meanErrors
