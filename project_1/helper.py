from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score,GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import zero_one_loss
import pandas as pd
import numpy as np
from classifiers import KNN, LDA, RandomForest
import matplotlib.pyplot as plt

"""Used for training/test error (no tuning)"""

def Evaluation(model,foldDataTraining,foldDataTest, trainFlag=True):
    numberOfFolds = len(foldDataTraining)
    trainingErrors = []
    testErrors = []

    for k in range(numberOfFolds):
        train = foldDataTraining[k]
        test = foldDataTest[k]

        xTrain = train[:, 1:]
        yTrain = train[:, 0]
        xTest = test[:, 1:]
        yTest = test[:, 0]

        #Fitting to training data
        if trainFlag:
            model.classifier.fit(xTrain,yTrain)

        #Training error
        if trainFlag:
            yPredtraining = model.classifier.predict(xTrain)
            trainingErrorForFold = np.mean(np.where(yPredtraining != yTrain,1,0))
            trainingErrors.append(trainingErrorForFold)

        #Cross validation error
        yPredTest = model.classifier.predict(xTest)
        testErrorForFold = np.mean(np.where(yPredTest != yTest,1,0))
        testErrors.append(testErrorForFold)


    return trainingErrors,testErrors

"""Tuning models using double cross-validation"""
def doubleCV(foldDataTraining, foldDataTest, model, paramGrid):
    numberOfFolds = len(foldDataTraining)
    #For each fold, perform inner CV
    
    trainingErrors = []
    TuningTestErrors = []
    df_CVresults = pd.DataFrame()
    best_models = []

    for k in range(numberOfFolds):

        train = foldDataTraining[k]
        test = foldDataTest[k]

        xTrain = train[:, 1:]
        yTrain = train[:, 0]
        xTest = test[:, 1:]
        yTest = test[:, 0]

        innerCV = KFold(n_splits=5, shuffle = True, random_state=42)

        gridSearch = GridSearchCV(
            estimator= model,
            param_grid=paramGrid,
            cv = innerCV,
            scoring='accuracy',
            return_train_score=True,
            refit=True
        )

        gridSearch.fit(xTrain,yTrain)

        bestModel = gridSearch.best_estimator_
        best_models.append(bestModel)
        grid = pd.DataFrame(gridSearch.cv_results_)
        grid["Outer fold iteration"] = k

        # Mark the best model (row) for this fold
        best_params = gridSearch.best_params_
        grid["Best model"] = grid["params"].apply(lambda p: int(p == best_params))

        df_CVresults = pd.concat([df_CVresults, grid])

        #xTest unbiased (not used for training)
        yPred = bestModel.predict(xTest)

        #Training Error (biased since model trained on xTrain)
        yPredTraining = bestModel.predict(xTrain)
        trainingErrors.append(zero_one_loss(yTrain, yPredTraining))
        
        #Test Error i.e cross validation error after tuning
        TuningTestErrors.append(zero_one_loss(yTest, yPred))
        

    return  trainingErrors, TuningTestErrors, best_models, df_CVresults


"""Trying different hyperparameters -> CV error vs hyperparam plot"""
def CVErrorVSHyperparam(model_class, hyperparam_name, hyperparam_values, foldDataTraining, foldDataTest):

    meanErrors = []
    for val in hyperparam_values:
        kwargs = {hyperparam_name: val}
        classInstance = model_class(**kwargs)
        _, TuningTestErrors, _, _ = Evaluation(classInstance, foldDataTraining, foldDataTest)
        meanErrors.append(np.mean(TuningTestErrors))

    return meanErrors

def evaluate_best_models(models, xTest, yTest):
    performance = {}
    errors = []
    for i, model in enumerate(models):
        yPred = model.predict(xTest)
        error = zero_one_loss(yTest, yPred)
        errors.append(error)
        
        
    performance["Test error"] = errors   
    df = pd.DataFrame(performance)

    return df

