from sklearn.neighbors import KNeighborsClassifier
#from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

class KNN:
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='euclidean'):
        self.classifier = KNeighborsClassifier(
            n_neighbors= n_neighbors,
            weights= weights,
            algorithm= algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric
        )

    def fit(self, X, y):
        self.classifier.fit(X, y)
        return self

    def predict(self, X):
        return self.classifier.predict(X)

    #def score(self, X, y):
        #return self.classifier.score(X, y)
    
    def Evaluation(self,foldDataTraining,foldDataTest):
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
            self.classifier.fit(xTrain,yTrain)

            #Training error
            yPredtraining = self.classifier.predict(xTrain)
            trainingErrorForFold = np.mean(np.where(yPredtraining != yTrain,1,0))
            trainingErrors.append(trainingErrorForFold)

            #Cross validation error
            yPredTest = self.classifier.predict(xTest)
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
    