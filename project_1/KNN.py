from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
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

    def score(self, X, y):
        return self.classifier.score(X, y)
    
    def Error(self,foldDataTraining,foldDataTest):
        #k = np.shape(foldData)[2]
        trainingerrors = []
        testErrors = []
        for k in range(len(foldDataTraining[0,0,:])):
            xTrain = foldDataTraining[:,1:,k]
            yTrain = foldDataTraining[:,0,k]

            self.model.fit(xTrain,yTrain)
            yPred = self.model.predict(xTrain)

            Trainingerrors = np.mean(np.where(yPred != yTrain,1,0))
        
        #for k in range(len(foldDataTest[0,0,:])):
        
    
        #sklearn.metrics.confusion_matrix(y_true, y_pred, *, labels=None, sample_weight=None, normalize=None)